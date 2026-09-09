# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the worker's poll loop, its await decision table, and env layering.

Each loop exit path has a consequence a live fleet makes expensive to discover: a worker that
misses a drain request keeps costing capacity, one that abandons work in flight leaves a task
assigned until the service times it out, and one that reports results at the wrong moment loses
them entirely.

The await decision table is the whole extension this sample adds, so every branch of it is
covered here rather than left to a deployment to find.

Run from the parent directory with:

    python3 -m unittest discover -s tests

Only the steps that would call AWS or run a session are replaced. The loop, the decision table,
the provider registry, and the `sleep` provider all run for real.
"""

from __future__ import annotations

import unittest
import unittest.mock as mock

from harness import (
    JOB_ID,
    QUEUE_ID,
    WORKER_ID,
    FakeDurableContext,
    action,
    await_token,
    env_enter_action,
    env_exit_action,
    poll_response,
    session_outcome,
    session_with,
    stub_session_runner,
    stub_worker,
    task_run_action,
)

import durable_worker

QUIET_HEARTBEAT = {
    "workerDeleted": False,
    "desiredWorkerStatus": None,
    "cancelSessionActions": {},
}


def _task_run_session(action_id: str = "sessionaction-1", **kwargs) -> dict:
    return session_with([task_run_action(action_id, **kwargs)])


class _WorkerLoopTestCase(unittest.TestCase):
    """Runs `lambda_handler` against a scripted sequence of schedule responses."""

    def _run_loop(self, polls: list[dict], *, max_idle_polls: int = 20, outcomes=None):
        poll_calls: list[dict] = []
        scripted = list(polls)

        def fake_poll_schedule(worker_id, updated_session_actions):
            poll_calls.append(
                {"workerId": worker_id, "updates": dict(updated_session_actions)}
            )
            if scripted:
                return scripted.pop(0)
            # A loop that outruns its script would otherwise spin to MAX_LOOP_ITERATIONS.
            return poll_response(desiredWorkerStatus="STOPPED")

        deregister = mock.MagicMock(return_value={"deregistered": True})
        context = FakeDurableContext()

        with stub_session_runner(outcomes=outcomes), mock.patch.object(
            durable_worker, "register_worker", lambda host_name: {"workerId": WORKER_ID}
        ), mock.patch.object(
            durable_worker, "poll_schedule", fake_poll_schedule
        ), mock.patch.object(
            durable_worker, "deregister_worker", deregister
        ), mock.patch.object(
            durable_worker, "heartbeat", lambda worker_id, progress: QUIET_HEARTBEAT
        ), mock.patch.object(
            durable_worker, "DeadlineWorker", return_value=stub_worker()
        ), mock.patch.object(
            durable_worker, "MAX_IDLE_POLLS", max_idle_polls
        ):
            result = durable_worker.lambda_handler({"hostName": "test-host"}, context)

        return result, poll_calls, deregister, context


class TestWorkerDeleted(_WorkerLoopTestCase):
    def test_reports_the_deletion_and_skips_deregistration(self):
        result, _, deregister, _ = self._run_loop([poll_response(workerDeleted=True)])
        self.assertEqual(result["stopReason"], "worker-deleted-by-service")
        # No valid worker ID is left to drive STOPPING/STOPPED/DeleteWorker with.
        deregister.assert_not_called()


class TestServiceRequestedStop(_WorkerLoopTestCase):
    def test_desired_worker_status_stopped_ends_the_loop_cleanly(self):
        result, _, deregister, _ = self._run_loop(
            [poll_response(desiredWorkerStatus="STOPPED")]
        )
        self.assertEqual(result["stopReason"], "service-requested-stop")
        deregister.assert_called_once_with(WORKER_ID)


class TestDrainRequested(_WorkerLoopTestCase):
    """Scale-in for a customer-managed fleet arrives as a registry drain flag."""

    def test_drain_flag_ends_the_loop(self):
        result, _, deregister, _ = self._run_loop([poll_response(drainRequested=True)])
        self.assertEqual(result["stopReason"], "scale-in-drain")
        deregister.assert_called_once_with(WORKER_ID)

    def test_work_assigned_in_the_draining_poll_is_still_finished(self):
        # Regression guard. Breaking out before running the work this poll assigned abandons a
        # task: it stays assigned until the service times it out.
        result, _, _, _ = self._run_loop(
            [poll_response(drainRequested=True, assignedSessions=_task_run_session())]
        )
        self.assertEqual(result["stopReason"], "scale-in-drain")
        self.assertEqual(result["tasksCompleted"], 1)

    def test_the_drained_worker_reports_its_last_result_before_leaving(self):
        _, poll_calls, _, _ = self._run_loop(
            [poll_response(drainRequested=True, assignedSessions=_task_run_session())]
        )
        # UpdateWorkerSchedule is the only way to report the result, so it rides out on an
        # extra call after the loop has ended.
        self.assertEqual(len(poll_calls), 2)
        self.assertEqual(
            poll_calls[1]["updates"]["sessionaction-1"]["completedStatus"], "SUCCEEDED"
        )


class TestRequestGaveUpRetrying(_WorkerLoopTestCase):
    """A request that spent its whole retry budget without landing.

    The distinction that matters: nothing was reported and nothing was learned, so waiting
    unbilled and asking again is the remedy, not failing the work.
    """

    def test_a_poll_that_gave_up_waits_and_asks_again(self):
        result, poll_calls, deregister, context = self._run_loop(
            [
                poll_response(retryLater=True, updateIntervalSeconds=30),
                poll_response(assignedSessions=_task_run_session()),
            ]
        )
        self.assertEqual(context.waits[0], 30)
        self.assertEqual(result["tasksCompleted"], 1)
        deregister.assert_called_once_with(WORKER_ID)

    def test_a_worker_that_can_never_reach_the_service_still_gives_up(self):
        # Otherwise it would spin to MAX_LOOP_ITERATIONS holding fleet capacity.
        result, _, _, _ = self._run_loop(
            [poll_response(retryLater=True) for _ in range(5)], max_idle_polls=3
        )
        self.assertEqual(result["stopReason"], "idle-timeout")

    def test_a_result_that_never_landed_holds_back_the_next_one(self):
        # The service rejects an out-of-order report and drops every result in the request,
        # so the second action's result has to wait for the first to be accepted.
        session = session_with(
            [task_run_action("sessionaction-1"), task_run_action("sessionaction-2")]
        )
        _, poll_calls, _, _ = self._run_loop(
            [
                poll_response(assignedSessions=session),
                poll_response(retryLater=True),
                poll_response(desiredWorkerStatus="STOPPED"),
            ]
        )
        reported = [call["updates"] for call in poll_calls if call["updates"]]
        self.assertEqual([sorted(update) for update in reported], [["sessionaction-1"]])

    def test_an_action_that_never_started_is_not_reported_at_all(self):
        with mock.patch.object(
            durable_worker,
            "run_action",
            lambda *args: {"state": "RETRY_LATER", "message": "gave up"},
        ):
            _, poll_calls, _, _ = self._run_loop(
                [poll_response(assignedSessions=_task_run_session())]
            )
        # Leaving it unreported is what lets the service assign it again.
        self.assertEqual([call["updates"] for call in poll_calls if call["updates"]], [])


class TestIdleBehavior(_WorkerLoopTestCase):
    def test_repeated_empty_polls_end_in_an_idle_timeout(self):
        result, poll_calls, deregister, _ = self._run_loop(
            [poll_response() for _ in range(5)], max_idle_polls=3
        )
        self.assertEqual(result["stopReason"], "idle-timeout")
        self.assertEqual(len(poll_calls), 3)
        deregister.assert_called_once_with(WORKER_ID)

    def test_the_worker_sleeps_for_the_interval_the_service_asked_for(self):
        _, _, _, context = self._run_loop(
            [
                poll_response(updateIntervalSeconds=30),
                poll_response(updateIntervalSeconds=45),
                poll_response(desiredWorkerStatus="STOPPED"),
            ],
            max_idle_polls=10,
        )
        # Drifting from the requested cadence is what gets a worker marked NOT_RESPONDING.
        self.assertEqual(context.waits, [30, 45])

    def test_finding_work_resets_the_idle_count(self):
        result, _, _, _ = self._run_loop(
            [
                poll_response(),
                poll_response(assignedSessions=_task_run_session()),
                poll_response(),
                poll_response(),
                poll_response(desiredWorkerStatus="STOPPED"),
            ],
            max_idle_polls=3,
        )
        self.assertEqual(result["stopReason"], "service-requested-stop")
        self.assertEqual(result["tasksCompleted"], 1)


class TestProgressReporting(_WorkerLoopTestCase):
    def test_a_completed_action_is_reported_with_the_fields_the_service_requires(self):
        result, poll_calls, _, _ = self._run_loop(
            [
                poll_response(assignedSessions=_task_run_session()),
                poll_response(desiredWorkerStatus="STOPPED"),
            ]
        )
        self.assertEqual(poll_calls[0]["updates"], {})
        reported = [call for call in poll_calls if "sessionaction-1" in call["updates"]]
        self.assertEqual(len(reported), 1)
        update = reported[0]["updates"]["sessionaction-1"]
        self.assertEqual(update["completedStatus"], "SUCCEEDED")
        self.assertEqual(update["processExitCode"], 0)
        # The service rejects a completed action that has no startedAt.
        self.assertTrue(update["startedAt"])
        self.assertTrue(update["endedAt"])
        self.assertEqual(result["tasksCompleted"], 1)

    def test_a_reported_result_is_not_sent_twice(self):
        # A second completion for the same action is a protocol error.
        _, poll_calls, _, _ = self._run_loop(
            [
                poll_response(assignedSessions=_task_run_session()),
                poll_response(),
                poll_response(desiredWorkerStatus="STOPPED"),
            ],
            max_idle_polls=10,
        )
        carrying = [call for call in poll_calls if "sessionaction-1" in call["updates"]]
        self.assertEqual(len(carrying), 1)


class _ActionTestCase(unittest.TestCase):
    """Drives one session action through the real decision table."""

    def _run_action(
        self,
        action_dict,
        *,
        outcomes=None,
        env_layers=None,
        statuses=None,
        heartbeat_results=None,
    ):
        self.beats: list[dict] = []
        self.polls: list[dict] = []
        layers = env_layers if env_layers is not None else []

        def fake_heartbeat(worker_id, progress):
            self.beats.append(progress)
            if heartbeat_results:
                return heartbeat_results[
                    min(len(self.beats) - 1, len(heartbeat_results) - 1)
                ]
            return QUIET_HEARTBEAT

        def fake_poll_await(provider_name, handle):
            status = (statuses or [{"state": "SUCCEEDED", "observedAt": "T2"}])[
                min(len(self.polls), len(statuses or [1]) - 1)
            ]
            self.polls.append(status)
            return status

        context = FakeDurableContext()
        with stub_session_runner(outcomes=outcomes) as runner, mock.patch.object(
            durable_worker, "DeadlineWorker", return_value=stub_worker()
        ), mock.patch.object(
            durable_worker, "heartbeat", fake_heartbeat
        ), mock.patch.object(
            durable_worker, "poll_await", fake_poll_await
        ):
            result = durable_worker._run_session_action(
                context=context,
                worker_id=WORKER_ID,
                session_id="session-1",
                queue_id=QUEUE_ID,
                job_id=JOB_ID,
                action=action_dict,
                env_layers=layers,
            )
        self.runner = runner
        self.context = context
        self.layers = layers
        return result


class TestAwaitDecisionTable(_ActionTestCase):
    """A succeeded action's await tokens decide whether the worker waits or reports.

    The table: failed action reports the failure and ignores any token; succeeded with no token
    reports SUCCEEDED at once; succeeded with one token awaits it; succeeded with more than one
    is an error.
    """

    def test_a_succeeded_action_with_no_token_reports_succeeded_immediately(self):
        result = self._run_action(task_run_action(), outcomes=[session_outcome()])
        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        # No await means no wait and no provider poll: this is an ordinary OpenJD task.
        self.assertEqual(self.context.waits, [])
        self.assertEqual(self.polls, [])

    def test_a_succeeded_action_with_one_token_is_awaited_unbilled(self):
        result = self._run_action(
            task_run_action(),
            outcomes=[session_outcome(tokens=[await_token("sleep", {"finishAt": 0})])],
            statuses=[{"state": "SUCCEEDED", "observedAt": "T2", "message": "done"}],
        )
        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        # The wait is what makes a long request cheap: it suspends the execution.
        self.assertIn(durable_worker.TASK_POLL_SECONDS, self.context.waits)
        self.assertEqual(len(self.polls), 1)

    def test_the_awaited_outcome_decides_the_result_not_the_action_exit_code(self):
        result = self._run_action(
            task_run_action(),
            outcomes=[session_outcome(exit_code=0, tokens=[await_token()])],
            statuses=[{"state": "FAILED", "observedAt": "T2", "message": "model said no"}],
        )
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("model said no", result["progressMessage"])

    def test_more_than_one_token_fails_the_action_with_a_clear_message(self):
        result = self._run_action(
            task_run_action(),
            outcomes=[session_outcome(tokens=[await_token(), await_token()])],
        )
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("2", result["progressMessage"])
        self.assertIn(durable_worker.action_output.AWAIT_PREFIX, result["progressMessage"])
        # Nothing is awaited, so no request is paid for on a template the worker cannot honor.
        self.assertEqual(self.context.waits, [])

    def test_a_failed_action_reports_the_failure_and_ignores_its_token(self):
        # Its request may well be running, but a failed action's output is not a promise.
        result = self._run_action(
            task_run_action(),
            outcomes=[
                session_outcome(
                    state="FAILED", exit_code=1, message="bad prompt", tokens=[await_token()]
                )
            ],
        )
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("bad prompt", result["progressMessage"])
        self.assertEqual(self.polls, [])


class TestActionStateMapping(_ActionTestCase):
    def test_each_action_state_maps_the_way_the_agent_maps_it(self):
        expected = {
            "SUCCESS": "SUCCEEDED",
            "FAILED": "FAILED",
            "CANCELED": "CANCELED",
            # The service has no separate status for a timeout.
            "TIMEOUT": "FAILED",
        }
        for state, completed_status in expected.items():
            with self.subTest(state=state):
                result = self._run_action(
                    task_run_action(), outcomes=[session_outcome(state=state)]
                )
                self.assertEqual(result["completedStatus"], completed_status)

    def test_a_state_with_no_message_still_says_something(self):
        result = self._run_action(
            task_run_action(), outcomes=[session_outcome(state="TIMEOUT", message="")]
        )
        self.assertIn("TIMEOUT", result["progressMessage"])

    def test_a_progress_message_is_truncated_to_the_service_limit(self):
        # A longer one fails the whole UpdateWorkerSchedule request, losing every result in it.
        result = self._run_action(
            task_run_action(),
            outcomes=[session_outcome(state="FAILED", message="x" * 9000)],
        )
        self.assertEqual(len(result["progressMessage"]), durable_worker.MAX_PROGRESS_MESSAGE)


class TestEnvironmentLayering(_ActionTestCase):
    """A queue environment's variables have to reach actions that run in later invocations."""

    def test_entering_records_the_environments_delta_as_its_own_layer(self):
        result = self._run_action(
            env_enter_action(environment_id="env-1"),
            outcomes=[session_outcome(env_set={"CONDA_PREFIX": "/opt/conda"})],
        )
        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        self.assertEqual(self.layers, [["env-1", {"set": {"CONDA_PREFIX": "/opt/conda"}, "unset": []}]])

    def test_a_failed_entry_records_no_layer(self):
        self._run_action(
            env_enter_action(), outcomes=[session_outcome(state="FAILED", message="no conda")]
        )
        self.assertEqual(self.layers, [])

    def test_layers_are_kept_in_entry_order(self):
        layers: list = []
        self._run_action(
            env_enter_action("a1", environment_id="env-1"),
            outcomes=[session_outcome(env_set={"TOOL": "/one"})],
            env_layers=layers,
        )
        self._run_action(
            env_enter_action("a2", environment_id="env-2"),
            outcomes=[session_outcome(env_set={"TOOL": "/two"})],
            env_layers=layers,
        )
        self.assertEqual([layer[0] for layer in layers], ["env-1", "env-2"])

    def test_a_later_action_runs_with_the_layers_composed_in_order(self):
        layers = [
            ["env-1", {"set": {"TOOL": "/one"}, "unset": []}],
            ["env-2", {"set": {"TOOL": "/two"}, "unset": []}],
        ]
        # DeadlineWorker is a MagicMock here, so run_action reaches the stubbed runner and the
        # composed environment is visible in the call it made.
        self._run_action(task_run_action(), env_layers=layers)
        self.assertEqual(self.runner.calls[0]["os_env_vars"]["TOOL"], "/two")

    def test_exiting_un_layers_only_that_environment(self):
        layers = [
            ["env-1", {"set": {"TOOL": "/one"}, "unset": []}],
            ["env-2", {"set": {"TOOL": "/two"}, "unset": []}],
        ]
        self._run_action(env_exit_action(environment_id="env-2"), env_layers=layers)
        # Merged layers could not do this: env-1's value has to come back.
        self.assertEqual([layer[0] for layer in layers], ["env-1"])

    def test_a_failed_exit_still_un_layers_because_the_environment_is_being_left(self):
        layers = [["env-1", {"set": {"TOOL": "/one"}, "unset": []}]]
        self._run_action(
            env_exit_action(environment_id="env-1"),
            outcomes=[session_outcome(state="FAILED", message="teardown blew up")],
            env_layers=layers,
        )
        self.assertEqual(layers, [])

    def test_an_unset_from_an_environment_reaches_a_later_action_as_a_removal(self):
        layers: list = []
        self._run_action(
            env_enter_action("a1"),
            outcomes=[session_outcome(env_unset=["PYTHONHOME"])],
            env_layers=layers,
        )
        self._run_action(task_run_action("a2"), env_layers=layers)
        self.assertIsNone(self.runner.calls[0]["os_env_vars"]["PYTHONHOME"])


class TestSessionActionTypes(_ActionTestCase):
    """Each action kind is handled on its own terms, not acknowledged wholesale."""

    def test_a_task_run_reaches_the_session_runner_as_a_task_run(self):
        self._run_action(task_run_action())
        self.assertEqual(self.runner.calls[0]["kind"], "taskRun")

    def test_an_environment_enter_reaches_the_runner_with_its_identifier(self):
        self._run_action(env_enter_action(environment_id="env-7"))
        self.assertEqual(self.runner.calls[0]["kind"], "envEnter")
        self.assertEqual(self.runner.calls[0]["environment_id"], "env-7")

    def test_an_environment_exit_reaches_the_runner_so_on_exit_actually_runs(self):
        # Previously this was acknowledged without running anything.
        self._run_action(env_exit_action(environment_id="env-7"))
        self.assertEqual(self.runner.calls[0]["kind"], "envExit")

    def test_job_attachments_fail_rather_than_silently_staging_nothing(self):
        result = self._run_action(action("a1", {"syncInputJobAttachments": {"stepId": "s"}}))
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("job attachments", result["progressMessage"])
        # Nothing is run, so the failure costs no session.
        self.assertEqual(self.runner.calls, [])

    def test_an_unknown_action_type_fails_rather_than_reporting_success(self):
        result = self._run_action(action("a1", {"somethingNew": {}}))
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("Unsupported", result["progressMessage"])


class TestHeartbeatWhileAwaiting(_ActionTestCase):
    """A worker awaiting a request must keep heartbeating or the service declares it dead.

    A durable execution is single-threaded, so the wait loop has to heartbeat itself.
    """

    def _await(self, *, statuses, heartbeat_results=None):
        return self._run_action(
            task_run_action(),
            outcomes=[session_outcome(tokens=[await_token()])],
            statuses=statuses,
            heartbeat_results=heartbeat_results,
        )

    def test_every_provider_poll_also_heartbeats(self):
        running = {"state": "RUNNING", "message": "still going", "observedAt": "T2"}
        done = {"state": "SUCCEEDED", "outputUri": "s3://b/k", "observedAt": "T3"}
        result = self._await(statuses=[running, running, done])
        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        self.assertEqual(len(self.beats), 3)

    def test_the_heartbeat_reports_progress_without_completing_the_action(self):
        self._await(statuses=[{"state": "SUCCEEDED", "observedAt": "T3"}])
        progress = self.beats[0]["sessionaction-1"]
        # No completedStatus: that is what marks the action still running.
        self.assertNotIn("completedStatus", progress)
        self.assertIn("updatedAt", progress)

    def test_a_canceled_action_stops_polling(self):
        canceled = {
            "workerDeleted": False,
            "desiredWorkerStatus": None,
            "cancelSessionActions": {"sessionaction-1": ["sessionaction-1"]},
        }
        result = self._await(
            statuses=[{"state": "RUNNING", "observedAt": "T2"}],
            heartbeat_results=[canceled],
        )
        self.assertEqual(result["completedStatus"], "CANCELED")
        self.assertEqual(len(self.beats), 1)

    def test_a_deleted_worker_abandons_the_action(self):
        deleted = {"workerDeleted": True, "cancelSessionActions": {}}
        result = self._await(
            statuses=[{"state": "RUNNING", "observedAt": "T2"}], heartbeat_results=[deleted]
        )
        # No worker means no way to report a result.
        self.assertEqual(result["completedStatus"], "INTERRUPTED")

    def test_a_request_that_never_finishes_times_out(self):
        with mock.patch.object(durable_worker, "MAX_TASK_POLLS", 2):
            result = self._await(statuses=[{"state": "RUNNING", "observedAt": "T2"}])
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertEqual(len(self.beats), 2)

    def test_an_unknown_provider_fails_one_action_rather_than_the_execution(self):
        # poll_await runs for real here, so the registry's error text is what is reported.
        context = FakeDurableContext()
        with stub_session_runner(
            outcomes=[session_outcome(tokens=[await_token("seedance")])]
        ), mock.patch.object(
            durable_worker, "DeadlineWorker", return_value=stub_worker()
        ), mock.patch.object(
            durable_worker, "heartbeat", lambda worker_id, progress: QUIET_HEARTBEAT
        ):
            result = durable_worker._run_session_action(
                context=context,
                worker_id=WORKER_ID,
                session_id="session-1",
                queue_id=QUEUE_ID,
                job_id=JOB_ID,
                action=task_run_action(),
                env_layers=[],
            )
        self.assertEqual(result["completedStatus"], "FAILED")
        # The message is what a user sees in the monitor, so it has to name the choices.
        self.assertIn("seedance", result["progressMessage"])
        self.assertIn("sleep", result["progressMessage"])


class TestFailedActionStopsTheSession(unittest.TestCase):
    """One unsuccessful action stops the rest of its session, except its envExit."""

    def _run(self, actions, *, results):
        calls = []

        def fake_run(*, context, worker_id, session_id, queue_id, job_id, action, env_layers):
            calls.append(action["sessionActionId"])
            return results[action["sessionActionId"]]

        reported: dict = {}

        def fake_poll_schedule(worker_id, updates):
            reported.update(updates)
            return poll_response()

        with mock.patch.object(
            durable_worker, "_run_session_action", fake_run
        ), mock.patch.object(durable_worker, "poll_schedule", fake_poll_schedule):
            succeeded = durable_worker._finish_assigned_work(
                context=FakeDurableContext(),
                worker_id=WORKER_ID,
                poll=poll_response(assignedSessions=session_with(actions)),
                env_layers={},
            )
        return succeeded, calls, reported

    def test_a_failure_marks_later_actions_never_attempted(self):
        actions = [env_enter_action("a1"), task_run_action("a2"), task_run_action("a3")]
        succeeded, calls, reported = self._run(
            actions,
            results={"a1": {"completedStatus": "FAILED", "startedAt": "T0", "endedAt": "T1"}},
        )
        # The two taskRuns are never attempted, so no session and no request is paid for.
        self.assertEqual(calls, ["a1"])
        self.assertEqual(succeeded, 0)
        self.assertEqual(reported["a2"]["completedStatus"], "NEVER_ATTEMPTED")
        self.assertEqual(reported["a3"]["completedStatus"], "NEVER_ATTEMPTED")

    def test_never_attempted_carries_no_timestamps(self):
        _, _, reported = self._run(
            [task_run_action("a1"), task_run_action("a2")],
            results={"a1": {"completedStatus": "FAILED", "startedAt": "T0", "endedAt": "T1"}},
        )
        # The service rejects a NEVER_ATTEMPTED report that claims to have run.
        self.assertNotIn("startedAt", reported["a2"])
        self.assertNotIn("endedAt", reported["a2"])

    def test_env_exit_still_runs_after_a_failure(self):
        actions = [task_run_action("a1"), env_exit_action("a2")]
        _, calls, reported = self._run(
            actions,
            results={
                "a1": {"completedStatus": "FAILED", "startedAt": "T0", "endedAt": "T1"},
                "a2": {"completedStatus": "SUCCEEDED", "startedAt": "T1", "endedAt": "T2"},
            },
        )
        self.assertEqual(calls, ["a1", "a2"])
        self.assertEqual(reported["a2"]["completedStatus"], "SUCCEEDED")

    def test_each_result_is_reported_separately_and_in_order(self):
        # The service rejects an out-of-order report with "comes in a wrong order", which drops
        # every result in that request, so results cannot be batched.
        actions = [env_enter_action("a0"), task_run_action("a1")]
        batches = []

        def fake_poll_schedule(worker_id, updates):
            batches.append(sorted(updates))
            return poll_response()

        with mock.patch.object(
            durable_worker,
            "_run_session_action",
            lambda **kwargs: {"completedStatus": "SUCCEEDED"},
        ), mock.patch.object(durable_worker, "poll_schedule", fake_poll_schedule):
            durable_worker._finish_assigned_work(
                context=FakeDurableContext(),
                worker_id=WORKER_ID,
                poll=poll_response(assignedSessions=session_with(actions)),
                env_layers={},
            )
        self.assertEqual(batches, [["a0"], ["a1"]])

    def test_all_actions_run_when_none_fail(self):
        succeeded, calls, _ = self._run(
            [task_run_action("a1"), task_run_action("a2")],
            results={
                "a1": {"completedStatus": "SUCCEEDED"},
                "a2": {"completedStatus": "SUCCEEDED"},
            },
        )
        self.assertEqual(calls, ["a1", "a2"])
        self.assertEqual(succeeded, 2)


class TestEnvironmentLayersSpanPolls(unittest.TestCase):
    """An envEnter and the taskRun that needs it usually arrive in different polls."""

    def test_a_layer_recorded_in_one_poll_reaches_an_action_in_the_next(self):
        env_layers: dict = {}
        recorded = []

        def fake_run_session_action(*, env_layers, action, **kwargs):
            recorded.append((action["sessionActionId"], [layer[0] for layer in env_layers]))
            if "envEnter" in action["definition"]:
                env_layers.append(["env-1", {"set": {"TOOL": "/one"}, "unset": []}])
            return {"completedStatus": "SUCCEEDED"}

        with mock.patch.object(
            durable_worker, "_run_session_action", fake_run_session_action
        ), mock.patch.object(
            durable_worker, "poll_schedule", lambda worker_id, updates: poll_response()
        ):
            for actions in ([env_enter_action("a1")], [task_run_action("a2")]):
                durable_worker._finish_assigned_work(
                    context=FakeDurableContext(),
                    worker_id=WORKER_ID,
                    poll=poll_response(assignedSessions=session_with(actions)),
                    env_layers=env_layers,
                )
        self.assertEqual(recorded, [("a1", []), ("a2", ["env-1"])])

    def test_a_session_with_no_layers_left_is_forgotten(self):
        # A worker lives for hours, so per-session state has to be released.
        env_layers: dict = {}
        with mock.patch.object(
            durable_worker,
            "_run_session_action",
            lambda **kwargs: {"completedStatus": "SUCCEEDED"},
        ), mock.patch.object(
            durable_worker, "poll_schedule", lambda worker_id, updates: poll_response()
        ):
            durable_worker._finish_assigned_work(
                context=FakeDurableContext(),
                worker_id=WORKER_ID,
                poll=poll_response(assignedSessions=session_with([task_run_action("a1")])),
                env_layers=env_layers,
            )
        self.assertEqual(env_layers, {})


if __name__ == "__main__":
    unittest.main()
