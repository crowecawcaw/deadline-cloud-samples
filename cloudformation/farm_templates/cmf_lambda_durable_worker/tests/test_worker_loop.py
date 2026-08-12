# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the worker's poll loop and its generic async-task driver.

Each exit path has a consequence a live fleet makes expensive to discover: a worker that
misses a drain request keeps costing capacity, one that abandons work in flight leaves a
task assigned until the service times it out, and one that reports results at the wrong
moment loses them entirely.

Run from the parent directory with:

    python3 -m unittest discover -s tests

Only the steps that would call AWS are replaced. The task driver, the provider registry,
and the `sleep` provider run for real.
"""

from __future__ import annotations

import unittest
import unittest.mock as mock

from harness import (
    WORKER_ID,
    FakeDurableContext,
    action,
    poll_response,
    session_with,
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

    def _run_loop(self, polls: list[dict], *, max_idle_polls: int = 20):
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

        with mock.patch.object(
            durable_worker, "register_worker", lambda host_name: {"workerId": WORKER_ID}
        ), mock.patch.object(
            durable_worker, "poll_schedule", fake_poll_schedule
        ), mock.patch.object(
            durable_worker, "deregister_worker", deregister
        ), mock.patch.object(
            durable_worker, "heartbeat", lambda worker_id, progress: QUIET_HEARTBEAT
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
        # Regression guard. Breaking out before running the work this poll assigned
        # abandons a task: the provider request keeps running and the action stays
        # assigned until the service times it out.
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


class TestProviderSeam(_WorkerLoopTestCase):
    """The provider is chosen per task, and the worker knows nothing about the request."""

    def test_the_sleep_provider_drives_a_task_to_succeeded_through_the_real_loop(self):
        result, poll_calls, _, context = self._run_loop(
            [
                poll_response(
                    assignedSessions=_task_run_session(
                        provider="sleep", request='{"seconds": 0, "outputUri": "s3://b/k"}'
                    )
                ),
                poll_response(desiredWorkerStatus="STOPPED"),
            ]
        )
        self.assertEqual(result["tasksCompleted"], 1)
        update = poll_calls[1]["updates"]["sessionaction-1"]
        self.assertEqual(update["completedStatus"], "SUCCEEDED")
        # The waits are what makes a long request cheap: they suspend the execution.
        self.assertIn(durable_worker.TASK_POLL_SECONDS, context.waits)

    def test_an_unknown_provider_fails_one_action_rather_than_the_execution(self):
        result, poll_calls, deregister, _ = self._run_loop(
            [
                poll_response(assignedSessions=_task_run_session(provider="seedance")),
                poll_response(desiredWorkerStatus="STOPPED"),
            ]
        )
        update = poll_calls[1]["updates"]["sessionaction-1"]
        self.assertEqual(update["completedStatus"], "FAILED")
        # The message is what a user sees in the monitor, so it has to name the choices.
        self.assertIn("seedance", update["progressMessage"])
        self.assertIn("sleep", update["progressMessage"])
        # The worker survives and shuts down normally.
        self.assertEqual(result["stopReason"], "service-requested-stop")
        deregister.assert_called_once_with(WORKER_ID)

    def test_a_malformed_request_fails_the_action_without_retrying(self):
        with self.assertLogs("durable-step", level="ERROR"):
            _, poll_calls, _, context = self._run_loop(
                [
                    poll_response(assignedSessions=_task_run_session(request="not json")),
                    poll_response(desiredWorkerStatus="STOPPED"),
                ]
            )
        self.assertEqual(
            poll_calls[1]["updates"]["sessionaction-1"]["completedStatus"], "FAILED"
        )
        self.assertNotIn(durable_worker.SUBMIT_RETRY_SECONDS, context.waits)

    def test_a_provider_failure_is_reported_as_a_failed_task(self):
        _, poll_calls, _, _ = self._run_loop(
            [
                poll_response(
                    assignedSessions=_task_run_session(
                        request='{"seconds": 0, "failMessage": "out of pixels"}'
                    )
                ),
                poll_response(desiredWorkerStatus="STOPPED"),
            ]
        )
        update = poll_calls[1]["updates"]["sessionaction-1"]
        self.assertEqual(update["completedStatus"], "FAILED")
        self.assertIn("out of pixels", update["progressMessage"])


class TestSubmitRetry(unittest.TestCase):
    """A retryable rejection waits behind a durable wait rather than failing the task.

    Waiting is unbilled, which makes it far cheaper than losing the task and re-running it.
    """

    def _run(self, submit_results):
        attempts = []

        def fake_submit(provider_name, request_json, task_id, attempt=0):
            attempts.append(attempt)
            result = dict(submit_results[min(attempt, len(submit_results) - 1)])
            result["submittedAt"] = "T1"
            return result

        context = FakeDurableContext()
        with mock.patch.object(
            durable_worker, "submit_task", fake_submit
        ), mock.patch.object(
            durable_worker,
            "poll_task",
            lambda provider_name, handle: {
                "state": "SUCCEEDED",
                "outputUri": "s3://b/k",
                "observedAt": "T2",
            },
        ), mock.patch.object(
            durable_worker, "heartbeat", lambda worker_id, progress: QUIET_HEARTBEAT
        ):
            result = durable_worker._run_session_action(
                context=context,
                worker_id=WORKER_ID,
                job_id="job-abc",
                action=task_run_action(),
            )
        return result, attempts, context.waits

    def test_a_retryable_rejection_is_retried_after_a_wait(self):
        retryable = {"error": "slow down", "retryable": True}
        accepted = {"handle": "handle-1"}
        result, attempts, waits = self._run([retryable, accepted])

        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        # Each retry is a distinct step, identified by its attempt number, so a replay
        # resubmits instead of returning the first rejection forever.
        self.assertEqual(attempts[:2], [0, 1])
        self.assertIn(durable_worker.SUBMIT_RETRY_SECONDS, waits)

    def test_a_permanent_rejection_fails_without_retrying(self):
        result, attempts, waits = self._run([{"error": "bad prompt", "retryable": False}])
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertEqual(attempts, [0])
        self.assertEqual(waits, [])
        self.assertIn("bad prompt", result["progressMessage"])

    def test_retries_are_bounded(self):
        with mock.patch.object(durable_worker, "MAX_SUBMIT_ATTEMPTS", 3):
            result, attempts, _ = self._run([{"error": "slow down", "retryable": True}])
        self.assertEqual(attempts, [0, 1, 2])
        self.assertEqual(result["completedStatus"], "FAILED")


class TestHeartbeatWhileWorking(unittest.TestCase):
    """A busy worker must keep heartbeating or the service declares it dead.

    A durable execution is single-threaded, so the wait loop has to heartbeat itself.
    """

    def _run(self, *, statuses, heartbeat_results=None):
        beats = []

        def fake_heartbeat(worker_id, progress):
            beats.append(progress)
            if heartbeat_results:
                return heartbeat_results[min(len(beats) - 1, len(heartbeat_results) - 1)]
            return QUIET_HEARTBEAT

        polls = []

        def fake_poll_task(provider_name, handle):
            status = statuses[min(len(polls), len(statuses) - 1)]
            polls.append(status)
            return status

        context = FakeDurableContext()
        with mock.patch.object(
            durable_worker,
            "submit_task",
            lambda provider_name, request_json, task_id, attempt=0: {
                "handle": "handle-1",
                "submittedAt": "T1",
            },
        ), mock.patch.object(
            durable_worker, "poll_task", fake_poll_task
        ), mock.patch.object(
            durable_worker, "heartbeat", fake_heartbeat
        ):
            result = durable_worker._run_session_action(
                context=context,
                worker_id=WORKER_ID,
                job_id="job-abc",
                action=task_run_action(),
            )
        return result, beats

    def test_every_task_poll_also_heartbeats(self):
        running = {"state": "RUNNING", "message": "still going", "observedAt": "T2"}
        done = {"state": "SUCCEEDED", "outputUri": "s3://b/k", "observedAt": "T3"}
        result, beats = self._run(statuses=[running, running, done])
        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        self.assertEqual(len(beats), 3)

    def test_the_heartbeat_reports_progress_without_completing_the_action(self):
        _, beats = self._run(statuses=[{"state": "SUCCEEDED", "observedAt": "T3"}])
        progress = beats[0]["sessionaction-1"]
        # No completedStatus: that is what marks the action still running.
        self.assertNotIn("completedStatus", progress)
        self.assertIn("updatedAt", progress)

    def test_a_canceled_action_stops_polling(self):
        canceled = {
            "workerDeleted": False,
            "desiredWorkerStatus": None,
            "cancelSessionActions": {"sessionaction-1": ["sessionaction-1"]},
        }
        result, beats = self._run(
            statuses=[{"state": "RUNNING", "observedAt": "T2"}],
            heartbeat_results=[canceled],
        )
        self.assertEqual(result["completedStatus"], "CANCELED")
        self.assertEqual(len(beats), 1)

    def test_a_deleted_worker_abandons_the_action(self):
        deleted = {"workerDeleted": True, "cancelSessionActions": {}}
        result, _ = self._run(
            statuses=[{"state": "RUNNING", "observedAt": "T2"}], heartbeat_results=[deleted]
        )
        # No worker means no way to report a result.
        self.assertEqual(result["completedStatus"], "INTERRUPTED")

    def test_a_request_that_never_finishes_times_out(self):
        with mock.patch.object(durable_worker, "MAX_TASK_POLLS", 2):
            result, beats = self._run(statuses=[{"state": "RUNNING", "observedAt": "T2"}])
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertEqual(len(beats), 2)


class TestSessionActionTypes(unittest.TestCase):
    """Each action kind is handled on its own terms, not acknowledged wholesale."""

    def _run(self, definition, *, entered=None):
        with mock.patch.object(
            durable_worker,
            "enter_queue_environment",
            lambda worker_id, job_id, environment_id: entered or {"variables": {}},
        ):
            return durable_worker._run_session_action(
                context=FakeDurableContext(),
                worker_id=WORKER_ID,
                job_id="job-abc",
                action=action("sessionaction-1", definition),
            )

    def test_an_environment_this_worker_can_honor_succeeds(self):
        result = self._run({"envEnter": {"environmentId": "env-1"}})
        self.assertEqual(result["completedStatus"], "SUCCEEDED")

    def test_an_environment_it_cannot_honor_fails_the_action(self):
        # Succeeding would let the task run in an environment that was never prepared.
        result = self._run(
            {"envEnter": {"environmentId": "env-1"}},
            entered={"error": "defines a script, which a Lambda durable worker cannot run"},
        )
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("script", result["progressMessage"])

    def test_env_exit_succeeds_because_there_is_nothing_to_tear_down(self):
        result = self._run({"envExit": {"environmentId": "env-1"}})
        self.assertEqual(result["completedStatus"], "SUCCEEDED")

    def test_job_attachments_fail_rather_than_silently_staging_nothing(self):
        result = self._run({"syncInputJobAttachments": {"stepId": "step-1"}})
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("job attachments", result["progressMessage"])

    def test_an_unknown_action_type_fails_rather_than_reporting_success(self):
        result = self._run({"somethingNew": {}})
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("Unsupported", result["progressMessage"])


class TestFailedActionStopsTheSession(unittest.TestCase):
    """One unsuccessful action stops the rest of its session, except its envExit."""

    def _run(self, actions, *, results):
        calls = []

        def fake_run(*, context, worker_id, job_id, action):
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
            )
        return succeeded, calls, reported

    def test_a_failure_marks_later_actions_never_attempted(self):
        actions = [
            action("a1", {"envEnter": {"environmentId": "env-1"}}),
            task_run_action("a2"),
            task_run_action("a3"),
        ]
        succeeded, calls, reported = self._run(
            actions,
            results={"a1": {"completedStatus": "FAILED", "startedAt": "T0", "endedAt": "T1"}},
        )
        # The two taskRuns are never attempted, so no provider request is paid for.
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
        actions = [
            task_run_action("a1"),
            action("a2", {"envExit": {"environmentId": "env-1"}}),
        ]
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
        # The service rejects an out-of-order report with "comes in a wrong order", which
        # drops every result in that request, so results cannot be batched.
        actions = [action("a0", {"envEnter": {"environmentId": "env-1"}}), task_run_action("a1")]
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


if __name__ == "__main__":
    unittest.main()
