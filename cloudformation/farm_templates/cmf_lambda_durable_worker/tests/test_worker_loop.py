# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the durable worker's poll loop.

The loop is where the sample decides when a worker lives and dies, and each exit path
has a consequence a live fleet makes expensive to discover: a worker that misses a drain
request keeps costing capacity, one that abandons work in flight leaves a task assigned
until the service times it out, and one that reports results at the wrong moment loses
them entirely. These tests drive the loop through every exit with scripted schedule
responses.

Run from the parent directory with:

    python3 -m unittest discover -s tests

The durable execution SDK is stubbed exactly as in `test_durable_worker.py`, and the
durable steps are replaced with fakes, so no AWS call is made and no wait sleeps.

Why the steps are faked rather than mocked at the boto3 layer
-------------------------------------------------------------
`durable_step` is stubbed to the identity function, so a call such as
`poll_schedule(worker_id, updates)` in the handler binds its arguments as if the
decorator had already supplied `step_context`. The real bodies cannot run under that
stub, so each step is replaced by a fake with the handler's calling convention: the
arguments the handler passes, and no `step_context`. `FakeDurableContext.step` then
receives an already-computed result and returns it, which is what the identity stub
makes `context.step(...)` mean here.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
import unittest.mock as mock
from pathlib import Path

LAMBDA_DIR = Path(__file__).resolve().parents[1] / "lambda"
sys.path.insert(0, str(LAMBDA_DIR))

# Same SDK stub as the sibling test module. `setdefault` keeps whichever module the
# discovery order imports first from clobbering the other's stub.
_sdk = types.ModuleType("aws_durable_execution_sdk_python")
_sdk.DurableContext = object
_sdk.durable_execution = lambda fn: fn
_sdk.durable_step = lambda fn: fn
_sdk_config = types.ModuleType("aws_durable_execution_sdk_python.config")


class _Duration:
    @staticmethod
    def from_seconds(seconds):
        return seconds


_sdk_config.Duration = _Duration
sys.modules.setdefault("aws_durable_execution_sdk_python", _sdk)
sys.modules.setdefault("aws_durable_execution_sdk_python.config", _sdk_config)

# Required at import time by the worker module.
os.environ.setdefault("FARM_ID", "farm-" + "0" * 32)
os.environ.setdefault("FLEET_ID", "fleet-" + "0" * 32)
os.environ.setdefault("OUTPUT_BUCKET", "test-bucket")

import durable_worker  # noqa: E402

WORKER_ID = "worker-" + "0" * 32


class FakeDurableContext:
    """A `DurableContext` that runs the loop straight through, without suspending.

    `step` returns what it was handed because the stubbed `durable_step` has already
    produced the value, and `wait` records the requested duration instead of sleeping so
    a test can assert the worker slept for the interval the service asked for.
    """

    def __init__(self) -> None:
        self.waits: list[int] = []

    def step(self, result):
        return result

    def wait(self, duration) -> None:
        self.waits.append(duration)


def _poll_response(**overrides) -> dict:
    """A `poll_schedule` result with the fields the loop reads."""
    response = {
        "workerDeleted": False,
        "drainRequested": False,
        "updateIntervalSeconds": 15,
        "desiredWorkerStatus": None,
        "assignedSessions": {},
        "cancelSessionActions": {},
    }
    response.update(overrides)
    return response


def _task_run_session(session_action_id: str = "sessionaction-1") -> dict:
    """One assigned session holding a single `taskRun` action, already summarized."""
    return {
        "session-1": {
            "queueId": "queue-abc",
            "jobId": "job-abc",
            "sessionActions": [
                {
                    "sessionActionId": session_action_id,
                    "definition": {
                        "taskRun": {
                            "taskId": "task-1",
                            "stepId": "step-1",
                            "parameters": {"Prompt": "a red car"},
                        }
                    },
                }
            ],
        }
    }


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
            # A loop that outruns its script would otherwise spin to
            # MAX_LOOP_ITERATIONS, so the fallback ends it and the test sees a
            # stopReason that does not match what it asked for.
            return _poll_response(desiredWorkerStatus="STOPPED")

        deregister = mock.MagicMock(return_value={"deregistered": True})
        context = FakeDurableContext()

        with mock.patch.object(
            durable_worker, "register_worker", lambda host_name: {"workerId": WORKER_ID}
        ), mock.patch.object(
            durable_worker, "poll_schedule", fake_poll_schedule
        ), mock.patch.object(
            durable_worker, "deregister_worker", deregister
        ), mock.patch.object(
            durable_worker,
            "mark_action_started",
            lambda session_action_id: "2026-01-01T00:00:00+00:00",
        ), mock.patch.object(
            durable_worker,
            "submit_bedrock_job",
            # Accepts the retry attempt number: each submit retry is a distinct step,
            # so the attempt is part of the step's arguments.
            lambda task_parameters, attempt=0: {
                "invocationArn": "arn:aws:bedrock:us-west-2:123456789012:async-invoke/abc",
                "outputUri": "s3://test-bucket/generated/task-1/",
                "submittedAt": "2026-01-01T00:00:01+00:00",
            },
        ), mock.patch.object(
            durable_worker,
            "check_bedrock_job",
            lambda invocation_arn: {
                "status": "Completed",
                "outputUri": "s3://test-bucket/generated/task-1/",
                "observedAt": "2026-01-01T00:05:00+00:00",
            },
        ), mock.patch.object(
            # A busy worker heartbeats from inside the generation wait loop. Nothing is
            # cancelled or deleted in these tests, so the quiet response is the default.
            durable_worker,
            "heartbeat",
            lambda worker_id, progress: {
                "workerDeleted": False,
                "desiredWorkerStatus": None,
                "cancelSessionActions": {},
            },
        ), mock.patch.object(
            durable_worker, "MAX_IDLE_POLLS", max_idle_polls
        ):
            result = durable_worker.lambda_handler({"hostName": "test-host"}, context)

        return result, poll_calls, deregister, context


class TestWorkerDeleted(_WorkerLoopTestCase):
    """The service deletes workers that stop heartbeating."""

    def test_reports_the_deletion_and_skips_deregistration(self):
        result, _, deregister, _ = self._run_loop([_poll_response(workerDeleted=True)])
        self.assertEqual(result["stopReason"], "worker-deleted-by-service")
        # There is no valid worker ID left to drive STOPPING/STOPPED/DeleteWorker with,
        # so attempting to deregister would only fail the execution on the way out.
        deregister.assert_not_called()


class TestServiceRequestedStop(_WorkerLoopTestCase):
    def test_desired_worker_status_stopped_ends_the_loop_cleanly(self):
        result, _, deregister, _ = self._run_loop(
            [_poll_response(desiredWorkerStatus="STOPPED")]
        )
        self.assertEqual(result["stopReason"], "service-requested-stop")
        # Unlike a deletion, the worker still exists here, so it has to walk itself
        # through STOPPING/STOPPED/DeleteWorker rather than just disappearing.
        deregister.assert_called_once_with(WORKER_ID)


class TestDrainRequested(_WorkerLoopTestCase):
    """Scale-in for a customer-managed fleet arrives as a registry drain flag."""

    def test_drain_flag_ends_the_loop(self):
        result, _, deregister, _ = self._run_loop([_poll_response(drainRequested=True)])
        self.assertEqual(result["stopReason"], "scale-in-drain")
        deregister.assert_called_once_with(WORKER_ID)

    def test_work_assigned_in_the_draining_poll_is_still_finished(self):
        # Regression guard. A drain check that breaks before running the work assigned
        # by the same poll silently abandons a task: the Bedrock request keeps running,
        # and the action stays assigned until the service times it out.
        result, poll_calls, _, _ = self._run_loop(
            [_poll_response(drainRequested=True, assignedSessions=_task_run_session())]
        )
        self.assertEqual(result["stopReason"], "scale-in-drain")
        self.assertEqual(result["tasksCompleted"], 1)

    def test_the_drained_worker_reports_its_last_result_before_leaving(self):
        _, poll_calls, _, _ = self._run_loop(
            [_poll_response(drainRequested=True, assignedSessions=_task_run_session())]
        )
        # The results of that final task ride out on an extra heartbeat, because
        # UpdateWorkerSchedule is the only way to report them and the loop has ended.
        self.assertEqual(len(poll_calls), 2)
        self.assertEqual(
            poll_calls[1]["updates"]["sessionaction-1"]["completedStatus"], "SUCCEEDED"
        )


class TestIdleBehavior(_WorkerLoopTestCase):
    """An idle worker sleeps at no compute cost, then eventually gives up."""

    def test_repeated_empty_polls_end_in_an_idle_timeout(self):
        result, poll_calls, deregister, _ = self._run_loop(
            [_poll_response() for _ in range(5)], max_idle_polls=3
        )
        self.assertEqual(result["stopReason"], "idle-timeout")
        # The backstop exists so an orphaned execution cannot idle for the full
        # one-year execution timeout, so it must stop polling once it trips.
        self.assertEqual(len(poll_calls), 3)
        deregister.assert_called_once_with(WORKER_ID)

    def test_the_worker_sleeps_for_the_interval_the_service_asked_for(self):
        _, _, _, context = self._run_loop(
            [
                _poll_response(updateIntervalSeconds=30),
                _poll_response(updateIntervalSeconds=45),
                _poll_response(desiredWorkerStatus="STOPPED"),
            ],
            max_idle_polls=10,
        )
        # Sleeping for anything else drifts from the service's expected heartbeat
        # cadence, which is what gets a worker marked NOT_RESPONDING and deleted.
        self.assertEqual(context.waits, [30, 45])

    def test_finding_work_resets_the_idle_count(self):
        # Otherwise a worker that is intermittently busy still times out as if idle.
        result, _, _, _ = self._run_loop(
            [
                _poll_response(),
                _poll_response(assignedSessions=_task_run_session()),
                _poll_response(),
                _poll_response(),
                _poll_response(desiredWorkerStatus="STOPPED"),
            ],
            max_idle_polls=3,
        )
        self.assertEqual(result["stopReason"], "service-requested-stop")
        self.assertEqual(result["tasksCompleted"], 1)


class TestProgressReporting(_WorkerLoopTestCase):
    """UpdateWorkerSchedule is both the heartbeat and the progress report."""

    def test_a_completed_action_is_reported_with_the_fields_the_service_requires(self):
        result, poll_calls, _, _ = self._run_loop(
            [
                _poll_response(assignedSessions=_task_run_session()),
                _poll_response(desiredWorkerStatus="STOPPED"),
            ]
        )
        # A poll that asks for work carries no results, because nothing is done yet.
        self.assertEqual(poll_calls[0]["updates"], {})
        # The result goes out in its own call as soon as the action finishes.
        reported = [c for c in poll_calls if "sessionaction-1" in c["updates"]]
        self.assertEqual(len(reported), 1)
        update = reported[0]["updates"]["sessionaction-1"]
        self.assertEqual(update["completedStatus"], "SUCCEEDED")
        self.assertEqual(update["processExitCode"], 0)
        # startedAt is required by the service on every completed action.
        self.assertEqual(update["startedAt"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(update["endedAt"], "2026-01-01T00:05:00+00:00")
        self.assertEqual(result["tasksCompleted"], 1)

    def test_a_reported_result_is_not_sent_twice(self):
        # The service treats a second completion for the same action as a protocol
        # error, so a result must not be resent on a later poll.
        _, poll_calls, _, _ = self._run_loop(
            [
                _poll_response(assignedSessions=_task_run_session()),
                _poll_response(),
                _poll_response(desiredWorkerStatus="STOPPED"),
            ],
            max_idle_polls=10,
        )
        carrying = [c for c in poll_calls if "sessionaction-1" in c["updates"]]
        self.assertEqual(len(carrying), 1)


class TestSubmitThrottleRetry(unittest.TestCase):
    """A throttled submit retries behind a durable wait rather than failing the task.

    Bedrock's per-account concurrency limits for generation models are low enough that
    a fleet scaling out will collide with them. Because the retry sleeps in a durable
    wait, waiting the limit out is unbilled and therefore far cheaper than losing the
    task and re-running it.
    """

    def _run(self, submit_results):
        """Run one session action whose submit returns the given results in order."""
        attempts = []

        def fake_submit(task_parameters, attempt=0):
            attempts.append(attempt)
            return submit_results[min(attempt, len(submit_results) - 1)]

        context = FakeDurableContext()
        with mock.patch.object(
            durable_worker, "mark_action_started", lambda session_action_id: "T0"
        ), mock.patch.object(
            durable_worker, "submit_bedrock_job", fake_submit
        ), mock.patch.object(
            durable_worker,
            "check_bedrock_job",
            lambda invocation_arn: {
                "status": "Completed",
                "outputUri": "s3://b/o/",
                "observedAt": "T2",
            },
        ), mock.patch.object(
            durable_worker,
            "heartbeat",
            lambda worker_id, progress: {
                "workerDeleted": False,
                "desiredWorkerStatus": None,
                "cancelSessionActions": {},
            },
        ):
            action = _task_run_session()["session-1"]["sessionActions"][0]
            result = durable_worker._run_session_action(
                context=context, worker_id=WORKER_ID, job_id="job-abc", action=action
            )
        return result, attempts, context.waits

    def test_a_throttled_submit_is_retried_after_a_wait(self):
        throttled = {"invocationArn": None, "throttled": True, "error": "slow down"}
        ok = {
            "invocationArn": "arn:aws:bedrock:us-west-2:123456789012:async-invoke/abc",
            "outputUri": "s3://b/o/",
            "submittedAt": "T1",
        }
        result, attempts, waits = self._run([throttled, ok])

        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        # Each retry is a distinct step, identified by its attempt number, so a replay
        # re-submits instead of returning the first attempt's throttled result forever.
        self.assertEqual(attempts[:2], [0, 1])
        # The retry slept, and did so for the submit interval rather than the much
        # shorter generation poll interval.
        import bedrock_task

        self.assertIn(bedrock_task.SUBMIT_RETRY_SECONDS, waits)

    def test_a_non_throttle_error_fails_without_retrying(self):
        bad = {"invocationArn": None, "throttled": False, "error": "bad prompt"}
        result, attempts, waits = self._run([bad])

        # A malformed request will never succeed, so retrying it only wastes time.
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertEqual(attempts, [0])
        self.assertEqual(waits, [])
        self.assertIn("bad prompt", result["progressMessage"])


class TestHeartbeatWhileWorking(unittest.TestCase):
    """A busy worker must keep heartbeating or the service declares it dead.

    The real worker agent runs sessions on a thread pool so its main loop can keep
    calling UpdateWorkerSchedule. A durable execution is single-threaded, so the
    generation wait loop has to heartbeat itself; without that the worker goes silent
    for the whole request and the service marks it NOT_RESPONDING.
    """

    def _run(self, *, statuses, heartbeat_results=None):
        beats = []

        def fake_heartbeat(worker_id, progress):
            beats.append(progress)
            if heartbeat_results:
                return heartbeat_results[min(len(beats) - 1, len(heartbeat_results) - 1)]
            return {
                "workerDeleted": False,
                "desiredWorkerStatus": None,
                "cancelSessionActions": {},
            }

        checks = []

        def fake_check(invocation_arn):
            result = statuses[min(len(checks), len(statuses) - 1)]
            checks.append(result)
            return result

        context = FakeDurableContext()
        with mock.patch.object(
            durable_worker, "mark_action_started", lambda session_action_id: "T0"
        ), mock.patch.object(
            durable_worker,
            "submit_bedrock_job",
            lambda task_parameters, attempt=0: {
                "invocationArn": "arn:aws:bedrock:us-west-2:123456789012:async-invoke/a",
                "submittedAt": "T1",
            },
        ), mock.patch.object(
            durable_worker, "check_bedrock_job", fake_check
        ), mock.patch.object(
            durable_worker, "heartbeat", fake_heartbeat
        ):
            action = _task_run_session()["session-1"]["sessionActions"][0]
            result = durable_worker._run_session_action(
                context=context, worker_id=WORKER_ID, job_id="job-abc", action=action
            )
        return result, beats

    def test_every_generation_poll_also_heartbeats(self):
        in_progress = {"status": "InProgress", "observedAt": "T2"}
        done = {"status": "Completed", "outputUri": "s3://b/o/", "observedAt": "T3"}
        result, beats = self._run(statuses=[in_progress, in_progress, done])

        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        # One heartbeat per poll, including the polls where nothing had finished yet.
        self.assertEqual(len(beats), 3)

    def test_the_heartbeat_reports_progress_without_completing_the_action(self):
        _, beats = self._run(
            statuses=[{"status": "Completed", "outputUri": "s3://b/o/", "observedAt": "T3"}]
        )
        progress = beats[0]["sessionaction-1"]
        # No completedStatus: that is what marks the action still running. Sending one
        # here would complete the action while the request was still in flight.
        self.assertNotIn("completedStatus", progress)
        self.assertEqual(progress["startedAt"], "T0")
        self.assertIn("updatedAt", progress)

    def test_a_cancelled_action_stops_polling_and_reports_canceled(self):
        cancelled = {
            "workerDeleted": False,
            "desiredWorkerStatus": None,
            "cancelSessionActions": {"sessionaction-1": ["sessionaction-1"]},
        }
        result, beats = self._run(
            statuses=[{"status": "InProgress", "observedAt": "T2"}],
            heartbeat_results=[cancelled],
        )
        # Finishing work the service has withdrawn wastes Bedrock spend and reports a
        # result nobody is waiting for.
        self.assertEqual(result["completedStatus"], "CANCELED")
        self.assertEqual(len(beats), 1)

    def test_a_deleted_worker_abandons_the_action(self):
        deleted = {"workerDeleted": True, "cancelSessionActions": {}}
        result, _ = self._run(
            statuses=[{"status": "InProgress", "observedAt": "T2"}],
            heartbeat_results=[deleted],
        )
        # No worker means no way to report a result, so the action is interrupted
        # rather than reported as succeeded or failed.
        self.assertEqual(result["completedStatus"], "INTERRUPTED")


def _session_with(actions: list[dict]) -> dict:
    """One assigned session holding the given already-summarized actions."""
    return {"session-1": {"queueId": "queue-abc", "jobId": "job-abc", "sessionActions": actions}}


def _action(action_id: str, definition: dict) -> dict:
    return {"sessionActionId": action_id, "definition": definition}


class TestSessionActionTypes(unittest.TestCase):
    """Each action kind is handled on its own terms, not acknowledged wholesale."""

    def _run(self, definition, *, entered=None):
        context = FakeDurableContext()
        with mock.patch.object(
            durable_worker, "mark_action_started", lambda session_action_id: "T0"
        ), mock.patch.object(
            durable_worker,
            "enter_queue_environment",
            lambda worker_id, job_id, environment_id: entered or {"variables": {}},
        ):
            return durable_worker._run_session_action(
                context=context,
                worker_id=WORKER_ID,
                job_id="job-abc",
                action=_action("sessionaction-1", definition),
            )

    def test_an_environment_this_worker_can_honor_succeeds(self):
        result = self._run({"envEnter": {"environmentId": "env-1"}})
        self.assertEqual(result["completedStatus"], "SUCCEEDED")

    def test_an_environment_it_cannot_honor_fails_the_action(self):
        # Succeeding here would let the task run in an environment that was never
        # prepared, which is exactly the silent failure this replaced.
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
        # The task would otherwise run expecting input files that never arrived.
        result = self._run({"syncInputJobAttachments": {"stepId": "step-1"}})
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("job attachments", result["progressMessage"])

    def test_an_unknown_action_type_fails_rather_than_reporting_success(self):
        result = self._run({"somethingNew": {}})
        self.assertEqual(result["completedStatus"], "FAILED")
        self.assertIn("Unsupported", result["progressMessage"])


class TestFailedActionStopsTheSession(unittest.TestCase):
    """One unsuccessful action stops the rest of its session.

    The service will not run further taskRun, envEnter, or syncInputJobAttachments
    actions in a session once any action in it has failed, so continuing would submit
    Bedrock requests whose results are discarded. envExit still runs, because it is the
    session's cleanup.
    """

    def _run(self, actions, *, results):
        calls = []

        def fake_run(*, context, worker_id, job_id, action):
            calls.append(action["sessionActionId"])
            return results[action["sessionActionId"]]

        # Each result is now reported in its own UpdateWorkerSchedule call, because the
        # service rejects out-of-order batches. Collect what each call carried so the
        # tests can assert on the reported results.
        reported: dict = {}

        def fake_poll_schedule(worker_id, updates):
            reported.update(updates)
            return _poll_response()

        with mock.patch.object(
            durable_worker, "_run_session_action", fake_run
        ), mock.patch.object(durable_worker, "poll_schedule", fake_poll_schedule):
            succeeded = durable_worker._finish_assigned_work(
                context=FakeDurableContext(),
                worker_id=WORKER_ID,
                poll=_poll_response(assignedSessions=_session_with(actions)),
            )
        return succeeded, calls, reported

    def test_a_failure_marks_later_actions_never_attempted(self):
        actions = [
            _action("a1", {"envEnter": {"environmentId": "env-1"}}),
            _action("a2", {"taskRun": {"taskId": "t", "stepId": "s", "parameters": {}}}),
            _action("a3", {"taskRun": {"taskId": "t2", "stepId": "s", "parameters": {}}}),
        ]
        succeeded, calls, pending = self._run(
            actions,
            results={"a1": {"completedStatus": "FAILED", "startedAt": "T0", "endedAt": "T1"}},
        )
        # The two taskRuns are never attempted, so no Bedrock request is paid for.
        self.assertEqual(calls, ["a1"])
        self.assertEqual(succeeded, 0)
        self.assertEqual(pending["a2"]["completedStatus"], "NEVER_ATTEMPTED")
        self.assertEqual(pending["a3"]["completedStatus"], "NEVER_ATTEMPTED")

    def test_never_attempted_carries_no_timestamps(self):
        actions = [
            _action("a1", {"taskRun": {"taskId": "t", "stepId": "s", "parameters": {}}}),
            _action("a2", {"taskRun": {"taskId": "t2", "stepId": "s", "parameters": {}}}),
        ]
        _, _, pending = self._run(
            actions,
            results={"a1": {"completedStatus": "FAILED", "startedAt": "T0", "endedAt": "T1"}},
        )
        # The service rejects a NEVER_ATTEMPTED report that claims to have run.
        self.assertNotIn("startedAt", pending["a2"])
        self.assertNotIn("endedAt", pending["a2"])

    def test_env_exit_still_runs_after_a_failure(self):
        actions = [
            _action("a1", {"taskRun": {"taskId": "t", "stepId": "s", "parameters": {}}}),
            _action("a2", {"envExit": {"environmentId": "env-1"}}),
        ]
        _, calls, pending = self._run(
            actions,
            results={
                "a1": {"completedStatus": "FAILED", "startedAt": "T0", "endedAt": "T1"},
                "a2": {"completedStatus": "SUCCEEDED", "startedAt": "T1", "endedAt": "T2"},
            },
        )
        # Cleanup has to happen even on the failure path.
        self.assertEqual(calls, ["a1", "a2"])
        self.assertEqual(pending["a2"]["completedStatus"], "SUCCEEDED")

    def test_each_result_is_reported_separately_and_in_order(self):
        # The service enforces the assigned order and rejects the whole request if a
        # later action is reported first: "comes in a wrong order". Batching results
        # therefore loses every result in the batch, not just the out-of-order one.
        actions = [
            _action("a0", {"envEnter": {"environmentId": "env-1"}}),
            _action("a1", {"taskRun": {"taskId": "t", "stepId": "s", "parameters": {}}}),
        ]
        batches = []

        def fake_poll_schedule(worker_id, updates):
            batches.append(sorted(updates))
            return _poll_response()

        with mock.patch.object(
            durable_worker,
            "_run_session_action",
            lambda **kw: {"completedStatus": "SUCCEEDED"},
        ), mock.patch.object(durable_worker, "poll_schedule", fake_poll_schedule):
            durable_worker._finish_assigned_work(
                context=FakeDurableContext(),
                worker_id=WORKER_ID,
                poll=_poll_response(assignedSessions=_session_with(actions)),
            )
        # One call per action, each carrying only its own result, in assigned order.
        self.assertEqual(batches, [["a0"], ["a1"]])

    def test_all_actions_run_when_none_fail(self):
        actions = [
            _action("a1", {"taskRun": {"taskId": "t", "stepId": "s", "parameters": {}}}),
            _action("a2", {"taskRun": {"taskId": "t2", "stepId": "s", "parameters": {}}}),
        ]
        succeeded, calls, _ = self._run(
            actions,
            results={
                "a1": {"completedStatus": "SUCCEEDED"},
                "a2": {"completedStatus": "SUCCEEDED"},
            },
        )
        self.assertEqual(calls, ["a1", "a2"])
        self.assertEqual(succeeded, 2)


if __name__ == "__main__":
    unittest.main()
