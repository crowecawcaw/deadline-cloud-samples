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
`poll_schedule(worker_id, pending_updates)` in the handler binds its arguments as if the
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
            lambda task_parameters: {
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

    def test_results_ride_out_on_the_following_poll(self):
        result, poll_calls, _, _ = self._run_loop(
            [
                _poll_response(assignedSessions=_task_run_session()),
                _poll_response(desiredWorkerStatus="STOPPED"),
            ]
        )
        # The first poll carries nothing, because nothing has been done yet.
        self.assertEqual(poll_calls[0]["updates"], {})
        # The second carries the finished action: one call, not a separate report.
        update = poll_calls[1]["updates"]["sessionaction-1"]
        self.assertEqual(update["completedStatus"], "SUCCEEDED")
        self.assertEqual(update["processExitCode"], 0)
        # startedAt is required by the service on every completed action.
        self.assertEqual(update["startedAt"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(update["endedAt"], "2026-01-01T00:05:00+00:00")
        self.assertEqual(result["tasksCompleted"], 1)

    def test_a_reported_result_is_not_sent_twice(self):
        # The service treats a second completion for the same action as a protocol
        # error, so pending results have to be cleared once they are handed over.
        _, poll_calls, _, _ = self._run_loop(
            [
                _poll_response(assignedSessions=_task_run_session()),
                _poll_response(),
                _poll_response(desiredWorkerStatus="STOPPED"),
            ],
            max_idle_polls=10,
        )
        self.assertIn("sessionaction-1", poll_calls[1]["updates"])
        self.assertEqual(poll_calls[2]["updates"], {})


if __name__ == "__main__":
    unittest.main()
