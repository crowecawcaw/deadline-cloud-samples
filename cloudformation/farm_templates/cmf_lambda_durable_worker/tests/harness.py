# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Shared test harness. Import this before any module from `lambda/`.

Two seams are stubbed. The durable execution SDK ships only inside the Lambda runtime, so
`durable_step` binds a fake step context instead of checkpointing, which means a call such as
`poll_schedule(worker_id, updates)` runs the real body and `FakeDurableContext.step` receives
its result. Checkpoint replay belongs to Lambda and is not modeled.

`session_runner` is stubbed per test rather than globally, because `durable_worker` imports it
by name on every call. That keeps the real module importable for the integration tests.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path
from typing import Any, Iterator, Optional

from botocore.exceptions import ClientError

LAMBDA_DIR = Path(__file__).resolve().parents[1] / "lambda"
if str(LAMBDA_DIR) not in sys.path:
    sys.path.insert(0, str(LAMBDA_DIR))

os.environ.setdefault("FARM_ID", "farm-" + "0" * 32)
os.environ.setdefault("FLEET_ID", "fleet-" + "0" * 32)
os.environ.setdefault("REGISTRY_TABLE", "test-registry")
os.environ.setdefault(
    "WORKER_FUNCTION_ARN", "arn:aws:lambda:us-west-2:123456789012:function:durable-worker"
)

FARM_ID = os.environ["FARM_ID"]
FLEET_ID = os.environ["FLEET_ID"]
WORKER_ID = "worker-" + "0" * 32
QUEUE_ID = "queue-" + "0" * 32
JOB_ID = "job-" + "0" * 32


class FakeStepContext:
    logger = logging.getLogger("durable-step")


class FakeDuration:
    @staticmethod
    def from_seconds(seconds):
        return seconds


class FakeDurableContext:
    """A `DurableContext` that runs the loop straight through without suspending.

    `wait` records the requested duration so a test can assert the worker slept for the
    interval the service asked for.
    """

    def __init__(self) -> None:
        self.waits: list[int] = []

    def step(self, result):
        return result

    def wait(self, duration) -> None:
        self.waits.append(duration)


def _install_sdk_stub() -> None:
    sdk = types.ModuleType("aws_durable_execution_sdk_python")
    sdk.DurableContext = object
    sdk.durable_execution = lambda fn: fn
    sdk.durable_step = lambda fn: functools.partial(fn, FakeStepContext())
    config = types.ModuleType("aws_durable_execution_sdk_python.config")
    config.Duration = FakeDuration
    sys.modules.setdefault("aws_durable_execution_sdk_python", sdk)
    sys.modules.setdefault("aws_durable_execution_sdk_python.config", config)


_install_sdk_stub()


class StubSessionRunnerError(Exception):
    """Stands in for `session_runner.SessionRunnerError`."""


@contextlib.contextmanager
def stub_session_runner(
    *,
    outcomes: Optional[list[dict]] = None,
    raises: Optional[BaseException] = None,
) -> Iterator[types.ModuleType]:
    """Replace the openjd-sessions seam for one test, recording the calls made through it.

    The last outcome repeats, so a test lists only the outcomes it cares about.
    """
    module = types.ModuleType("session_runner")
    module.SessionRunnerError = StubSessionRunnerError  # type: ignore[attr-defined]
    calls: list[dict[str, Any]] = []
    queue = list(outcomes or [session_outcome()])

    def run_action(**kwargs):
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return dict(queue[min(len(calls) - 1, len(queue) - 1)])

    module.run_action = run_action  # type: ignore[attr-defined]
    module.calls = calls  # type: ignore[attr-defined]
    with mock.patch.dict(sys.modules, {"session_runner": module}):
        yield module


def session_outcome(
    *,
    state: str = "SUCCESS",
    exit_code: Optional[int] = 0,
    message: str = "",
    tokens: Optional[list[dict]] = None,
    env_set: Optional[dict] = None,
    env_unset: Optional[list[str]] = None,
    ended_at: str = "T1",
) -> dict:
    """What `session_runner.run_action` hands back, plus the `endedAt` the step adds."""
    return {
        "state": state,
        "exitCode": exit_code,
        "message": message,
        "progress": None,
        "awaitTokens": list(tokens or []),
        "envDelta": {"set": dict(env_set or {}), "unset": list(env_unset or [])},
        "endedAt": ended_at,
    }


def await_token(provider: str = "sleep", handle: Any = "handle-1") -> dict:
    return {"provider": provider, "handle": handle}


def client_error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": f"simulated {code}"}}, operation)


def poll_response(**overrides) -> dict:
    """A `poll_schedule` result carrying the fields the loop reads."""
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


def action(action_id: str, definition: dict) -> dict:
    return {"sessionActionId": action_id, "definition": definition}


def task_run_action(
    action_id: str = "sessionaction-1",
    *,
    step_id: str = "step-1",
    parameters: Optional[dict] = None,
) -> dict:
    """A summarized `taskRun` action, with parameters still in their tagged wire form."""
    return action(
        action_id,
        {
            "taskRun": {
                "taskId": "task-1",
                "stepId": step_id,
                "parameters": (
                    {"Prompt": {"string": "a car"}} if parameters is None else parameters
                ),
            }
        },
    )


def env_enter_action(
    action_id: str = "sessionaction-1", *, environment_id: str = "env-1"
) -> dict:
    return action(action_id, {"envEnter": {"environmentId": environment_id}})


def env_exit_action(
    action_id: str = "sessionaction-2", *, environment_id: str = "env-1"
) -> dict:
    return action(action_id, {"envExit": {"environmentId": environment_id}})


def session_with(actions: list[dict], session_id: str = "session-1") -> dict:
    """One assigned session holding the given already-summarized actions."""
    return {session_id: {"queueId": QUEUE_ID, "jobId": JOB_ID, "sessionActions": actions}}
