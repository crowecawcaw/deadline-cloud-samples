# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Shared test harness. Import this before any module from `lambda/`.

The durable execution SDK ships only inside the Lambda runtime, so it is stubbed here:
`durable_step` binds a fake step context instead of checkpointing, which means a call such
as `poll_schedule(worker_id, updates)` runs the real body and `FakeDurableContext.step`
receives its result. Checkpoint replay belongs to Lambda and is not modeled.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import types
from pathlib import Path

from botocore.exceptions import ClientError

LAMBDA_DIR = Path(__file__).resolve().parents[1] / "lambda"
if str(LAMBDA_DIR) not in sys.path:
    sys.path.insert(0, str(LAMBDA_DIR))

os.environ.setdefault("FARM_ID", "farm-" + "0" * 32)
os.environ.setdefault("FLEET_ID", "fleet-" + "0" * 32)
os.environ.setdefault("OUTPUT_BUCKET", "test-bucket")
os.environ.setdefault("REGISTRY_TABLE", "test-registry")
os.environ.setdefault(
    "WORKER_FUNCTION_ARN", "arn:aws:lambda:us-west-2:123456789012:function:durable-worker"
)

FARM_ID = os.environ["FARM_ID"]
FLEET_ID = os.environ["FLEET_ID"]
WORKER_ID = "worker-" + "0" * 32


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
    provider: str = "sleep",
    request: str = '{"seconds": 0}',
) -> dict:
    return action(
        action_id,
        {
            "taskRun": {
                "taskId": "task-1",
                "stepId": "step-1",
                "parameters": {"Provider": provider, "Request": request},
            }
        },
    )


def session_with(actions: list[dict]) -> dict:
    """One assigned session holding the given already-summarized actions."""
    return {
        "session-1": {"queueId": "queue-abc", "jobId": "job-abc", "sessionActions": actions}
    }
