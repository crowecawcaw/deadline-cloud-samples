# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Shared test harness. Import this before any module from `lambda/`.

Two seams are stubbed. The durable execution SDK ships only inside the Lambda runtime, so
`durable_step` binds a fake step context instead of checkpointing, which means a call such as
`poll_schedule(worker_id, updates)` runs the real body and `FakeDurableContext.step` receives
its result. Checkpoint replay belongs to Lambda and is not modeled.

`session_runner` is stubbed per test rather than globally, because `durable_worker` imports it
by name on every call. That keeps the real module importable for the integration tests.

The Deadline Cloud protocol layer is not stubbed. `JobEntities` below is the worker agent's
own, driven by scripted BatchGetJobEntity responses, so its batching, its caching, and its
validation are all exercised rather than mocked away.
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

import boto3
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


# -- job entities ----------------------------------------------------------------

# The members BatchGetJobEntity marks required. Spelled out because the worker agent validates
# every entity strictly, which is what makes these fixtures evidence about the real wire shape.


def job_details(**extra) -> dict:
    return {
        "jobDetails": {
            "jobId": JOB_ID,
            "logGroupName": "/aws/deadline/farm-x/queue-y",
            "schemaVersion": "jobtemplate-2023-09",
            **extra,
        }
    }


def step_details(template: Optional[dict] = None, *, step_id: str = "step-1", **extra) -> dict:
    return {
        "stepDetails": {
            "jobId": JOB_ID,
            "stepId": step_id,
            "schemaVersion": "jobtemplate-2023-09",
            "template": template if template is not None else minimal_step_template(),
            "dependencies": [],
            **extra,
        }
    }


def environment_details(template: Optional[dict] = None, *, environment_id: str = "env-1") -> dict:
    return {
        "environmentDetails": {
            "jobId": JOB_ID,
            "environmentId": environment_id,
            "schemaVersion": "jobtemplate-2023-09",
            "template": template if template is not None else minimal_environment_template(),
        }
    }


def minimal_step_template() -> dict:
    return {"name": "Generate", "script": {"actions": {"onRun": {"command": "/bin/true"}}}}


def minimal_environment_template() -> dict:
    # An environment needs at least one of `script` or `variables` to be valid.
    return {"name": "Config", "variables": {"CONFIGURED": "yes"}}


def entity_error(kind: str, code: str, message: str, **fields) -> dict:
    return {kind: {"jobId": JOB_ID, "code": code, "message": message, **fields}}


@functools.lru_cache(maxsize=1)
def deadline_client():
    """A real, unstubbed client, used only for the service model it carries."""
    return boto3.client(
        "deadline", region_name="us-west-2", aws_access_key_id="a", aws_secret_access_key="b"
    )


class ScriptedEntityClient:
    """A `DeadlineClient` that answers BatchGetJobEntity from a list of responses.

    The last response repeats, so a test scripts only the calls it cares about. The service
    model is the installed botocore's, so `JobEntities` still reads the real identifier limit
    from it rather than from anything this class invents.
    """

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []
        self._real_client = deadline_client()

    def batch_get_job_entity(self, *, farmId, fleetId, workerId, identifiers):
        self.calls.append(list(identifiers))
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


class AnyEntityClient(ScriptedEntityClient):
    """A `DeadlineClient` that answers whatever identifier it is given.

    For tests about the loop rather than about a template: every action's entities resolve,
    whichever step or environment it names.
    """

    def __init__(self, *, job_fields: Optional[dict] = None, template: Optional[dict] = None):
        super().__init__([])
        self._job_fields = job_fields or {}
        self._template = template

    def batch_get_job_entity(self, *, farmId, fleetId, workerId, identifiers):
        self.calls.append(list(identifiers))
        entities = []
        for identifier in identifiers:
            ((kind, fields),) = identifier.items()
            if kind == "jobDetails":
                entities.append(job_details(**self._job_fields))
            elif kind == "stepDetails":
                entities.append(step_details(self._template, step_id=fields["stepId"]))
            elif kind == "environmentDetails":
                entities.append(
                    environment_details(
                        self._template or minimal_environment_template(),
                        environment_id=fields["environmentId"],
                    )
                )
        return {"entities": entities, "errors": []}


def job_entities(client, *, job_id: str = JOB_ID):
    """The worker agent's own JobEntities, over one of the clients above."""
    from deadline_worker_agent.sessions.job_entities import JobEntities

    return JobEntities(
        farm_id=FARM_ID,
        fleet_id=FLEET_ID,
        worker_id=WORKER_ID,
        job_id=job_id,
        deadline_client=client,
        windows_credentials_resolver=None,
        job_run_as_user_override=None,
    )


def stub_worker(client=None, *, queue_credentials: Optional[dict] = None):
    """A `DeadlineWorker` whose entity fetching is real and whose API calls are not."""
    worker = mock.MagicMock()
    worker.entity_client = client if client is not None else AnyEntityClient()
    worker.job_entities.side_effect = lambda *, job_id: job_entities(
        worker.entity_client, job_id=job_id
    )
    worker.assume_queue_role.return_value = queue_credentials or {
        "accessKeyId": "AKIAQUEUE",
        "secretAccessKey": "s",
        "sessionToken": "t",
        "expiration": "2026-01-01T00:00:00+00:00",
    }
    return worker
