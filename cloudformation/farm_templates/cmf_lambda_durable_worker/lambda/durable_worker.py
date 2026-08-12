# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A Deadline Cloud customer-managed fleet worker that runs as a Lambda durable function.

Replay contract: everything non-deterministic or side-effecting happens inside a
`context.step()`, control flow depends only on step results, and step identity is
positional, so a step that repeats must vary its arguments to earn its own checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from aws_durable_execution_sdk_python import (  # type: ignore[import-not-found]
    DurableContext,
    durable_execution,
    durable_step,
)
from aws_durable_execution_sdk_python.config import (  # type: ignore[import-not-found]
    Duration,
)

import providers
import queue_environment
import worker_registry
from worker_protocol import (
    DeadlineWorker,
    WorkerNotUsableError,
    WorkerProtocolError,
    default_capabilities,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FARM_ID = os.environ["FARM_ID"]
FLEET_ID = os.environ["FLEET_ID"]
REGION = os.environ.get("AWS_REGION", "us-west-2")

# Retry and poll timing is the worker's policy, not a provider's: every wait below
# suspends the execution, so a longer interval costs nothing and only delays noticing.
TASK_POLL_SECONDS = int(os.environ.get("TASK_POLL_SECONDS", "30"))
MAX_TASK_POLLS = int(os.environ.get("MAX_TASK_POLLS", "120"))
SUBMIT_RETRY_SECONDS = int(os.environ.get("SUBMIT_RETRY_SECONDS", "60"))
MAX_SUBMIT_ATTEMPTS = max(1, int(os.environ.get("MAX_SUBMIT_ATTEMPTS", "10")))

# Backstop so an orphaned execution cannot idle for the stack's whole ExecutionTimeout.
# Raise it for bursty jobs; at the usual 15s interval 20 polls is about five minutes.
MAX_IDLE_POLLS = int(os.environ.get("MAX_IDLE_POLLS", "20"))
MAX_LOOP_ITERATIONS = int(os.environ.get("MAX_LOOP_ITERATIONS", "2000"))

# The service's limit on a progressMessage; a longer one fails the whole request.
MAX_PROGRESS_MESSAGE = 4096


# -- durable steps ---------------------------------------------------------------


@durable_step
def register_worker(step_context, host_name: str) -> dict[str, Any]:
    """Create the worker and move it to STARTED.

    One step, so a replay cannot leave a worker created but never started.
    """
    worker = DeadlineWorker(farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION)
    worker_id = worker.create_worker(host_name=host_name)
    worker.update_worker_status(status="STARTED", capabilities=default_capabilities())
    worker_registry.register(
        fleet_id=FLEET_ID, worker_id=worker_id, started_at=worker_registry.utc_now_iso()
    )
    step_context.logger.info(f"Worker {worker_id} registered and STARTED")
    return {"workerId": worker_id}


@durable_step
def mark_action_started(step_context, session_action_id: str) -> str:
    """Capture the `startedAt` that UpdateWorkerSchedule requires on a completed action.

    Inside a step because a clock read outside one changes on every replay.
    """
    started_at = worker_registry.utc_now_iso()
    step_context.logger.info(f"Starting {session_action_id} at {started_at}")
    return started_at


@durable_step
def heartbeat(step_context, worker_id: str, progress: dict[str, Any]) -> dict[str, Any]:
    """Report progress on a running action without completing it.

    A durable execution is single-threaded, so the wait loop has to heartbeat itself.
    Sending no `completedStatus` is what marks the action still running.
    """
    worker = DeadlineWorker(
        farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION, worker_id=worker_id
    )
    try:
        response = worker.update_worker_schedule(updated_session_actions=progress)
    except WorkerNotUsableError:
        step_context.logger.warning(f"Worker {worker_id} is no longer usable while working")
        return {"workerDeleted": True, "cancelSessionActions": {}}
    return {
        "workerDeleted": False,
        "desiredWorkerStatus": response.get("desiredWorkerStatus"),
        "cancelSessionActions": response.get("cancelSessionActions", {}),
    }


@durable_step
def enter_queue_environment(
    step_context, worker_id: str, job_id: str, environment_id: str
) -> dict[str, Any]:
    """Return the variables a queue environment defines, or why it cannot be honored.

    The error is returned rather than raised so one action fails, not the execution.
    """
    worker = DeadlineWorker(
        farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION, worker_id=worker_id
    )
    try:
        details = worker.get_environment_details(
            job_id=job_id, environment_id=environment_id
        )
        variables = queue_environment.apply(
            environment_details=details, logger_=step_context.logger
        )
        return {"variables": variables}
    except (queue_environment.UnsupportedEnvironmentError, WorkerProtocolError) as exc:
        step_context.logger.error(f"Queue environment {environment_id} failed: {exc}")
        return {"error": str(exc)}


@durable_step
def poll_schedule(
    step_context, worker_id: str, updated_session_actions: dict[str, Any]
) -> dict[str, Any]:
    """Send a heartbeat with any progress, and collect newly assigned work."""
    worker = DeadlineWorker(
        farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION, worker_id=worker_id
    )
    try:
        response = worker.update_worker_schedule(
            updated_session_actions=updated_session_actions
        )
    except WorkerNotUsableError:
        step_context.logger.warning(f"Worker {worker_id} is no longer usable")
        return {"workerDeleted": True, "updateIntervalSeconds": 0, "assignedSessions": {}}

    # Scale-in for a customer-managed fleet is the fleet owner's business, so this flag,
    # not `desiredWorkerStatus`, is what normally ends a worker's life here.
    drain_requested = worker_registry.should_drain(fleet_id=FLEET_ID, worker_id=worker_id)

    return {
        "workerDeleted": False,
        "drainRequested": drain_requested,
        "updateIntervalSeconds": response.get("updateIntervalSeconds", 15),
        "desiredWorkerStatus": response.get("desiredWorkerStatus"),
        "assignedSessions": _summarize_sessions(response.get("assignedSessions", {})),
        "cancelSessionActions": response.get("cancelSessionActions", {}),
    }


@durable_step
def submit_task(
    step_context,
    provider_name: str,
    request_json: str,
    task_id: str,
    attempt: int = 0,
) -> dict[str, Any]:
    """Ask a provider to start its request, returning its handle or an error.

    `attempt` is an argument so each retry is a distinct step; without it a replay would
    return the first attempt's error forever instead of resubmitting.
    """
    try:
        provider = providers.resolve(provider_name)
        request = json.loads(request_json) if request_json else {}
        if not isinstance(request, dict):
            raise ValueError("A Request must be a JSON object.")
        result = provider.submit(request, task_id=task_id)
    except providers.UnknownProviderError as exc:
        result = {"error": str(exc), "retryable": False}
    except Exception as exc:  # A provider defect must cost one task, not the worker.
        step_context.logger.exception(f"Provider {provider_name} failed to submit")
        result = {"error": f"{type(exc).__name__}: {exc}", "retryable": False}
    # Checkpointed so the failure path reports a stable `endedAt` on replay.
    result["submittedAt"] = worker_registry.utc_now_iso()
    return result


@durable_step
def poll_task(step_context, provider_name: str, handle: Any) -> dict[str, Any]:
    """Ask a provider whether its request has finished."""
    try:
        provider = providers.resolve(provider_name)
        result = provider.poll(handle)
    except Exception as exc:
        step_context.logger.exception(f"Provider {provider_name} failed to poll")
        result = {"state": "FAILED", "message": f"{type(exc).__name__}: {exc}"}
    result["observedAt"] = worker_registry.utc_now_iso()
    return result


@durable_step
def deregister_worker(step_context, worker_id: str) -> dict[str, Any]:
    """Drain and deregister: STOPPING, then STOPPED, then DeleteWorker.

    STOPPING tells the service to stop assigning work before the worker disappears.
    """
    worker = DeadlineWorker(
        farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION, worker_id=worker_id
    )
    try:
        worker.update_worker_status(status="STOPPING")
        worker.update_worker_status(status="STOPPED")
        worker.delete_worker()
        step_context.logger.info(f"Worker {worker_id} deregistered")
    except WorkerNotUsableError:
        pass
    # Last, so this worker keeps counting toward fleet capacity until it has stopped.
    worker_registry.deregister(fleet_id=FLEET_ID, worker_id=worker_id)
    return {"deregistered": True}


# -- handler ---------------------------------------------------------------------


@durable_execution
def lambda_handler(event: dict[str, Any], context: DurableContext) -> dict[str, Any]:
    """Run one worker for its whole lifetime as a single durable execution."""
    host_name = event.get("hostName", "durable-lambda-worker")

    registration = context.step(register_worker(host_name))
    worker_id = registration["workerId"]

    idle_polls = 0
    tasks_completed = 0
    stop_reason = "loop-limit-reached"

    for _ in range(MAX_LOOP_ITERATIONS):
        poll = context.step(poll_schedule(worker_id, {}))

        if poll["workerDeleted"]:
            # No valid worker ID left to drain with.
            return {
                "workerId": worker_id,
                "stopReason": "worker-deleted-by-service",
                "tasksCompleted": tasks_completed,
            }

        if poll.get("desiredWorkerStatus") == "STOPPED":
            stop_reason = "service-requested-stop"
            break

        if poll.get("drainRequested"):
            # Work assigned by this same poll is finished first, so draining never
            # abandons a request in flight.
            stop_reason = "scale-in-drain"
            tasks_completed += _finish_assigned_work(
                context=context, worker_id=worker_id, poll=poll
            )
            break

        if not (poll.get("assignedSessions") or {}):
            idle_polls += 1
            if idle_polls >= MAX_IDLE_POLLS:
                stop_reason = "idle-timeout"
                break
            context.wait(Duration.from_seconds(poll["updateIntervalSeconds"]))
            continue

        idle_polls = 0
        tasks_completed += _finish_assigned_work(
            context=context, worker_id=worker_id, poll=poll
        )

    context.step(deregister_worker(worker_id))
    return {
        "workerId": worker_id,
        "stopReason": stop_reason,
        "tasksCompleted": tasks_completed,
    }


def _finish_assigned_work(
    *, context: DurableContext, worker_id: str, poll: dict[str, Any]
) -> int:
    """Run every action this poll assigned, reporting each result as it completes.

    Returns how many actions succeeded.
    """
    succeeded = 0
    for session in (poll.get("assignedSessions") or {}).values():
        session_failed = False
        for action in session["sessionActions"]:
            action_id = action["sessionActionId"]
            is_env_exit = "envExit" in action.get("definition", {})

            if session_failed and not is_env_exit:
                # The service runs no further taskRun, envEnter, or
                # syncInputJobAttachments action in a failed session, and rejects a
                # NEVER_ATTEMPTED report that carries timestamps. envExit still runs.
                result = {
                    "completedStatus": "NEVER_ATTEMPTED",
                    "progressMessage": "An earlier action in this session did not succeed",
                }
            else:
                result = _run_session_action(
                    context=context,
                    worker_id=worker_id,
                    job_id=session["jobId"],
                    action=action,
                )
                if result.get("completedStatus") == "SUCCEEDED":
                    succeeded += 1
                else:
                    session_failed = True

            # One result per call, in assigned order. The service rejects an out-of-order
            # report with "comes in a wrong order" and drops every result in the request.
            context.step(poll_schedule(worker_id, {action_id: result}))
    return succeeded


def _run_session_action(
    *,
    context: DurableContext,
    worker_id: str,
    job_id: str,
    action: dict[str, Any],
) -> dict[str, Any]:
    """Run one assigned session action and return its result for the next heartbeat."""
    definition = action["definition"]
    action_id = action["sessionActionId"]
    started_at = context.step(mark_action_started(action_id))

    if "envEnter" in definition:
        entered = context.step(
            enter_queue_environment(
                worker_id, job_id, definition["envEnter"]["environmentId"]
            )
        )
        if "error" in entered:
            return _action_result(
                "FAILED", started_at, started_at, message=entered["error"]
            )
        return _action_result("SUCCEEDED", started_at, started_at, exit_code=0)

    if "envExit" in definition:
        # Entering only collected variables, and the Lambda sandbox is discarded anyway.
        return _action_result("SUCCEEDED", started_at, started_at, exit_code=0)

    if "syncInputJobAttachments" in definition:
        return _action_result(
            "FAILED",
            started_at,
            started_at,
            message=(
                "This worker does not support job attachments: it has no session "
                "directory to stage input files into. Submit without job attachments, "
                "or use a fleet whose workers have a filesystem."
            ),
        )

    if "taskRun" not in definition:
        return _action_result(
            "FAILED",
            started_at,
            started_at,
            message=f"Unsupported session action type: {sorted(definition)}",
        )

    return _run_provider_task(
        context=context,
        worker_id=worker_id,
        action_id=action_id,
        started_at=started_at,
        task_run=definition["taskRun"],
    )


def _run_provider_task(
    *,
    context: DurableContext,
    worker_id: str,
    action_id: str,
    started_at: str,
    task_run: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a task's request to its provider and wait, unbilled, for the result."""
    parameters = task_run.get("parameters") or {}
    provider_name = parameters.get("Provider") or ""
    request_json = parameters.get("Request") or ""
    task_id = task_run.get("taskId") or action_id

    for attempt in range(MAX_SUBMIT_ATTEMPTS):
        submission = context.step(
            submit_task(provider_name, request_json, task_id, attempt)
        )
        if "handle" in submission or not submission.get("retryable"):
            break
        context.wait(Duration.from_seconds(SUBMIT_RETRY_SECONDS))

    if "handle" not in submission:
        return _action_result(
            "FAILED",
            started_at,
            submission["submittedAt"],
            exit_code=1,
            message=submission.get("error", "The provider did not accept the request"),
        )

    last_observed_at = submission["submittedAt"]
    for _ in range(MAX_TASK_POLLS):
        context.wait(Duration.from_seconds(TASK_POLL_SECONDS))
        status = context.step(poll_task(provider_name, submission["handle"]))

        # Heartbeat on every poll, or the service marks the worker NOT_RESPONDING and
        # reassigns the task. The same response is where cancellation is observed.
        beat = context.step(
            heartbeat(
                worker_id,
                {
                    action_id: {
                        "startedAt": started_at,
                        "updatedAt": status["observedAt"],
                        "progressMessage": status.get("message", status["state"])[
                            :MAX_PROGRESS_MESSAGE
                        ],
                    }
                },
            )
        )
        if beat["workerDeleted"]:
            return _action_result(
                "INTERRUPTED",
                started_at,
                status["observedAt"],
                message="Worker was deleted while the request was running",
            )
        if action_id in beat.get("cancelSessionActions", {}):
            return _action_result(
                "CANCELED",
                started_at,
                status["observedAt"],
                message="Canceled by the service while the request was running",
            )

        if status["state"] == "SUCCEEDED":
            return _action_result(
                "SUCCEEDED",
                started_at,
                status["observedAt"],
                exit_code=0,
                message=status.get("message", "Request finished"),
                progressPercent=100.0,
            )
        if status["state"] == "FAILED":
            return _action_result(
                "FAILED",
                started_at,
                status["observedAt"],
                exit_code=1,
                message=status.get("message", "Request failed"),
            )
        last_observed_at = status["observedAt"]

    return _action_result(
        "FAILED",
        started_at,
        last_observed_at,
        exit_code=1,
        message=f"Timed out after {MAX_TASK_POLLS} polls waiting for the request",
    )


def _action_result(
    completed_status: str,
    started_at: str,
    ended_at: str,
    *,
    exit_code: Optional[int] = None,
    message: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Build the session action result that UpdateWorkerSchedule expects."""
    result: dict[str, Any] = {
        "completedStatus": completed_status,
        "startedAt": started_at,
        "endedAt": ended_at,
        **extra,
    }
    if exit_code is not None:
        result["processExitCode"] = exit_code
    if message:
        result["progressMessage"] = message[:MAX_PROGRESS_MESSAGE]
    return result


def _summarize_sessions(assigned_sessions: dict[str, Any]) -> dict[str, Any]:
    """Keep only the session fields this worker acts on, for a smaller checkpoint."""
    summary: dict[str, Any] = {}
    for session_id, session in assigned_sessions.items():
        summary[session_id] = {
            "queueId": session["queueId"],
            "jobId": session["jobId"],
            "sessionActions": [
                {
                    "sessionActionId": action["sessionActionId"],
                    "definition": _summarize_definition(action["definition"]),
                }
                for action in session.get("sessionActions", [])
            ],
        }
    return summary


def _summarize_definition(definition: dict[str, Any]) -> dict[str, Any]:
    """Flatten a session action definition to its plain-value essentials."""
    if "taskRun" in definition:
        raw_parameters = definition["taskRun"].get("parameters", {})
        return {
            "taskRun": {
                "taskId": definition["taskRun"].get("taskId"),
                "stepId": definition["taskRun"].get("stepId"),
                "parameters": {
                    name: _unwrap_parameter(value) for name, value in raw_parameters.items()
                },
            }
        }
    for key in ("envEnter", "envExit", "syncInputJobAttachments"):
        if key in definition:
            return {key: definition[key]}
    return {}


def _unwrap_parameter(tagged_value: dict[str, Any]) -> Optional[str]:
    """Return the value from a `{type: value}` task parameter."""
    for key in ("string", "path", "int", "float", "chunkInt"):
        if key in tagged_value:
            return tagged_value[key]
    return None
