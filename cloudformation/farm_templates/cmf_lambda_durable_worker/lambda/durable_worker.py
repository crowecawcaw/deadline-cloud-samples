# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A Deadline Cloud customer-managed fleet worker that runs as a Lambda durable function.

Why a durable function
----------------------
A conventional worker holds a host for its whole lifetime, even though a worker that
dispatches API calls to another service spends nearly all of its time waiting: idle
between schedule polls, then blocked on a request that takes minutes to finish. A
durable function turns that waiting into `context.wait()`, which suspends the
execution and stops compute charges. The worker stays registered and keeps
heartbeating, but consumes no compute while asleep.

The loop below therefore has two kinds of sleep, and neither one burns compute:

  * idle polling  - sleep for the `updateIntervalSeconds` the service asks for
  * work in flight - submit to Bedrock, then sleep between `GetAsyncInvoke` polls

Writing for replay
------------------
Lambda resumes a suspended execution by re-running the handler from the top and
substituting stored results for completed steps. That makes determinism a
correctness requirement, so this module follows three rules:

1. Anything non-deterministic or side-effecting happens inside `context.step()`.
   Each step is checkpointed once and replayed from its stored result, so a
   `CreateWorker` call cannot register a second worker on replay.
2. Control flow depends only on step results, never on ambient state such as a
   clock read or a random value at the top level of the handler.
3. Credentials are refreshed in their own step after every wait rather than being
   carried across one. A replayed credential blob would likely be expired, and
   refreshing is cheap compared to the request it protects.
"""

from __future__ import annotations

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

import bedrock_task
import worker_registry
from worker_protocol import (
    DeadlineWorker,
    WorkerDeletedError,
    default_capabilities,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FARM_ID = os.environ["FARM_ID"]
FLEET_ID = os.environ["FLEET_ID"]
REGION = os.environ.get("AWS_REGION", "us-west-2")

# A worker that finds no work for this many consecutive polls deletes itself. Scale-in
# is normally driven by the service setting `desiredWorkerStatus` to STOPPED; this is a
# backstop so an orphaned execution cannot idle for the full one-year execution
# timeout.
MAX_IDLE_POLLS = int(os.environ.get("MAX_IDLE_POLLS", "20"))

# Bound the loop so a wedged execution cannot run forever. Each iteration is one
# schedule poll, so this is a count of polls rather than a wall-clock limit.
MAX_LOOP_ITERATIONS = int(os.environ.get("MAX_LOOP_ITERATIONS", "2000"))


# -- durable steps ---------------------------------------------------------------
# Each step is checkpointed. On replay the stored result is returned and the body
# does not run again.


@durable_step
def register_worker(step_context, host_name: str) -> dict[str, Any]:
    """Create the worker and move it to STARTED.

    Registration and the STARTED transition share one step so that a replay cannot
    leave a worker created but never started.
    """
    worker = DeadlineWorker(farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION)
    worker_id = worker.create_worker(host_name=host_name)
    worker.assume_fleet_role()
    worker.update_worker_status(status="STARTED", capabilities=default_capabilities())
    worker_registry.register(
        fleet_id=FLEET_ID, worker_id=worker_id, started_at=worker_registry.utc_now_iso()
    )
    step_context.logger.info(f"Worker {worker_id} registered and STARTED")
    return {"workerId": worker_id}


@durable_step
def mark_action_started(step_context, session_action_id: str) -> str:
    """Record when work on a session action began.

    UpdateWorkerSchedule rejects a completed action that has no `startedAt`, so the
    start time has to be captured before the work runs, not invented afterwards. It
    is taken inside a step because a clock read is non-deterministic: on replay the
    checkpointed value is returned instead of the current time, which keeps the
    reported timestamps stable across suspensions.
    """
    started_at = worker_registry.utc_now_iso()
    step_context.logger.info(f"Starting {session_action_id} at {started_at}")
    return started_at


@durable_step
def poll_schedule(
    step_context, worker_id: str, updated_session_actions: dict[str, Any]
) -> dict[str, Any]:
    """Send a heartbeat with any progress, and collect newly assigned work.

    Credentials are acquired fresh here because this step runs after a wait, when a
    replayed credential blob would likely have expired.
    """
    worker = DeadlineWorker(
        farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION, worker_id=worker_id
    )
    worker.assume_fleet_role()
    try:
        response = worker.update_worker_schedule(
            updated_session_actions=updated_session_actions
        )
    except WorkerDeletedError:
        step_context.logger.warning(f"Worker {worker_id} was deleted by the service")
        return {"workerDeleted": True, "updateIntervalSeconds": 0, "assignedSessions": {}}

    # Check the drain flag on the same beat as the heartbeat. Scale-in for a
    # customer-managed fleet is the fleet owner's responsibility, so this flag, not
    # `desiredWorkerStatus`, is what normally ends a worker's life here.
    drain_requested = worker_registry.should_drain(fleet_id=FLEET_ID, worker_id=worker_id)

    # Reduce the response to the plain, JSON-serializable fields the loop needs.
    # Checkpoint payloads are size-limited, and the raw response contains timestamps
    # and log configuration this worker does not use.
    return {
        "workerDeleted": False,
        "drainRequested": drain_requested,
        "updateIntervalSeconds": response.get("updateIntervalSeconds", 15),
        "desiredWorkerStatus": response.get("desiredWorkerStatus"),
        "assignedSessions": _summarize_sessions(response.get("assignedSessions", {})),
        "cancelSessionActions": response.get("cancelSessionActions", {}),
    }


@durable_step
def submit_bedrock_job(
    step_context, task_parameters: dict[str, Any], attempt: int = 0
) -> dict[str, Any]:
    """Start the asynchronous Bedrock request described by the task's parameters.

    `attempt` is part of the step's arguments so each retry is a distinct step with its
    own checkpoint. Without it a replay would return the first attempt's throttled
    result forever instead of re-submitting.
    """
    result = bedrock_task.start_generation(
        task_parameters=task_parameters, logger=step_context.logger
    )
    # Checkpointed so a failure path can report a stable `endedAt` on replay.
    result["submittedAt"] = worker_registry.utc_now_iso()
    return result


@durable_step
def check_bedrock_job(step_context, invocation_arn: str) -> dict[str, Any]:
    """Check whether the Bedrock request has finished.

    The completion time is captured here, in the same step that observes completion,
    so the reported `endedAt` is a checkpointed value rather than a fresh clock read
    on a later replay.
    """
    result = bedrock_task.check_generation(
        invocation_arn=invocation_arn, logger=step_context.logger
    )
    result["observedAt"] = worker_registry.utc_now_iso()
    return result


@durable_step
def deregister_worker(step_context, worker_id: str) -> dict[str, Any]:
    """Drain and deregister: STOPPING, then STOPPED, then DeleteWorker.

    Going through STOPPING tells the service to stop assigning work before the
    worker disappears, so an in-flight assignment is not lost to a hard delete.
    """
    worker = DeadlineWorker(
        farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION, worker_id=worker_id
    )
    try:
        worker.assume_fleet_role()
        worker.update_worker_status(status="STOPPING")
        worker.update_worker_status(status="STOPPED")
        worker.delete_worker()
        step_context.logger.info(f"Worker {worker_id} deregistered")
    except WorkerDeletedError:
        # Already gone. Deregistration is idempotent by intent, so this is success.
        pass
    # Drop the registry row last, so this worker keeps counting toward fleet capacity
    # until it has actually stopped.
    worker_registry.deregister(fleet_id=FLEET_ID, worker_id=worker_id)
    return {"deregistered": True}


# -- handler ---------------------------------------------------------------------


@durable_execution
def lambda_handler(event: dict[str, Any], context: DurableContext) -> dict[str, Any]:
    """Run one worker for its whole lifetime as a single durable execution.

    The execution is started by the scaling handler on scale-out and ends when the
    service asks the worker to stop, when the worker has been idle too long, or when
    the service has deleted the worker.
    """
    host_name = event.get("hostName", "durable-lambda-worker")

    registration = context.step(register_worker(host_name))
    worker_id = registration["workerId"]

    # Progress to report on the next heartbeat. UpdateWorkerSchedule is both the
    # heartbeat and the progress-reporting call, so completed work rides along with
    # the next poll instead of needing a separate request.
    pending_updates: dict[str, Any] = {}
    idle_polls = 0
    tasks_completed = 0
    stop_reason = "loop-limit-reached"

    for _ in range(MAX_LOOP_ITERATIONS):
        poll = context.step(poll_schedule(worker_id, pending_updates))
        pending_updates = {}

        if poll["workerDeleted"]:
            # Nothing left to drain, and no valid worker ID to drain it with.
            return {
                "workerId": worker_id,
                "stopReason": "worker-deleted-by-service",
                "tasksCompleted": tasks_completed,
            }

        if poll.get("desiredWorkerStatus") == "STOPPED":
            stop_reason = "service-requested-stop"
            break

        if poll.get("drainRequested"):
            # Scale-in. Any work already assigned in this poll is finished first, so
            # draining never abandons a request in flight. The results land in
            # pending_updates and are reported by the final heartbeat below.
            stop_reason = "scale-in-drain"
            tasks_completed += _finish_assigned_work(
                context=context,
                poll=poll,
                pending_updates=pending_updates,
            )
            break

        assigned = poll.get("assignedSessions") or {}
        if not assigned:
            idle_polls += 1
            if idle_polls >= MAX_IDLE_POLLS:
                stop_reason = "idle-timeout"
                break
            # Sleep exactly as long as the service asked. This is the idle case, and
            # the execution is suspended for the whole interval at no compute cost.
            context.wait(Duration.from_seconds(poll["updateIntervalSeconds"]))
            continue

        idle_polls = 0
        tasks_completed += _finish_assigned_work(
            context=context,
            poll=poll,
            pending_updates=pending_updates,
        )

    # Report the final batch of results before draining, otherwise the last task's
    # outcome is never seen by the service.
    if pending_updates:
        context.step(poll_schedule(worker_id, pending_updates))

    context.step(deregister_worker(worker_id))
    return {
        "workerId": worker_id,
        "stopReason": stop_reason,
        "tasksCompleted": tasks_completed,
    }


def _finish_assigned_work(
    *,
    context: DurableContext,
    poll: dict[str, Any],
    pending_updates: dict[str, Any],
) -> int:
    """Run every action assigned by this poll, recording results for the next beat.

    Results accumulate in `pending_updates` rather than being reported immediately,
    because `UpdateWorkerSchedule` carries both the heartbeat and the progress report.
    Returns the number of actions that succeeded.
    """
    succeeded = 0
    for session in (poll.get("assignedSessions") or {}).values():
        for action in session["sessionActions"]:
            result = _run_session_action(context=context, action=action)
            pending_updates[action["sessionActionId"]] = result
            if result.get("completedStatus") == "SUCCEEDED":
                succeeded += 1
    return succeeded


def _run_session_action(
    *,
    context: DurableContext,
    action: dict[str, Any],
) -> dict[str, Any]:
    """Run one assigned session action and return its result for the next heartbeat.

    Deadline Cloud assigns environment enter/exit actions around task runs. This
    worker has no local session to prepare, so those are acknowledged as succeeded
    and only `taskRun` actions do real work.
    """
    definition = action["definition"]
    session_action_id = action["sessionActionId"]

    # UpdateWorkerSchedule requires startedAt on every completed action, so the start
    # time is recorded before any work begins.
    started_at = context.step(mark_action_started(session_action_id))

    if "taskRun" not in definition:
        # envEnter, envExit, or syncInputJobAttachments: nothing to do locally.
        return {
            "completedStatus": "SUCCEEDED",
            "processExitCode": 0,
            "startedAt": started_at,
            "endedAt": started_at,
        }

    task_parameters = definition["taskRun"].get("parameters", {})
    # Give Bedrock output a stable, unique prefix. The task ID is used rather than a
    # generated value so the S3 location is the same across replays.
    task_parameters = dict(task_parameters)
    task_parameters.setdefault("TaskId", definition["taskRun"].get("taskId") or session_action_id)
    # Submit, retrying a throttle behind a durable wait. Bedrock's per-account
    # concurrency limits for generation models are low, so several workers starting at
    # once will collide; sleeping here is unbilled, which makes waiting out the limit
    # far cheaper than failing the task.
    for attempt in range(bedrock_task.MAX_SUBMIT_ATTEMPTS):
        submission = context.step(submit_bedrock_job(task_parameters, attempt))
        if submission.get("invocationArn") or not submission.get("throttled"):
            break
        context.wait(Duration.from_seconds(bedrock_task.SUBMIT_RETRY_SECONDS))

    if not submission.get("invocationArn"):
        return {
            "completedStatus": "FAILED",
            "processExitCode": 1,
            "startedAt": started_at,
            # `submittedAt` is checkpointed inside the submit step, so it replays
            # identically. Reading the clock here instead would drift on replay.
            "endedAt": submission.get("submittedAt", started_at),
            "progressMessage": submission.get("error", "Failed to start Bedrock request")[
                :4096
            ],
        }

    # Wait out the generation. Each iteration suspends the execution, so a request
    # that takes ten minutes costs compute only for the brief polls, not the wait.
    last_observed_at = submission.get("submittedAt", started_at)
    for _ in range(bedrock_task.MAX_GENERATION_POLLS):
        context.wait(Duration.from_seconds(bedrock_task.GENERATION_POLL_SECONDS))
        status = context.step(check_bedrock_job(submission["invocationArn"]))

        if status["status"] == "Completed":
            return {
                "completedStatus": "SUCCEEDED",
                "processExitCode": 0,
                "progressPercent": 100.0,
                "startedAt": started_at,
                "endedAt": status["observedAt"],
                "progressMessage": f"Output written to {status.get('outputUri', 'S3')}"[
                    :4096
                ],
            }
        if status["status"] == "Failed":
            return {
                "completedStatus": "FAILED",
                "processExitCode": 1,
                "startedAt": started_at,
                "endedAt": status["observedAt"],
                "progressMessage": status.get("failureMessage", "Generation failed")[:4096],
            }
        last_observed_at = status["observedAt"]

    return {
        "completedStatus": "FAILED",
        "processExitCode": 1,
        "startedAt": started_at,
        "endedAt": last_observed_at,
        "progressMessage": "Timed out waiting for the Bedrock request to finish",
    }


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
                # Task parameters arrive tagged by type, as {"string": "..."} or
                # {"int": "3"}. Unwrap to the single contained value.
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
