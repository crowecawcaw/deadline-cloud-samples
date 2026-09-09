# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A Deadline Cloud customer-managed fleet worker that runs as a Lambda durable function.

Replay contract: everything non-deterministic or side-effecting happens inside a
`context.step()`, control flow depends only on step results, and step identity is
positional, so a step that repeats must vary its arguments to earn its own checkpoint.

One invocation runs exactly one Open Job Description session action, start to finish. An
action can never span a `context.wait()`, so the unbilled waiting happens only between
actions, or while awaiting the long-running request an action handed over.
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

import action_output
import providers
import session_env
import worker_protocol
import worker_registry
from worker_protocol import (
    WORKER_UNUSABLE,
    DeadlineRequestError,
    DeadlineRequestInterrupted,
    DeadlineWorker,
    WorkerStatus,
    default_capabilities,
    unwrap_parameters,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FARM_ID = os.environ["FARM_ID"]
FLEET_ID = os.environ["FLEET_ID"]
REGION = os.environ.get("AWS_REGION", "us-west-2")

# Poll timing is the worker's policy, not a provider's: every wait below suspends the
# execution, so a longer interval costs nothing and only delays noticing.
TASK_POLL_SECONDS = int(os.environ.get("TASK_POLL_SECONDS", "30"))
MAX_TASK_POLLS = int(os.environ.get("MAX_TASK_POLLS", "120"))

# Backstop so an orphaned execution cannot idle for the stack's whole ExecutionTimeout.
# Raise it for bursty jobs; at the usual 15s interval 20 polls is about five minutes.
MAX_IDLE_POLLS = int(os.environ.get("MAX_IDLE_POLLS", "20"))
MAX_LOOP_ITERATIONS = int(os.environ.get("MAX_LOOP_ITERATIONS", "2000"))

# Used only when a poll never reached the service and so brought back no interval of its own.
DEFAULT_UPDATE_INTERVAL_SECONDS = 15

# The service's limit on a progressMessage; a longer one fails the whole request.
MAX_PROGRESS_MESSAGE = 4096

# The same mapping the Deadline Cloud worker agent uses. A timed-out action is a failure to
# the service, which has no separate status for it.
COMPLETED_STATUS = {
    "SUCCESS": "SUCCEEDED",
    "FAILED": "FAILED",
    "CANCELED": "CANCELED",
    "TIMEOUT": "FAILED",
}

ACTION_KINDS = ("taskRun", "envEnter", "envExit")

JOB_ATTACHMENTS_MESSAGE = (
    "This worker does not support job attachments: its session directory is discarded "
    "whenever the worker suspends, so staged input files would be gone before the task "
    "ran. Submit without job attachments, or use a fleet whose workers keep a filesystem."
)


# -- durable steps ---------------------------------------------------------------


@durable_step
def register_worker(step_context, host_name: str) -> dict[str, Any]:
    """Create the worker and move it to STARTED.

    One step, so a replay cannot leave a worker created but never started.
    """
    worker = DeadlineWorker(farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION)
    worker_id = worker.create_worker(host_name=host_name)
    worker.update_worker_status(
        status=WorkerStatus.STARTED, capabilities=default_capabilities()
    )
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
    except WORKER_UNUSABLE:
        step_context.logger.warning(f"Worker {worker_id} is no longer usable while working")
        return {"workerDeleted": True, "cancelSessionActions": {}}
    except DeadlineRequestInterrupted as exc:
        # Nothing was reported and nothing was learned. The next poll of the running request
        # heartbeats again, so one missed beat is not worth ending the action over.
        step_context.logger.warning(f"Heartbeat for {worker_id} gave up retrying: {exc}")
        return {"workerDeleted": False, "retryLater": True, "cancelSessionActions": {}}
    return {
        "workerDeleted": False,
        "desiredWorkerStatus": response.get("desiredWorkerStatus"),
        "cancelSessionActions": response.get("cancelSessionActions", {}),
    }


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
    except WORKER_UNUSABLE:
        step_context.logger.warning(f"Worker {worker_id} is no longer usable")
        return {"workerDeleted": True, "updateIntervalSeconds": 0, "assignedSessions": {}}
    except DeadlineRequestInterrupted as exc:
        # The request never landed, so any results it carried were not recorded and the
        # service will assign the same work again.
        step_context.logger.warning(f"Poll for {worker_id} gave up retrying: {exc}")
        return {
            "workerDeleted": False,
            "retryLater": True,
            "updateIntervalSeconds": DEFAULT_UPDATE_INTERVAL_SECONDS,
            "assignedSessions": {},
        }

    # Scale-in for a customer-managed fleet is the fleet owner's business, so this flag,
    # not `desiredWorkerStatus`, is what normally ends a worker's life here.
    drain_requested = worker_registry.should_drain(fleet_id=FLEET_ID, worker_id=worker_id)

    return {
        "workerDeleted": False,
        "drainRequested": drain_requested,
        "updateIntervalSeconds": response.get(
            "updateIntervalSeconds", DEFAULT_UPDATE_INTERVAL_SECONDS
        ),
        "desiredWorkerStatus": response.get("desiredWorkerStatus"),
        "assignedSessions": _summarize_sessions(response.get("assignedSessions", {})),
        "cancelSessionActions": response.get("cancelSessionActions", {}),
    }


@durable_step
def run_action(
    step_context,
    worker_id: str,
    session_id: str,
    queue_id: str,
    job_id: str,
    action: dict[str, Any],
    env_layers: list[list[Any]],
) -> dict[str, Any]:
    """Run one session action to completion and report what it did.

    Templates and job parameters are fetched here rather than checkpointed: a step template
    can be far larger than the checkpoint size limit. Errors are returned rather than raised
    so a bad job costs one action, not the worker.
    """
    # Imported here so the durable worker module stays importable without openjd-sessions,
    # which the tests rely on to stub this seam out.
    import session_runner

    definition = action["definition"]
    kind = next(name for name in ACTION_KINDS if name in definition)
    worker = DeadlineWorker(
        farm_id=FARM_ID, fleet_id=FLEET_ID, region=REGION, worker_id=worker_id
    )
    try:
        environment_id = (
            None if kind == "taskRun" else definition[kind].get("environmentId")
        )
        step_id = definition["taskRun"]["stepId"] if kind == "taskRun" else None
        entities = worker.job_entities(job_id=job_id)
        # Warmed in one call because each round trip happens inside a billed invocation. An
        # entity too large to share a response is left uncached and fetched alone below.
        entities.cache_entities(
            worker_protocol.action_identifiers(
                job_id=job_id, step_id=step_id, environment_id=environment_id
            )
        )
        job_details = entities.job_details()
        template = (
            entities.step_details(step_id=step_id).step_template
            if kind == "taskRun"
            else worker_protocol.environment_template(
                entities,
                job_id=job_id,
                environment_id=environment_id,
                exiting=kind == "envExit",
            )
        )
        # A queue with no role leaves the base environment without credentials, which is
        # what keeps a task's script off the worker's own identity.
        credentials = (
            worker.assume_queue_role(queue_id=queue_id) if job_details.queue_role_arn else None
        )
        result = session_runner.run_action(
            kind=kind,
            session_id=session_id,
            template=template,
            job_parameters=job_details.parameters,
            task_parameters=unwrap_parameters(definition.get("taskRun", {}).get("parameters")),
            path_mapping_rules=job_details.path_mapping_rules,
            os_env_vars=session_env.compose(
                session_env.base_env(region=REGION, credentials=credentials), env_layers
            ),
            environment_id=environment_id,
        )
    except DeadlineRequestInterrupted as exc:
        # Nothing ran, so this action is still the service's to assign. Reporting a failure
        # would spend the task's retry on a throttle.
        step_context.logger.warning(f"Action {action['sessionActionId']} not started: {exc}")
        return {"state": "RETRY_LATER", "message": str(exc)}
    except (
        DeadlineRequestError,
        # How the agent's entity layer reports a broken job: RuntimeError for an entity the
        # service refused, ValueError for one that failed its validation.
        RuntimeError,
        ValueError,
        session_runner.SessionRunnerError,
        action_output.MalformedOutputError,
    ) as exc:
        step_context.logger.error(f"Action {action['sessionActionId']} could not run: {exc}")
        result = _unrun(str(exc))
    except Exception as exc:  # A defect here must cost one action, not the worker.
        step_context.logger.exception(f"Action {action['sessionActionId']} raised")
        result = _unrun(f"{type(exc).__name__}: {exc}")
    result["endedAt"] = worker_registry.utc_now_iso()
    return result


@durable_step
def poll_await(step_context, provider_name: str, handle: Any) -> dict[str, Any]:
    """Ask a provider whether the request an action handed over has finished."""
    try:
        provider = providers.resolve(provider_name)
        result = provider.poll(handle)
    except providers.UnknownProviderError as exc:
        result = {"state": "FAILED", "message": str(exc)}
    except Exception as exc:  # A provider defect must cost one action, not the worker.
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
        worker.update_worker_status(status=WorkerStatus.STOPPING)
        worker.update_worker_status(status=WorkerStatus.STOPPED)
        worker.delete_worker()
        step_context.logger.info(f"Worker {worker_id} deregistered")
    except (*worker_protocol.WORKER_UNDRAINABLE, DeadlineRequestInterrupted):
        # A worker the service has already taken away needs no draining, and the registry row
        # is removed below either way.
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

    # Per session, in entry order. Rebuilt from step results on replay, and the only way an
    # environment's variables reach an action that runs in a later invocation.
    env_layers: dict[str, list[list[Any]]] = {}
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

        if poll.get("retryLater"):
            # The request budget ran out before the service answered. Counted as idle so a
            # worker that can never reach the service still gives up eventually.
            idle_polls += 1
            if idle_polls >= MAX_IDLE_POLLS:
                stop_reason = "idle-timeout"
                break
            context.wait(Duration.from_seconds(poll["updateIntervalSeconds"]))
            continue

        if poll.get("desiredWorkerStatus") == "STOPPED":
            stop_reason = "service-requested-stop"
            break

        if poll.get("drainRequested"):
            # Work assigned by this same poll is finished first, so draining never
            # abandons a request in flight.
            stop_reason = "scale-in-drain"
            tasks_completed += _finish_assigned_work(
                context=context, worker_id=worker_id, poll=poll, env_layers=env_layers
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
            context=context, worker_id=worker_id, poll=poll, env_layers=env_layers
        )

    context.step(deregister_worker(worker_id))
    return {
        "workerId": worker_id,
        "stopReason": stop_reason,
        "tasksCompleted": tasks_completed,
    }


def _finish_assigned_work(
    *,
    context: DurableContext,
    worker_id: str,
    poll: dict[str, Any],
    env_layers: dict[str, list[list[Any]]],
) -> int:
    """Run every action this poll assigned, reporting each result as it completes.

    Returns how many actions succeeded.
    """
    succeeded = 0
    for session_id, session in (poll.get("assignedSessions") or {}).items():
        layers = env_layers.setdefault(session_id, [])
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
                    session_id=session_id,
                    queue_id=session["queueId"],
                    job_id=session["jobId"],
                    action=action,
                    env_layers=layers,
                )
                if result is None:
                    # Never attempted and never reported, so the service will assign it
                    # again. Later actions have to wait: reporting one of those now would
                    # arrive out of order.
                    break
                if result.get("completedStatus") == "SUCCEEDED":
                    succeeded += 1
                else:
                    session_failed = True

            # One result per call, in assigned order. The service rejects an out-of-order
            # report with "comes in a wrong order" and drops every result in the request.
            if context.step(poll_schedule(worker_id, {action_id: result})).get("retryLater"):
                # This result never landed, so the next one would be out of order too.
                break
        if not layers:
            del env_layers[session_id]
    return succeeded


def _run_session_action(
    *,
    context: DurableContext,
    worker_id: str,
    session_id: str,
    queue_id: str,
    job_id: str,
    action: dict[str, Any],
    env_layers: list[list[Any]],
) -> Optional[dict[str, Any]]:
    """Run one assigned session action and return its result for the next heartbeat.

    Returns None when the action was never attempted and should be assigned again.
    """
    definition = action["definition"]
    action_id = action["sessionActionId"]
    started_at = context.step(mark_action_started(action_id))

    if "syncInputJobAttachments" in definition:
        return _action_result("FAILED", started_at, started_at, message=JOB_ATTACHMENTS_MESSAGE)

    kind = next((name for name in ACTION_KINDS if name in definition), None)
    if kind is None:
        return _action_result(
            "FAILED",
            started_at,
            started_at,
            message=f"Unsupported session action type: {sorted(definition)}",
        )

    outcome = context.step(
        run_action(worker_id, session_id, queue_id, job_id, action, list(env_layers))
    )

    if outcome["state"] == "RETRY_LATER":
        return None

    if kind == "envEnter" and outcome["state"] == "SUCCESS":
        env_layers.append([definition["envEnter"]["environmentId"], outcome["envDelta"]])
    elif kind == "envExit":
        # Un-layered however the exit went: the environment is being left either way.
        env_layers[:] = session_env.drop(env_layers, definition["envExit"]["environmentId"])

    if outcome["state"] != "SUCCESS":
        return _action_result(
            COMPLETED_STATUS[outcome["state"]],
            started_at,
            outcome["endedAt"],
            exit_code=outcome["exitCode"],
            message=outcome["message"] or f"The action ended {outcome['state']}",
        )

    tokens = outcome["awaitTokens"]
    if len(tokens) > 1:
        return _action_result(
            "FAILED",
            started_at,
            outcome["endedAt"],
            exit_code=outcome["exitCode"],
            message=(
                f"This action printed {len(tokens)} '{action_output.AWAIT_PREFIX}' lines. "
                f"An action can hand the worker at most one request to await."
            ),
        )
    if not tokens:
        # An ordinary Open Job Description action: it did its own work and is finished.
        return _action_result(
            "SUCCEEDED",
            started_at,
            outcome["endedAt"],
            exit_code=outcome["exitCode"],
            message=outcome["message"],
            progressPercent=100.0,
        )

    return _await_request(
        context=context,
        worker_id=worker_id,
        action_id=action_id,
        started_at=started_at,
        token=tokens[0],
    )


def _await_request(
    *,
    context: DurableContext,
    worker_id: str,
    action_id: str,
    started_at: str,
    token: dict[str, Any],
) -> dict[str, Any]:
    """Wait, unbilled, for the long-running request an action handed over."""
    provider_name = token["provider"]
    last_observed_at = started_at
    for _ in range(MAX_TASK_POLLS):
        context.wait(Duration.from_seconds(TASK_POLL_SECONDS))
        status = context.step(poll_await(provider_name, token["handle"]))

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
        if beat.get("workerDeleted"):
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


def _unrun(message: str) -> dict[str, Any]:
    """The outcome of an action the worker could not get as far as running."""
    return {
        "state": "FAILED",
        "exitCode": None,
        "message": message,
        "progress": None,
        "awaitTokens": [],
        "envDelta": {"set": {}, "unset": []},
    }


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
    """Keep the parts of a session action definition the worker needs.

    Task parameters stay in their tagged wire form: unwrapping them can fail, and it has to
    fail inside the step that runs the action rather than the one that collects the schedule.
    """
    if "taskRun" in definition:
        return {
            "taskRun": {
                "taskId": definition["taskRun"].get("taskId"),
                "stepId": definition["taskRun"].get("stepId"),
                "parameters": definition["taskRun"].get("parameters", {}),
            }
        }
    for key in ("envEnter", "envExit", "syncInputJobAttachments"):
        if key in definition:
            return {key: definition[key]}
    return {}
