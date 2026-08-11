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
3. Timestamps are captured inside steps. `UpdateWorkerSchedule` requires `startedAt`
   on any completed action, and a clock read outside a checkpoint would report a
   different time on every replay.
4. Step identity is positional. The SDK matches a checkpoint to a step by call order,
   not by function name, so inserting a step ahead of an existing one shifts every
   later step's identity. That is the usual way an adapted durable function breaks:
   in-flight executions resume against a checkpoint log that no longer lines up.
   Adding a step at the end is safe, and a retry loop that re-runs the same step must
   vary its arguments, as `submit_bedrock_job` does with its attempt number, so each
   attempt gets its own checkpoint instead of replaying the first result forever.

Credentials are deliberately not checkpointed. They are far shorter-lived than a
durable execution, so a replayed copy would usually be expired. `worker_protocol` wraps
them in botocore's `RefreshableCredentials` instead and lets botocore renew them.
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

# A worker that finds no work for this many consecutive polls deletes itself. Scale-in
# normally drives shutdown through the drain flag; this is a backstop so an orphaned
# execution cannot idle for the stack's whole ExecutionTimeout, which is 24 hours.
# At the usual ~15s interval this is roughly five minutes of idle before the worker
# gives up its capacity, which is the knob to raise for bursty jobs.
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
    # The fleet role is assumed on first use, so no explicit call is needed here.
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
def heartbeat(step_context, worker_id: str, progress: dict[str, Any]) -> dict[str, Any]:
    """Report progress on a running action without completing it.

    A worker must keep calling `UpdateWorkerSchedule` even while busy, or the service
    stops hearing from it and marks it NOT_RESPONDING. The real worker agent runs
    sessions on a thread pool so its main loop can keep heartbeating; a durable
    execution is single-threaded, so the wait loop has to heartbeat itself.

    Sends the action's progress with no `completedStatus`, which tells the service the
    action is still running. Any cancellation or shutdown request in the response is
    returned so the caller can react.
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
    """Fetch a queue environment's template and apply what this worker can.

    Returns the variables the environment defines, or an error explaining why the
    environment cannot be honored. The error is returned rather than raised so the
    caller fails one action instead of the whole execution.
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
    """Send a heartbeat with any progress, and collect newly assigned work.

    Credentials are obtained on first use and refreshed by botocore, so nothing about
    expiry needs handling here even though this step runs after a suspension.
    """
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
        worker.update_worker_status(status="STOPPING")
        worker.update_worker_status(status="STOPPED")
        worker.delete_worker()
        step_context.logger.info(f"Worker {worker_id} deregistered")
    except WorkerNotUsableError:
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

    idle_polls = 0
    tasks_completed = 0
    stop_reason = "loop-limit-reached"

    for _ in range(MAX_LOOP_ITERATIONS):
        # Results are reported as each action completes, so a poll only ever needs to
        # ask for work.
        poll = context.step(poll_schedule(worker_id, {}))

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
            # draining never abandons a request in flight.
            stop_reason = "scale-in-drain"
            tasks_completed += _finish_assigned_work(
                context=context,
                worker_id=worker_id,
                poll=poll,
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
            worker_id=worker_id,
            poll=poll,
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
) -> int:
    """Run every action assigned by this poll, reporting each result as it completes.

    Each result is reported in its own `UpdateWorkerSchedule` call, in the order the
    service assigned the actions. The service enforces that order and rejects the whole
    request otherwise: reporting action 1 before action 0 fails with "comes in a wrong
    order", which loses every result in the batch, not just the out-of-order one. So
    results cannot be accumulated and sent together.

    Returns the number of actions that succeeded.

    One action that does not succeed stops the rest of its session: the service will not
    run further `taskRun`, `envEnter`, or `syncInputJobAttachments` actions once any
    action in the session has failed, been canceled, or been interrupted. Continuing
    anyway would submit Bedrock requests whose results the service discards. `envExit`
    actions still run, because they are the session's cleanup and have to happen even on
    the failure path.
    """
    succeeded = 0
    for session in (poll.get("assignedSessions") or {}).values():
        session_failed = False
        for action in session["sessionActions"]:
            action_id = action["sessionActionId"]
            is_env_exit = "envExit" in action.get("definition", {})

            if session_failed and not is_env_exit:
                result = {
                    # No timestamps: the action was never started, and the service
                    # rejects a NEVER_ATTEMPTED report that claims to have run.
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

            # Report this result on its own, before moving to the next action.
            context.step(poll_schedule(worker_id, {action_id: result}))
    return succeeded


def _run_session_action(
    *,
    context: DurableContext,
    worker_id: str,
    job_id: str,
    action: dict[str, Any],
) -> dict[str, Any]:
    """Run one assigned session action and return its result for the next heartbeat.

    Deadline Cloud wraps a task in environment enter and exit actions, one pair per
    queue environment, and may also assign a job attachments sync. Each kind is handled
    on its own terms rather than acknowledged wholesale.
    """
    definition = action["definition"]
    session_action_id = action["sessionActionId"]

    # UpdateWorkerSchedule requires startedAt on every completed action, so the start
    # time is recorded before any work begins.
    started_at = context.step(mark_action_started(session_action_id))

    if "envEnter" in definition:
        entered = context.step(
            enter_queue_environment(
                worker_id, job_id, definition["envEnter"]["environmentId"]
            )
        )
        if "error" in entered:
            # Failing stops the session, which surfaces the misconfiguration. Reporting
            # success here would let the task run in an environment that was never
            # prepared, which is the silent failure this replaced.
            return {
                "completedStatus": "FAILED",
                "startedAt": started_at,
                "endedAt": started_at,
                "progressMessage": entered["error"][:4096],
            }
        return {
            "completedStatus": "SUCCEEDED",
            "processExitCode": 0,
            "startedAt": started_at,
            "endedAt": started_at,
        }

    if "envExit" in definition:
        # Nothing to tear down: entering an environment only collected variables, and a
        # Lambda sandbox is discarded after the execution regardless.
        return {
            "completedStatus": "SUCCEEDED",
            "processExitCode": 0,
            "startedAt": started_at,
            "endedAt": started_at,
        }

    if "syncInputJobAttachments" in definition:
        # Job attachments stage files into a session directory, which this worker does
        # not have. Succeeding would leave the task expecting inputs that never arrived.
        return {
            "completedStatus": "FAILED",
            "startedAt": started_at,
            "endedAt": started_at,
            "progressMessage": (
                "This worker does not support job attachments: it has no session "
                "directory to stage input files into. Submit without job attachments, "
                "or use a fleet whose workers have a filesystem."
            ),
        }

    if "taskRun" not in definition:
        # An action type this worker does not recognize. Reporting success for work that
        # was never done is the worst option, so fail and say what happened.
        return {
            "completedStatus": "FAILED",
            "startedAt": started_at,
            "endedAt": started_at,
            "progressMessage": f"Unsupported session action type: {sorted(definition)}",
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
    # Every iteration also heartbeats: without that the service stops hearing from the
    # worker for the whole generation and marks it NOT_RESPONDING.
    last_observed_at = submission.get("submittedAt", started_at)
    for _ in range(bedrock_task.MAX_GENERATION_POLLS):
        context.wait(Duration.from_seconds(bedrock_task.GENERATION_POLL_SECONDS))
        status = context.step(check_bedrock_job(submission["invocationArn"]))

        beat = context.step(
            heartbeat(
                worker_id,
                {
                    session_action_id: {
                        "startedAt": started_at,
                        "updatedAt": status["observedAt"],
                        "progressMessage": f"Generation {status['status']}"[:4096],
                    }
                },
            )
        )
        if beat["workerDeleted"]:
            # The worker no longer exists, so no result can be reported for this
            # action. Abandon it and let the loop shut the worker down.
            return {
                "completedStatus": "INTERRUPTED",
                "startedAt": started_at,
                "endedAt": status["observedAt"],
                "progressMessage": "Worker was deleted while the request was running",
            }
        if session_action_id in beat.get("cancelSessionActions", {}):
            # The service withdrew this action. Stop polling and report it as canceled
            # rather than finishing work nobody is waiting for.
            return {
                "completedStatus": "CANCELED",
                "startedAt": started_at,
                "endedAt": status["observedAt"],
                "progressMessage": "Canceled by the service while generating",
            }

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
