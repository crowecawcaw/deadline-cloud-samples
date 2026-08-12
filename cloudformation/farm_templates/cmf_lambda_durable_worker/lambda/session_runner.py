# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Runs one Open Job Description session action to completion, using `openjd-sessions`.

This is the same library `deadline-cloud-worker-agent` runs sessions with. Passing
`user=None` short-circuits every privileged path in it, so it needs no sudo, no root, no
group setup, and no user of its own, which is what makes it usable inside Lambda.

Nothing here may span a `context.wait()`. The wait ends the invocation, which kills the
subprocess and discards the working directory, so the caller runs exactly one action per
invocation and this module blocks until that action has finished and been reaped.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

from openjd.model import (  # type: ignore[import-not-found]
    DecodeValidationError,
    ParameterValue,
    ParameterValueType,
    SymbolTable,
    parse_model,
)
from openjd.model.v2023_09 import (  # type: ignore[import-not-found]
    Environment as Environment_2023_09,
)
from openjd.model.v2023_09 import (  # type: ignore[import-not-found]
    StepTemplate as StepTemplate_2023_09,
)
from openjd.sessions import (  # type: ignore[import-not-found]
    LOG,
    ActionState,
    PathMappingRule,
    Session,
)

import action_output

logger = logging.getLogger(__name__)

# Passed explicitly so nothing depends on what tempfile.gettempdir() resolves to. Lambda
# gives /tmp its own size, configured separately from memory.
SESSION_ROOT = Path(os.environ.get("SESSION_ROOT", "/tmp/openjd"))

# Must stay comfortably below the function's own Timeout: the action has to finish, or be
# canceled and reaped, inside this one invocation.
ACTION_TIMEOUT_SECONDS = int(os.environ.get("ACTION_TIMEOUT_SECONDS", "240"))
CANCEL_GRACE_SECONDS = int(os.environ.get("CANCEL_GRACE_SECONDS", "20"))

# Stands in for an `onEnter` that already ran in an earlier invocation. /bin/sh rather than
# /bin/true because openjd-sessions already execs a #!/bin/sh wrapper, so this adds no new
# assumption about what the runtime image contains.
NOOP_ACTION = {"command": "/bin/sh", "args": ["-c", "exit 0"]}

ACTION_STATES = {
    ActionState.SUCCESS: "SUCCESS",
    ActionState.FAILED: "FAILED",
    ActionState.CANCELED: "CANCELED",
    ActionState.TIMEOUT: "TIMEOUT",
}


class SessionRunnerError(Exception):
    """An action could not be run at all, as opposed to running and failing."""


def run_action(
    *,
    kind: str,
    session_id: str,
    template: dict[str, Any],
    job_parameters: dict[str, dict[str, str]],
    os_env_vars: dict[str, Optional[str]],
    task_parameters: Optional[dict[str, dict[str, str]]] = None,
    path_mapping_rules: Optional[list[dict[str, str]]] = None,
    environment_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run one `taskRun`, `envEnter`, or `envExit` action and report what it did."""
    _sweep_stale_sessions()
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)

    waiter = _ActionWaiter()
    capture = _ProtocolCapture()
    LOG.setLevel(logging.INFO)
    LOG.addHandler(capture)
    try:
        session = Session(
            session_id=session_id,
            job_parameter_values=_parameter_values(job_parameters, task=False),
            path_mapping_rules=_path_mapping_rules(path_mapping_rules),
            user=None,
            callback=waiter.on_status,
            os_env_vars=os_env_vars,  # type: ignore[arg-type]  # a None value means "remove"
            session_root_directory=SESSION_ROOT,
        )
        try:
            if kind == "taskRun":
                static_variables: dict[str, str] = {}
                step = _parse(StepTemplate_2023_09, template, "step")
                waiter.arm()
                session.run_task(
                    step_script=step.script,
                    task_parameter_values=_parameter_values(task_parameters or {}, task=True),
                )
            else:
                environment, static_variables = _prepare_environment(
                    kind=kind, template=template, job_parameters=job_parameters
                )
                waiter.arm()
                session.enter_environment(
                    environment=environment, identifier=environment_id or session_id
                )
            status = waiter.settle(session)

            if kind == "envExit" and status.state is ActionState.SUCCESS:
                # The entry above did nothing: its onEnter was replaced so that only the
                # real onExit runs in this session.
                waiter.arm()
                session.exit_environment(identifier=environment_id or session_id)
                status = waiter.settle(session)
        finally:
            session.cleanup()
    finally:
        LOG.removeHandler(capture)

    result: dict[str, Any] = {
        "state": ACTION_STATES.get(status.state, "FAILED"),
        "exitCode": status.exit_code,
        "message": status.fail_message or status.status_message or "",
        "progress": status.progress,
        "awaitTokens": [],
        "envDelta": {"set": dict(static_variables), "unset": []},
    }
    if result["state"] != "SUCCESS":
        return result

    # Only read on success: a failed action's half-written output is not a promise.
    result["awaitTokens"] = action_output.await_tokens(capture.lines)
    delta = action_output.env_delta(capture.lines)
    result["envDelta"]["set"].update(delta["set"])
    result["envDelta"]["unset"] = delta["unset"]
    return result


class _ActionWaiter:
    """Turns the library's non-blocking action API into a blocking one.

    `run_task` and the environment calls return once the subprocess has started; the terminal
    state arrives later, on a runner-pool thread.
    """

    def __init__(self) -> None:
        self._finished = threading.Event()
        self._status: Any = None

    def on_status(self, _session_id: str, status: Any) -> None:
        if status.state is not ActionState.RUNNING:
            self._status = status
            self._finished.set()

    def arm(self) -> None:
        """Call immediately before starting an action, so a previous one cannot satisfy it."""
        self._status = None
        self._finished.clear()

    def settle(self, session: Session) -> Any:
        """Block until the action ends, cancelling it rather than orphaning it.

        A subprocess that outlives the invocation is frozen with it and thaws inside a later
        one, with a working directory that no longer exists.
        """
        if not self._finished.wait(ACTION_TIMEOUT_SECONDS):
            logger.error("Action did not finish within %ss; cancelling", ACTION_TIMEOUT_SECONDS)
            session.cancel_action(mark_action_failed=True)
            self._finished.wait(CANCEL_GRACE_SECONDS)
        status = self._status or session.action_status
        if status is None:
            raise SessionRunnerError("The session reported no status for the action")
        return status


class _ProtocolCapture(logging.Handler):
    """Collects only the protocol lines from the session log.

    The library hands `openjd_env` changes to its own logging filter and returns before the
    application callback runs, so parsing the specified stdout protocol here is the only way
    to see them without reaching into library internals. Everything else the session logs
    reaches CloudWatch by itself, because `openjd.sessions` propagates to the root logger.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.strip().startswith(action_output.CAPTURED_PREFIXES):
            self.lines.append(message)


def _prepare_environment(
    *, kind: str, template: dict[str, Any], job_parameters: dict[str, dict[str, str]]
) -> tuple[Any, dict[str, str]]:
    """Parse an environment template and resolve the variables it defines statically."""
    # BatchGetJobEntity returns the definition unwrapped from the `environment` key an
    # authored template nests it under. Both shapes are accepted because the nested form is
    # what a reader sees in `queue_environments/`.
    definition = template.get("environment") or template
    if kind == "envExit":
        definition = _with_noop_on_enter(definition)
    environment = _parse(Environment_2023_09, definition, "environment")
    if kind == "envExit":
        # The layer is being un-applied, so nothing this exit prints is worth keeping.
        return environment, {}
    return environment, _resolve_variables(environment, job_parameters)


def _parse(model: Any, obj: dict[str, Any], label: str) -> Any:
    try:
        return parse_model(model=model, obj=obj)
    except DecodeValidationError as exc:
        raise SessionRunnerError(f"This job's {label} template is not valid: {exc}") from exc


def _parameter_values(
    parameters: dict[str, dict[str, str]], *, task: bool
) -> dict[str, ParameterValue]:
    """Convert already-unwrapped parameters into what a session expects."""
    values = {}
    for name, parameter in parameters.items():
        try:
            value_type = ParameterValueType(parameter["type"])
        except ValueError as exc:
            raise SessionRunnerError(
                f"Parameter {name} has an unusable type {parameter['type']!r}"
            ) from exc
        if value_type is ParameterValueType.CHUNK_INT and not task:
            raise SessionRunnerError(f"Job parameter {name} cannot be a chunked integer")
        values[name] = ParameterValue(type=value_type, value=parameter["value"])
    return values


def _path_mapping_rules(
    rules: Optional[list[dict[str, str]]],
) -> Optional[list[PathMappingRule]]:
    """Translate the API's rules into the library's, which spells its fields differently."""
    if not rules:
        return None
    return [
        PathMappingRule.from_dict(
            {
                "source_path_format": rule["sourcePathFormat"],
                "source_path": rule["sourcePath"],
                "destination_path": rule["destinationPath"],
            }
        )
        for rule in rules
    ]


def _resolve_variables(
    environment: Any, job_parameters: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Resolve an environment's static `variables`, which the library does not report.

    Only job parameters are in scope. A variable built from the session working directory
    cannot survive the wait between actions, so failing to resolve one is the right answer.
    """
    if not environment.variables:
        return {}
    symbols: dict[str, str] = {}
    for name, parameter in job_parameters.items():
        symbols[f"Param.{name}"] = parameter["value"]
        symbols[f"RawParam.{name}"] = parameter["value"]
    symbol_table = SymbolTable(source=symbols)
    try:
        return {
            name: value.resolve(symtab=symbol_table)
            for name, value in environment.variables.items()
        }
    except Exception as exc:
        raise SessionRunnerError(
            f"A variable of environment '{environment.name}' could not be resolved from job "
            f"parameters alone: {exc}"
        ) from exc


def _with_noop_on_enter(definition: dict[str, Any]) -> dict[str, Any]:
    """Return the environment with its `onEnter` replaced by a command that does nothing.

    Replaced rather than removed because `onEnter` is required in newer revisions of the
    model. The real `onEnter` already ran in an earlier invocation, and its variables come
    back through `os_env_vars`, so running it again would only repeat its side effects.
    """
    script = definition.get("script")
    if not script or not script.get("actions"):
        return definition
    actions = {**script["actions"], "onEnter": dict(NOOP_ACTION)}
    return {**definition, "script": {**script, "actions": actions}}


def _sweep_stale_sessions() -> None:
    """Delete session directories left by an invocation that was frozen mid-action.

    /tmp may or may not survive to the next invocation, and one that does survive belongs to
    a session that will not.
    """
    if not SESSION_ROOT.is_dir():
        return
    for stale in SESSION_ROOT.iterdir():
        shutil.rmtree(stale, ignore_errors=True)
