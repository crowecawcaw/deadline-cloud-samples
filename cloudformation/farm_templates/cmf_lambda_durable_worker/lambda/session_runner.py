# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Runs one Open Job Description session action to completion, using `openjd-sessions`.

This is the same library `deadline-cloud-worker-agent` runs sessions with. Passing
`user=None` short-circuits every privileged path in it, so it needs no sudo, no root, no
group setup, and no user of its own, which is what makes it usable inside Lambda.

Templates and parameters arrive already parsed and validated, by the same agent code that
parses them for a real worker. This module only runs them.

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
    ParameterValue,
    ParameterValueType,
    SymbolTable,
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
    template: Any,
    job_parameters: dict[str, ParameterValue],
    os_env_vars: dict[str, Optional[str]],
    task_parameters: Optional[dict[str, ParameterValue]] = None,
    path_mapping_rules: Optional[list[PathMappingRule]] = None,
    environment_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run one `taskRun`, `envEnter`, or `envExit` action and report what it did.

    `template` is an already-parsed step template or environment.
    """
    _reject_chunked_job_parameters(job_parameters)
    _sweep_stale_sessions()
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)

    waiter = _ActionWaiter()
    capture = _ProtocolCapture()
    LOG.setLevel(logging.INFO)
    LOG.addHandler(capture)
    try:
        session = Session(
            session_id=session_id,
            job_parameter_values=job_parameters,
            path_mapping_rules=path_mapping_rules or None,
            user=None,
            callback=waiter.on_status,
            os_env_vars=os_env_vars,  # type: ignore[arg-type]  # a None value means "remove"
            session_root_directory=SESSION_ROOT,
        )
        try:
            if kind == "taskRun":
                static_variables: dict[str, str] = {}
                waiter.arm()
                session.run_task(
                    step_script=template.script,
                    task_parameter_values=task_parameters or {},
                )
            else:
                # An exit un-applies the layer, so nothing that exit resolves is worth keeping.
                static_variables = (
                    {} if kind == "envExit" else _resolve_variables(template, job_parameters)
                )
                waiter.arm()
                session.enter_environment(
                    environment=template, identifier=environment_id or session_id
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


def _reject_chunked_job_parameters(job_parameters: dict[str, ParameterValue]) -> None:
    """A chunked integer is a task parameter only, so one here is a wire-format error."""
    for name, value in job_parameters.items():
        if value.type is ParameterValueType.CHUNK_INT:
            raise SessionRunnerError(f"Job parameter {name} cannot be a chunked integer")


def _resolve_variables(
    environment: Any, job_parameters: dict[str, ParameterValue]
) -> dict[str, str]:
    """Resolve an environment's static `variables`, which the library does not report.

    Only job parameters are in scope. A variable built from the session working directory
    cannot survive the wait between actions, so failing to resolve one is the right answer.
    """
    if not environment.variables:
        return {}
    symbols: dict[str, str] = {}
    for name, parameter in job_parameters.items():
        symbols[f"Param.{name}"] = parameter.value
        symbols[f"RawParam.{name}"] = parameter.value
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


def _sweep_stale_sessions() -> None:
    """Delete session directories left by an invocation that was frozen mid-action.

    /tmp may or may not survive to the next invocation, and one that does survive belongs to
    a session that will not.
    """
    if not SESSION_ROOT.is_dir():
        return
    for stale in SESSION_ROOT.iterdir():
        shutil.rmtree(stale, ignore_errors=True)
