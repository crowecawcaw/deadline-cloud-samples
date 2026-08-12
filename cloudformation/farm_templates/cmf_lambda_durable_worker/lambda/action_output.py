# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The two stdout line protocols the worker reads out of a session action.

`openjd_env:` and `openjd_unset_env:` belong to the Open Job Description specification and
are how an environment hands variables to the actions that follow it.

`durable_lambda_await:` is this worker's own extension: Open Job Description has no notion
of an action that suspends, so a task that wants its long-running request awaited says so
here. The prefix deliberately avoids the `openjd_` namespace, which belongs to the
specification and could otherwise grow a token that collides with this one.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

AWAIT_PREFIX = "durable_lambda_await:"
ENV_PREFIX = "openjd_env:"
UNSET_ENV_PREFIX = "openjd_unset_env:"
REDACTED_ENV_PREFIX = "openjd_redacted_env:"

CAPTURED_PREFIXES = (AWAIT_PREFIX, ENV_PREFIX, UNSET_ENV_PREFIX, REDACTED_ENV_PREFIX)


class MalformedOutputError(Exception):
    """An action printed a protocol line the worker cannot act on."""


def await_tokens(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Every await token an action printed, in the order it printed them."""
    tokens = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(AWAIT_PREFIX):
            continue
        payload = stripped[len(AWAIT_PREFIX) :].strip()
        try:
            token = json.loads(payload)
        except ValueError as exc:
            raise MalformedOutputError(
                f"A {AWAIT_PREFIX} line must be followed by a JSON object, but "
                f"{payload!r} could not be parsed: {exc}"
            ) from exc
        if not isinstance(token, dict) or not token.get("provider") or "handle" not in token:
            raise MalformedOutputError(
                f"A {AWAIT_PREFIX} object needs a non-empty 'provider' and a 'handle', "
                f"but got {payload!r}."
            )
        tokens.append({"provider": token["provider"], "handle": token["handle"]})
    return tokens


def env_delta(lines: Iterable[str]) -> dict[str, Any]:
    """The variables an action set and unset, as one layer.

    `unset` is applied after `set`, which is what makes an unset win over a set of the same
    name in the same action.
    """
    variables: dict[str, str] = {}
    unset: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(REDACTED_ENV_PREFIX):
            # By the time this line reaches the worker the library has already replaced the
            # value with asterisks, so carrying it across a wait would set the wrong value.
            raise MalformedOutputError(
                f"{REDACTED_ENV_PREFIX} is not supported by this worker: its value is "
                f"redacted before the worker can record it, and every variable has to be "
                f"recorded to survive the unbilled wait between session actions. Use "
                f"{ENV_PREFIX} for values the log may show, or fetch the secret in the "
                f"task's own script."
            )
        if stripped.startswith(ENV_PREFIX):
            assignment = stripped[len(ENV_PREFIX) :].strip()
            name, separator, value = assignment.partition("=")
            if not name or not separator:
                raise MalformedOutputError(
                    f"A {ENV_PREFIX} line must read NAME=VALUE, but got {assignment!r}."
                )
            variables[name] = value
        elif stripped.startswith(UNSET_ENV_PREFIX):
            name = stripped[len(UNSET_ENV_PREFIX) :].strip()
            if not name:
                raise MalformedOutputError(f"A {UNSET_ENV_PREFIX} line must name a variable.")
            unset.append(name)
    return {"set": variables, "unset": unset}
