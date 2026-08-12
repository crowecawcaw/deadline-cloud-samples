# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The environment one session action runs in, composed fresh for every action.

An Open Job Description session cannot outlive one Lambda invocation, so each environment's
variable delta is recorded on its own and the layers are recomposed, in entry order, for
every later action. Merged layers could not be un-layered again on `envExit`.

A `None` value means "remove this variable": `openjd-sessions` starts the subprocess from a
copy of the worker's own environment, so a variable is only absent if it is explicitly
removed.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

# Removed rather than left alone, so a task's script cannot fall back on the worker's own
# execution-role credentials when the queue supplies none of its own.
CREDENTIAL_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_CREDENTIAL_EXPIRATION",
)


def base_env(
    *, region: str, credentials: Optional[dict[str, Any]] = None
) -> dict[str, Optional[str]]:
    """What every session action starts from, before any environment layer."""
    env: dict[str, Optional[str]] = {
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
        # The managed runtime's documented PATH does not include the interpreter's own
        # directory, and a template that runs `python3` needs it. The AWS CLI is not in the
        # Python runtime at all, so a script has to use python3 with boto3 instead.
        "PATH": os.pathsep.join(
            [os.path.dirname(sys.executable), os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")]
        ),
    }
    for name in CREDENTIAL_VARIABLES:
        env[name] = None
    if credentials:
        env["AWS_ACCESS_KEY_ID"] = credentials["accessKeyId"]
        env["AWS_SECRET_ACCESS_KEY"] = credentials["secretAccessKey"]
        env["AWS_SESSION_TOKEN"] = credentials["sessionToken"]
        # Without this the runtime's own expiry for the worker's credentials would still be
        # in the environment, and botocore would try to refresh these credentials with it.
        env["AWS_CREDENTIAL_EXPIRATION"] = _as_iso(credentials["expiration"])
    return env


def compose(
    base: dict[str, Optional[str]], layers: list[list[Any]]
) -> dict[str, Optional[str]]:
    """Apply every environment layer over `base`, in entry order."""
    env = dict(base)
    for _, delta in layers:
        env.update(delta.get("set") or {})
        for name in delta.get("unset") or ():
            env[name] = None
    return env


def drop(layers: list[list[Any]], identifier: str) -> list[list[Any]]:
    """Remove one environment's layer, keeping the rest in entry order."""
    return [layer for layer in layers if layer[0] != identifier]


def _as_iso(expiration: Any) -> str:
    """Accept an expiry as either a datetime or an ISO-8601 string."""
    return expiration if isinstance(expiration, str) else expiration.isoformat()
