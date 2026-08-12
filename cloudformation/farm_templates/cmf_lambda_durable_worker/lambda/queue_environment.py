# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Queue environment support for a worker that cannot run scripts.

An environment that only defines `variables` is applied. One with a `script` is refused,
because a Lambda worker has no session directory, no writable filesystem outside /tmp,
and none of the tools such scripts drive. Refusing fails the `envEnter` action, which
surfaces the misconfiguration instead of running the task in an unprepared environment.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class UnsupportedEnvironmentError(Exception):
    """A queue environment needs something this worker cannot provide."""


def apply(
    *,
    environment_details: dict[str, Any],
    logger_: Optional[logging.Logger] = None,
) -> dict[str, str]:
    """Interpret an environment template and return the variables it defines."""
    log = logger_ or logger
    environment_id = environment_details.get("environmentId", "unknown")
    # BatchGetJobEntity returns the definition unwrapped from the `environment` key an
    # authored template nests it under. Both shapes are accepted because the nested form
    # is what a reader sees in `queue_environments/`.
    template = environment_details.get("template") or {}
    environment = template.get("environment") or template
    name = environment.get("name", environment_id)

    if environment.get("script"):
        raise UnsupportedEnvironmentError(
            f"Queue environment '{name}' defines a script, which a Lambda durable "
            f"worker cannot run: it has no persistent session directory and no writable "
            f"filesystem outside /tmp. Remove the environment from this queue, or run "
            f"this job on a fleet whose workers execute scripts."
        )

    variables = environment.get("variables") or {}
    if not isinstance(variables, dict):
        raise UnsupportedEnvironmentError(
            f"Queue environment '{name}' has a 'variables' value that is not a mapping."
        )

    applied = {str(key): str(value) for key, value in variables.items()}
    if applied:
        log.info(f"Applied {len(applied)} variable(s) from queue environment '{name}'")
    else:
        log.info(f"Queue environment '{name}' defines nothing this worker needs to apply")
    return applied
