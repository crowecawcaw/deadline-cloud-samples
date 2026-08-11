# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Queue environment support for a worker that cannot run scripts.

Deadline Cloud wraps a task in `envEnter` and `envExit` session actions, one pair per
queue environment. On a conventional worker the agent fetches each environment's Open Job
Description template and runs its `onEnter` script inside the session, which is how a
Conda or Rez environment installs software before the task runs.

A Lambda worker cannot do that. There is no persistent session directory, the filesystem
is read-only outside `/tmp`, the sandbox is discarded between invocations, and the tools
those scripts drive are not present. Every queue environment in
[`queue_environments/`](../../../queue_environments/) is script-based for exactly this
reason: their job is to install software onto a host.

So this module supports the part of the specification that is meaningful here and refuses
the part that is not:

* An environment that only defines `variables` is applied. The variables become part of
  the session's environment and are visible to the task, which is enough for
  environments that pass configuration rather than install software.
* An environment with a `script` fails the `envEnter` action with an explanation.

Failing is the important half. The previous behavior reported success without running
anything, so a queue environment that was supposed to install software silently did
nothing and the task ran in an environment that had never been prepared. A failed
`envEnter` stops the session, which surfaces the misconfiguration instead of hiding it.
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
    """Interpret an environment template and return the variables it defines.

    Raises UnsupportedEnvironmentError when the environment has a script, because
    running it is what this worker cannot do.
    """
    log = logger_ or logger
    environment_id = environment_details.get("environmentId", "unknown")
    # BatchGetJobEntity returns the environment definition itself, already unwrapped
    # from the `environment` key an authored template nests it under. Verified against
    # the service: a queue environment authored as
    # `{"environment": {"name": ..., "variables": {...}}}` comes back as
    # `{"name": ..., "variables": {...}}`. Both shapes are accepted, because the nested
    # form is what a reader sees in `queue_environments/` and will reasonably expect.
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

    # Values are coerced to strings because that is what an environment variable is, and
    # a template may legitimately carry a number.
    applied = {str(key): str(value) for key, value in variables.items()}
    if applied:
        log.info(f"Applied {len(applied)} variable(s) from queue environment '{name}'")
    else:
        log.info(f"Queue environment '{name}' defines nothing this worker needs to apply")
    return applied
