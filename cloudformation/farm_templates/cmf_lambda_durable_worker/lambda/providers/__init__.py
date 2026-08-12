# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Registry of the long-running APIs a task can dispatch to.

A provider is a module with exactly two functions. Every value is plain JSON, because
results cross durable checkpoint boundaries:

    submit(request: dict, *, task_id: str) -> {"handle": <any JSON>}
                                            | {"error": str, "retryable": bool}
    poll(handle) -> {"state": "RUNNING" | "SUCCEEDED" | "FAILED",
                     "message": str, "outputUri": str}     # last two optional

`request` is whatever the job template put in the task's `Request` parameter, and
`handle` is whatever the provider needs to identify the request it started. Neither is
inspected by the worker.

Providers never sleep, retry, or count attempts: they report `retryable` and the worker
decides the timing, because only the worker can wait without being billed for it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

PROVIDER_MODULES = {
    "bedrock-async": "providers.bedrock_async",
    "sleep": "providers.sleep",
}


class UnknownProviderError(Exception):
    """A task named a provider that is not registered."""


def known_providers() -> list[str]:
    return sorted(PROVIDER_MODULES)


def resolve(name: str) -> ModuleType:
    """Import the named provider.

    Imported on use so that resolving one provider never loads another's SDK.
    """
    module_path = PROVIDER_MODULES.get(name)
    if module_path is None:
        raise UnknownProviderError(
            f"Unknown provider {name!r}. Known providers: {', '.join(known_providers())}."
        )
    return importlib.import_module(module_path)
