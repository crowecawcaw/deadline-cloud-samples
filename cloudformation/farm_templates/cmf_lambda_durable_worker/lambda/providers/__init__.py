# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Registry of the long-running requests a task can hand back to the worker to await.

A provider is a module with exactly one function. Every value is plain JSON, because
results cross durable checkpoint boundaries:

    poll(handle) -> {"state": "RUNNING" | "SUCCEEDED" | "FAILED",
                     "message": str, "outputUri": str}     # last two optional

The task's own `onRun` script starts the request and names the provider and handle in a
`durable_lambda_await:` token. `handle` is whatever that provider needs to recognize the
request again; the worker never inspects it.

Providers never sleep, retry, or count attempts: the worker decides the timing, because
only the worker can wait without being billed for it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

PROVIDER_MODULES = {
    "bedrock-async": "providers.bedrock_async",
    "sleep": "providers.sleep",
}


class UnknownProviderError(Exception):
    """An await token named a provider that is not registered."""


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
