# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A fake long-running job that needs no credentials.

Useful for exercising the worker end to end, and for showing what a provider has to
supply: a request shape, an opaque handle, and three states.

Request fields: `seconds` (how long to run), optional `outputUri`, optional
`failMessage` to finish FAILED instead of SUCCEEDED.
"""

from __future__ import annotations

import time
from typing import Any


def submit(request: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    """Handle is the epoch second at which this fake job finishes."""
    try:
        seconds = float(request.get("seconds", 0))
    except (TypeError, ValueError):
        return {
            "error": f"'seconds' must be a number, got {request.get('seconds')!r}",
            "retryable": False,
        }
    return {
        "handle": {
            "taskId": task_id,
            "finishAt": time.time() + seconds,
            "outputUri": request.get("outputUri", ""),
            "failMessage": request.get("failMessage", ""),
        }
    }


def poll(handle: Any) -> dict[str, Any]:
    remaining = handle["finishAt"] - time.time()
    if remaining > 0:
        return {"state": "RUNNING", "message": f"{remaining:.0f}s remaining"}
    if handle["failMessage"]:
        return {"state": "FAILED", "message": handle["failMessage"]}
    return {
        "state": "SUCCEEDED",
        "outputUri": handle["outputUri"],
        "message": f"Slept until {handle['finishAt']:.0f}",
    }
