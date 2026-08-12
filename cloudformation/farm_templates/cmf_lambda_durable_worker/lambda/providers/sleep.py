# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A fake long-running request that needs no credentials.

The handle is `{"finishAt": <epoch seconds>}`, which a job template's script can produce
with `date`, so the whole await path is exercisable from a template with no SDK at all.
An optional `failMessage` makes the failure path reachable too.
"""

from __future__ import annotations

import time
from typing import Any


def poll(handle: Any) -> dict[str, Any]:
    remaining = float(handle["finishAt"]) - time.time()
    if remaining > 0:
        return {"state": "RUNNING", "message": f"{remaining:.0f}s remaining"}
    if handle.get("failMessage"):
        return {"state": "FAILED", "message": handle["failMessage"]}
    return {
        "state": "SUCCEEDED",
        "outputUri": handle.get("outputUri", ""),
        "message": f"Slept until {float(handle['finishAt']):.0f}",
    }
