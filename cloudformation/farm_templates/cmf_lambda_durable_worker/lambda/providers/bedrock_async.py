# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Amazon Bedrock asynchronous invocation. The only Bedrock-aware module in the worker.

The handle is the `invocationArn` the task's own script got back from `StartAsyncInvoke`,
so the model, its input, and the output location are all the job template's business.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "us-west-2")

INVOCATION_STATES = {"Completed": "SUCCEEDED", "Failed": "FAILED", "InProgress": "RUNNING"}


@lru_cache(maxsize=1)
def _client():
    # No in-process retries: botocore's backoff sleeps inside the call, and that sleep is
    # billed compute. Cached because building a credentialed client costs roughly 80ms.
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(retries={"max_attempts": 1, "mode": "standard"}),
    )


def poll(handle: Any) -> dict[str, Any]:
    try:
        response = _client().get_async_invoke(invocationArn=handle)
    except ClientError as exc:
        # A failed status read is not evidence the invocation failed, so keep waiting.
        message = exc.response.get("Error", {}).get("Message", str(exc))
        logger.warning("GetAsyncInvoke failed, will retry: %s", message)
        return {"state": "RUNNING", "message": f"Status unavailable: {message}"}

    status = response["status"]
    state = INVOCATION_STATES.get(status, "RUNNING")
    if state == "SUCCEEDED":
        output_uri = (
            response.get("outputDataConfig", {})
            .get("s3OutputDataConfig", {})
            .get("s3Uri", "")
        )
        return {
            "state": state,
            "outputUri": output_uri,
            "message": f"Output written to {output_uri or 'S3'}",
        }
    if state == "FAILED":
        return {"state": state, "message": response.get("failureMessage", "Invocation failed")}
    return {"state": state, "message": f"Invocation {status}"}
