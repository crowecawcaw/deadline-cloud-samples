# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Amazon Bedrock asynchronous invocation. The only Bedrock-aware module in the worker.

`modelId` and `modelInput` are passed to `StartAsyncInvoke` verbatim, so a job template
can change model or generation settings without a code change.
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

OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
# The stack's bucket lifecycle rule expires this prefix; change both together.
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "generated")
REGION = os.environ.get("AWS_REGION", "us-west-2")

# "Try again later" rather than "this request is bad". Per-account concurrency limits for
# generation models are low, so a fleet of several workers hits them routinely.
RETRYABLE_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceQuotaExceededException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelNotReadyException",
        "ModelTimeoutException",
    }
)

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


def submit(request: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    """Start an asynchronous invocation, handled by its invocation ARN."""
    model_id = request.get("modelId")
    model_input = request.get("modelInput")
    if not model_id or not isinstance(model_input, dict):
        return {
            "error": "A bedrock-async request needs a 'modelId' string and a "
            "'modelInput' object.",
            "retryable": False,
        }

    # The task ID keeps concurrent workers from colliding in S3 and keeps the location
    # identical across replays.
    output_uri = f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}/{task_id}/"
    logger.info("Starting %s asynchronous invocation to %s", model_id, output_uri)
    try:
        response = _client().start_async_invoke(
            modelId=model_id,
            modelInput=model_input,
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_uri}},
        )
    except ClientError as exc:
        error = exc.response.get("Error", {})
        return {
            "error": error.get("Message", str(exc)),
            "retryable": error.get("Code", "") in RETRYABLE_ERROR_CODES,
        }
    return {"handle": response["invocationArn"]}


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
