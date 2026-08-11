# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Translate Deadline Cloud task parameters into asynchronous Amazon Bedrock requests.

This is the "call another service" half of the sample. A task carries the description
of the request to make, and this module starts it and reports whether it has finished.
Nothing here blocks: `start_generation` returns as soon as Bedrock accepts the
request, and `check_generation` reports status once. The waiting between those two is
the durable function's job, which is what lets the worker sleep through it.

Bedrock's asynchronous invocation path is used deliberately. `StartAsyncInvoke`
returns an invocation ARN immediately and writes output to S3 minutes later, which is
the shape that makes a sleeping worker worthwhile. Note that Bedrock's *image*
models, such as Nova Canvas, are synchronous-only via `InvokeModel` and return their
result inline in seconds; they have no async path to poll. Video generation is used
here because it is the generation workload Bedrock actually exposes asynchronously.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

# How long to sleep between `GetAsyncInvoke` polls, and how many polls to allow.
# The default pair tolerates a request taking roughly an hour. Because each wait
# suspends the execution, a longer interval costs nothing extra; it only changes how
# soon the worker notices completion.
GENERATION_POLL_SECONDS = int(os.environ.get("GENERATION_POLL_SECONDS", "30"))
MAX_GENERATION_POLLS = int(os.environ.get("MAX_GENERATION_POLLS", "120"))

OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "luma.ray-v2:0")
REGION = os.environ.get("AWS_REGION", "us-west-2")

# Errors that mean "try again later" rather than "this request is bad". Bedrock's
# per-account concurrency limits for generation models are low, so a fleet that scales
# out to several workers will routinely hit them.
_RETRYABLE_ERROR_CODES = frozenset(
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


def _client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def build_model_input(task_parameters: dict[str, Any]) -> dict[str, Any]:
    """Build the model payload from a task's parameters.

    `modelInput` is model-specific and passed through by Bedrock unchanged, so this
    mapping is what a job template controls when it describes the call to make.
    """
    prompt = task_parameters.get("Prompt") or "a calm ocean at sunrise, cinematic"
    model_input: dict[str, Any] = {
        "prompt": prompt,
        "duration": task_parameters.get("Duration") or "5s",
        "resolution": task_parameters.get("Resolution") or "540p",
        "aspect_ratio": task_parameters.get("AspectRatio") or "16:9",
        "loop": False,
    }
    return model_input


def start_generation(
    *, task_parameters: dict[str, Any], logger: Optional[logging.Logger] = None
) -> dict[str, Any]:
    """Start an asynchronous generation request and return its invocation ARN.

    Most errors are returned rather than raised so the caller can fail the task with a
    useful message instead of failing the whole durable execution: a malformed prompt
    should cost one task, not the worker.

    Throttling is the exception. Bedrock's concurrency limits are low enough that a
    fleet scaling out will hit them, and a throttle means "try later", not "this task
    is bad". Raising lets the durable step's built-in retry with backoff handle it,
    which suspends the execution between attempts at no compute cost. Returning an
    error instead would fail tasks that would have succeeded moments later.
    """
    log = logger or logging.getLogger(__name__)
    model_id = task_parameters.get("ModelId") or DEFAULT_MODEL_ID
    model_input = build_model_input(task_parameters)

    # Give each task its own prefix so concurrent workers cannot collide in S3.
    task_id = task_parameters.get("TaskId") or "task"
    s3_uri = f"s3://{OUTPUT_BUCKET}/generated/{task_id}/"

    log.info(f"Starting {model_id} generation for prompt: {model_input['prompt'][:120]}")
    try:
        response = _client().start_async_invoke(
            modelId=model_id,
            modelInput=model_input,
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": s3_uri}},
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        message = exc.response.get("Error", {}).get("Message", str(exc))
        if code in _RETRYABLE_ERROR_CODES:
            # Propagate so the durable step retries with backoff instead of failing
            # a task that is only temporarily blocked.
            log.warning(f"StartAsyncInvoke throttled, will retry: {message}")
            raise
        log.error(f"StartAsyncInvoke failed: {message}")
        return {"invocationArn": None, "error": message}

    invocation_arn = response["invocationArn"]
    log.info(f"Started generation {invocation_arn}")
    return {"invocationArn": invocation_arn, "modelId": model_id, "outputUri": s3_uri}


def check_generation(
    *, invocation_arn: str, logger: Optional[logging.Logger] = None
) -> dict[str, Any]:
    """Report whether a generation request has finished.

    Returns a status of `InProgress`, `Completed`, or `Failed`. A transient API error
    is reported as `InProgress` so the caller retries on its next poll instead of
    failing a task that is probably still running.
    """
    log = logger or logging.getLogger(__name__)
    try:
        response = _client().get_async_invoke(invocationArn=invocation_arn)
    except ClientError as exc:
        message = exc.response.get("Error", {}).get("Message", str(exc))
        log.warning(f"GetAsyncInvoke failed, will retry: {message}")
        return {"status": "InProgress", "transientError": message}

    status = response["status"]
    result: dict[str, Any] = {"status": status}
    if status == "Completed":
        result["outputUri"] = (
            response.get("outputDataConfig", {})
            .get("s3OutputDataConfig", {})
            .get("s3Uri", "")
        )
        log.info(f"Generation completed: {result['outputUri']}")
    elif status == "Failed":
        result["failureMessage"] = response.get("failureMessage", "unknown failure")
        log.error(f"Generation failed: {result['failureMessage']}")
    return result
