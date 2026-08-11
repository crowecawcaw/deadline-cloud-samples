# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Turn Deadline Cloud fleet size recommendations into durable worker executions.

A customer-managed fleet in `EVENT_BASED_AUTO_SCALING` mode emits an EventBridge
event whenever its recommended size changes:

    {"detail-type": "Fleet Size Recommendation Change",
     "source": "aws.deadline",
     "detail": {"farmId": ..., "fleetId": ..., "oldFleetSize": 1, "newFleetSize": 5}}

Deadline Cloud never starts or stops workers itself for a customer-managed fleet, so
this handler owns that. Where the documented Amazon EC2 pattern forwards the
recommendation to an Auto Scaling group, here one worker is one durable execution:
scaling out invokes more executions, scaling in asks some of them to drain.

Scaling in without abandoning work
----------------------------------
`StopDurableExecution` would strand a worker mid-task: the Bedrock request would keep
running, the task would never be reported, and the Deadline worker would stay
registered until the service timed it out. Because the recommendation says only *how
many* workers to run and not *which* ones to stop, this handler keeps a small registry
of live workers and sets a drain flag on the ones it selects. Each worker checks its
own flag on its next heartbeat and then shuts down through the proper
STOPPING/STOPPED/DeleteWorker path.

Newest workers are drained first. They are least likely to hold a long-running
request, so draining them sheds capacity soonest.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

WORKER_FUNCTION_ARN = os.environ["WORKER_FUNCTION_ARN"]
REGISTRY_TABLE = os.environ["REGISTRY_TABLE"]
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))

lambda_client = boto3.client("lambda")
dynamodb = boto3.resource("dynamodb")


def lambda_handler(event: dict, context) -> dict:
    """Reconcile the number of running workers with the recommended fleet size."""
    logger.info(json.dumps(event))
    detail = event["detail"]
    fleet_id = detail["fleetId"]
    desired = min(int(detail["newFleetSize"]), MAX_WORKERS)

    table = dynamodb.Table(REGISTRY_TABLE)
    active = _active_workers(table, fleet_id)
    current = len(active)

    if desired > current:
        # Never ask for more workers than the fleet has room for. The fleet's own
        # worker count is the binding constraint, not the registry's.
        headroom = _headroom(farm_id=detail["farmId"], fleet_id=fleet_id)
        wanted = desired - current
        if headroom < wanted:
            logger.warning(
                f"Fleet {fleet_id} has room for {headroom} more worker(s) but {wanted} "
                f"were requested; starting {headroom}. Stale workers may still be "
                f"occupying fleet capacity."
            )
        started = _scale_out(fleet_id=fleet_id, count=min(wanted, headroom))
        action = f"started {started} worker(s)"
    elif desired < current:
        drained = _scale_in(table=table, active=active, count=current - desired)
        action = f"marked {drained} worker(s) to drain"
    else:
        action = "no change"

    logger.info(f"Fleet {fleet_id}: {current} running, {desired} recommended; {action}")
    return {
        "statusCode": 200,
        "body": json.dumps(
            {"fleetId": fleet_id, "current": current, "desired": desired, "action": action}
        ),
    }


def _active_workers(table, fleet_id: str) -> list[dict]:
    """Return registered workers for this fleet that are not already draining.

    The registry is the handler's own view of its workers. A worker writes its entry
    as part of registering and removes it as part of deregistering.
    """
    response = table.query(
        KeyConditionExpression="fleetId = :f",
        ExpressionAttributeValues={":f": fleet_id},
    )
    return [item for item in response.get("Items", []) if not item.get("drain")]


def _fleet_worker_count(*, farm_id: str, fleet_id: str) -> int:
    """Count workers Deadline Cloud currently has for this fleet.

    Deadline Cloud is the authority on how many workers exist, and the registry is
    not: a worker that registered but then failed before writing its row, or one
    whose registry write failed, is invisible to the registry yet still counts
    against the fleet's `maxWorkerCount`. Starting more workers on the strength of an
    empty registry produces a ConflictException at CreateWorker, so scale-out is
    bounded by this count rather than by the registry.
    """
    deadline = boto3.client("deadline")
    paginator = deadline.get_paginator("list_workers")
    count = 0
    for page in paginator.paginate(farmId=farm_id, fleetId=fleet_id):
        # NOT_RESPONDING workers still occupy fleet capacity, so they are counted.
        # Only fully deleted workers stop counting, and those no longer appear here.
        count += len(page.get("workers", []))
    return count


def _headroom(*, farm_id: str, fleet_id: str) -> int:
    """Return how many more workers the fleet has room for.

    A failure to count is not a reason to drop a scaling event: CreateWorker enforces
    `maxWorkerCount` on its own, so the fallback is to allow the request and let a
    genuinely over-capacity start fail there instead of stalling the fleet at zero
    workers.
    """
    try:
        return max(0, MAX_WORKERS - _fleet_worker_count(farm_id=farm_id, fleet_id=fleet_id))
    except ClientError as exc:
        logger.warning(f"Could not count fleet workers, proceeding without a cap: {exc}")
        return MAX_WORKERS


def _scale_out(*, fleet_id: str, count: int) -> int:
    """Start `count` new durable worker executions."""
    started = 0
    for _ in range(count):
        # A random name is deliberately NOT idempotent: a redelivered event starts
        # another worker. Deriving the name from the event id would make retries
        # idempotent, but then a genuine second scale-out for the same recommendation
        # could not start a worker either. Over-starting is bounded by the headroom
        # check above and self-corrects when idle workers time out, so the simpler
        # random name is the better trade here.
        execution_name = f"worker-{uuid.uuid4().hex[:16]}"
        try:
            lambda_client.invoke(
                FunctionName=WORKER_FUNCTION_ARN,
                # Event, not RequestResponse: a synchronous invoke would be capped at
                # 15 minutes, while an asynchronous durable execution may run for up
                # to a year.
                InvocationType="Event",
                DurableExecutionName=execution_name,
                Payload=json.dumps({"hostName": execution_name, "fleetId": fleet_id}),
            )
            started += 1
        except ClientError as exc:
            logger.error(f"Failed to start worker {execution_name}: {exc}")
    return started


def _scale_in(*, table, active: list[dict], count: int) -> int:
    """Flag the `count` newest workers to drain on their next heartbeat."""
    newest_first = sorted(active, key=lambda item: item.get("startedAt", ""), reverse=True)
    drained = 0
    for item in newest_first[:count]:
        try:
            table.update_item(
                Key={"fleetId": item["fleetId"], "workerId": item["workerId"]},
                UpdateExpression="SET drain = :true",
                ExpressionAttributeValues={":true": True},
            )
            logger.info(f"Marked worker {item['workerId']} to drain")
            drained += 1
        except ClientError as exc:
            logger.error(f"Failed to mark {item['workerId']} to drain: {exc}")
    return drained
