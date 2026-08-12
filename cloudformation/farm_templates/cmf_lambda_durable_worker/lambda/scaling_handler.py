# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Turn Deadline Cloud fleet size recommendations into durable worker executions.

A customer-managed fleet in `EVENT_BASED_AUTO_SCALING` mode emits a "Fleet Size
Recommendation Change" event but never starts or stops workers itself. Here one worker is
one durable execution: scaling out invokes more executions, scaling in sets a drain flag.

Draining rather than stopping is deliberate. `StopDurableExecution` would strand a worker
mid-task, leaving the third-party request running and the task never reported. The
recommendation says only how many workers to run, not which to stop, so this handler picks
the newest workers, which are least likely to hold a long-running request.
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
    """Return registered workers for this fleet that are not already draining."""
    response = table.query(
        KeyConditionExpression="fleetId = :f",
        ExpressionAttributeValues={":f": fleet_id},
    )
    return [item for item in response.get("Items", []) if not item.get("drain")]


def _fleet_worker_count(*, farm_id: str, fleet_id: str) -> int:
    """Count workers Deadline Cloud currently has for this fleet.

    The registry can undercount, and a worker it never recorded still occupies fleet
    capacity, so scale-out is bounded by this rather than by the registry. NOT_RESPONDING
    workers count too; only deleted workers stop appearing here.
    """
    deadline = boto3.client("deadline")
    paginator = deadline.get_paginator("list_workers")
    count = 0
    for page in paginator.paginate(farmId=farm_id, fleetId=fleet_id):
        count += len(page.get("workers", []))
    return count


def _headroom(*, farm_id: str, fleet_id: str) -> int:
    """Return how many more workers the fleet has room for.

    A failed count allows the request: CreateWorker enforces `maxWorkerCount` itself, so
    failing open beats stalling the fleet at zero workers.
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
        # Deliberately not idempotent: deriving the name from the event id would make a
        # genuine second scale-out a no-op. Over-starting is bounded by the headroom
        # check and self-corrects when idle workers time out.
        execution_name = f"worker-{uuid.uuid4().hex[:16]}"
        try:
            lambda_client.invoke(
                FunctionName=WORKER_FUNCTION_ARN,
                # Event, not RequestResponse: a synchronous invoke is capped at 15
                # minutes, while a durable execution may run for up to a year.
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
