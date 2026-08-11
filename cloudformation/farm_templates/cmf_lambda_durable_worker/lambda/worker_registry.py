# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The shared registry of live durable workers.

The scaling handler needs to know how many workers are running and needs a way to ask
specific ones to stop. Deadline Cloud's fleet size recommendation says only how many
workers should exist, and a customer-managed fleet leaves worker lifecycle entirely to
the fleet owner, so this table is what connects the two halves of the sample.

One row per worker, keyed by (fleetId, workerId):

    fleetId    partition key, so a scaling event can query just its own fleet
    workerId   sort key, the Deadline Cloud worker ID
    startedAt  ISO-8601 registration time, used to pick drain candidates
    drain      set by the scaling handler; the worker shuts down when it sees this

The worker writes its row when it registers, polls `drain` on each heartbeat, and
deletes its row when it deregisters.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

REGISTRY_TABLE = os.environ.get("REGISTRY_TABLE", "")


@lru_cache(maxsize=1)
def _table():
    """Return the registry table, built once per execution environment."""
    return boto3.resource("dynamodb").Table(REGISTRY_TABLE)


def register(*, fleet_id: str, worker_id: str, started_at: str) -> None:
    """Add this worker to the registry so it is counted and can be drained."""
    try:
        _table().put_item(
            Item={
                "fleetId": fleet_id,
                "workerId": worker_id,
                "startedAt": started_at,
                "drain": False,
            }
        )
    except ClientError as exc:
        # A registry write failure must not take down a worker that registered
        # successfully with Deadline Cloud. The cost is that this worker is invisible
        # to scaling decisions until it exits.
        logger.error(f"Failed to add {worker_id} to the registry: {exc}")


def should_drain(*, fleet_id: str, worker_id: str) -> bool:
    """Return whether the scaling handler has asked this worker to stop."""
    try:
        response = _table().get_item(
            Key={"fleetId": fleet_id, "workerId": worker_id},
            ConsistentRead=True,
        )
    except ClientError as exc:
        # Treat an unreadable registry as "keep working". Draining on a transient
        # read error would shrink the fleet for the wrong reason.
        logger.error(f"Failed to read drain flag for {worker_id}: {exc}")
        return False
    return bool(response.get("Item", {}).get("drain", False))


def deregister(*, fleet_id: str, worker_id: str) -> None:
    """Remove this worker from the registry once it has stopped."""
    try:
        _table().delete_item(Key={"fleetId": fleet_id, "workerId": worker_id})
    except ClientError as exc:
        logger.error(f"Failed to remove {worker_id} from the registry: {exc}")


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 form.

    Only ever called inside a durable step. A clock read is non-deterministic, so
    calling it during replay outside a checkpoint would change the replayed value.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
