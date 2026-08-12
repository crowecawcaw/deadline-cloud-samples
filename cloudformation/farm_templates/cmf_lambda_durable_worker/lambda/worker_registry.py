# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The shared registry of live durable workers, one row per worker.

Keyed by (fleetId, workerId), with a `startedAt` that scale-in ranks drain candidates by
and a `drain` flag the scaling handler sets and each worker polls on its heartbeat.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

REGISTRY_TABLE = os.environ.get("REGISTRY_TABLE", "")

# Comfortably longer than any worker should live, so expiry only catches abandoned rows.
REGISTRY_TTL_SECONDS = int(os.environ.get("REGISTRY_TTL_SECONDS", str(48 * 3600)))


@lru_cache(maxsize=1)
def _table():
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
                # A row that is never cleaned up counts against fleet capacity forever
                # and would quietly stop the fleet from scaling out again.
                "expiresAt": _expiry_epoch(),
            }
        )
    except ClientError as exc:
        # Best effort: a registry failure must not fail a worker the service accepted.
        logger.error(f"Failed to add {worker_id} to the registry: {exc}")


def should_drain(*, fleet_id: str, worker_id: str) -> bool:
    """Return whether the scaling handler has asked this worker to stop."""
    try:
        response = _table().get_item(
            Key={"fleetId": fleet_id, "workerId": worker_id},
            ConsistentRead=True,
        )
    except ClientError as exc:
        # Draining on a transient read error would shrink the fleet for the wrong reason.
        logger.error(f"Failed to read drain flag for {worker_id}: {exc}")
        return False
    return bool(response.get("Item", {}).get("drain", False))


def deregister(*, fleet_id: str, worker_id: str) -> None:
    """Remove this worker from the registry once it has stopped."""
    try:
        _table().delete_item(Key={"fleetId": fleet_id, "workerId": worker_id})
    except ClientError as exc:
        logger.error(f"Failed to remove {worker_id} from the registry: {exc}")


def _expiry_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp()) + REGISTRY_TTL_SECONDS


def utc_now_iso() -> str:
    """The current UTC time in ISO-8601 form. Only ever called inside a durable step."""
    return datetime.now(timezone.utc).isoformat()
