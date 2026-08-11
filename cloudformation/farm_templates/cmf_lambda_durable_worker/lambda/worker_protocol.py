# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Minimal AWS Deadline Cloud worker protocol client.

The `deadline-cloud-worker-agent` package assumes a long-lived process on a host it
owns: it persists worker IDs to disk, caches credentials in a local file, runs jobs
as OS users in local sessions, and streams logs from background threads. A durable
Lambda worker has none of that, so this module implements the wire protocol directly.

Only the calls a worker must make are covered:

    CreateWorker              -> register with the fleet
    AssumeFleetRoleForWorker  -> get worker-scoped credentials
    UpdateWorker              -> announce STARTED / STOPPING / STOPPED
    UpdateWorkerSchedule      -> heartbeat, receive work, report progress
    DeleteWorker              -> deregister

Every method is a plain request/response call with no local state, which keeps the
caller free to wrap each one in a durable step. See the module docstring in
`durable_worker.py` for how that interacts with checkpoint replay.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

# The worker agent uses adaptive retries against Deadline Cloud so that a fleet
# reconnecting after an outage backs off instead of stampeding the service.
DEADLINE_BOTOCORE_CONFIG = Config(retries={"max_attempts": 5, "mode": "adaptive"})


class WorkerProtocolError(Exception):
    """A worker protocol call failed in a way the caller cannot recover from."""


class WorkerDeletedError(WorkerProtocolError):
    """The service no longer recognizes this worker.

    Deadline Cloud deletes workers that stop heartbeating. A durable execution that
    was suspended for longer than the fleet's tolerance can wake up to find itself
    already gone, in which case the only correct action is to stop cleanly rather
    than keep polling a dead worker ID.
    """


class DeadlineWorker:
    """A Deadline Cloud worker identity plus the calls it makes on its own behalf."""

    def __init__(
        self,
        *,
        farm_id: str,
        fleet_id: str,
        region: str,
        worker_id: Optional[str] = None,
        credentials: Optional[dict[str, Any]] = None,
    ) -> None:
        self.farm_id = farm_id
        self.fleet_id = fleet_id
        self.region = region
        self.worker_id = worker_id
        self._credentials = credentials
        self._client_cache: dict[bool, Any] = {}

    # -- clients ---------------------------------------------------------------

    def _client(self, *, use_worker_credentials: bool):
        """Return a Deadline Cloud client, building it once per credential set.

        Registration happens with the Lambda execution role, which holds only
        `deadline:CreateWorker` and `deadline:AssumeFleetRoleForWorker`. Everything
        afterwards uses the worker-scoped fleet role credentials, mirroring the
        least-privilege split the worker agent uses on EC2.

        Clients are cached because building one is expensive relative to the call it
        makes: constructing a credentialed `boto3.Session` plus client costs roughly
        80ms, and unlike the default session it cannot reuse botocore's warm loader
        cache. A step that assumed the fleet role and then made one API call was
        paying that twice for a single network round trip.
        """
        if use_worker_credentials and not self._credentials:
            raise WorkerProtocolError(
                "Worker credentials are required but have not been obtained yet."
            )

        cached = self._client_cache.get(use_worker_credentials)
        if cached is not None:
            return cached

        if use_worker_credentials:
            assert self._credentials is not None  # guarded above
            session = boto3.Session(
                aws_access_key_id=self._credentials["accessKeyId"],
                aws_secret_access_key=self._credentials["secretAccessKey"],
                aws_session_token=self._credentials["sessionToken"],
                region_name=self.region,
            )
        else:
            session = boto3.Session(region_name=self.region)
        client = session.client("deadline", config=DEADLINE_BOTOCORE_CONFIG)
        self._client_cache[use_worker_credentials] = client
        return client

    # -- lifecycle -------------------------------------------------------------

    def create_worker(self, *, host_name: str) -> str:
        """Register a new worker with the fleet and return its worker ID."""
        client = self._client(use_worker_credentials=False)
        response = client.create_worker(
            farmId=self.farm_id,
            fleetId=self.fleet_id,
            hostProperties={"hostName": host_name},
        )
        self.worker_id = response["workerId"]
        logger.info("Created worker %s in fleet %s", self.worker_id, self.fleet_id)
        return self.worker_id

    def assume_fleet_role(self) -> dict[str, Any]:
        """Fetch worker-scoped credentials for this worker.

        Returns the credentials so the caller can checkpoint them. They are
        short-lived, so a worker that wakes from a long sleep refreshes rather than
        reusing a replayed value.
        """
        client = self._client(use_worker_credentials=False)
        response = client.assume_fleet_role_for_worker(
            farmId=self.farm_id, fleetId=self.fleet_id, workerId=self.worker_id
        )
        credentials = response["credentials"]
        self._credentials = credentials
        # Drop the cached worker client so the next call picks up these credentials
        # rather than continuing to sign with the previous, possibly expired, set.
        self._client_cache.pop(True, None)
        return {
            "accessKeyId": credentials["accessKeyId"],
            "secretAccessKey": credentials["secretAccessKey"],
            "sessionToken": credentials["sessionToken"],
            "expiration": credentials["expiration"].isoformat(),
        }

    def set_credentials(self, credentials: dict[str, Any]) -> None:
        """Restore credentials obtained by an earlier call."""
        self._credentials = credentials
        self._client_cache.pop(True, None)

    def update_worker_status(
        self, *, status: str, capabilities: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Move the worker to STARTED, STOPPING, or STOPPED.

        `capabilities` is required by the service on the transition to STARTED; it
        declares what this worker can run so the scheduler only assigns matching
        steps.
        """
        client = self._client(use_worker_credentials=True)
        request: dict[str, Any] = {
            "farmId": self.farm_id,
            "fleetId": self.fleet_id,
            "workerId": self.worker_id,
            "status": status,
        }
        if capabilities:
            request["capabilities"] = capabilities
        try:
            return client.update_worker(**request)
        except client.exceptions.ResourceNotFoundException as exc:
            raise WorkerDeletedError(f"Worker {self.worker_id} no longer exists") from exc

    def update_worker_schedule(
        self, *, updated_session_actions: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Heartbeat, report action progress, and receive assigned work.

        This single call is the worker's entire scheduling interface. The response
        carries `assignedSessions` (new work), `cancelSessionActions` (work to
        abandon), an optional `desiredWorkerStatus` of STOPPED (the service asking
        this worker to drain), and `updateIntervalSeconds` (how long until the next
        call is expected).
        """
        client = self._client(use_worker_credentials=True)
        try:
            return client.update_worker_schedule(
                farmId=self.farm_id,
                fleetId=self.fleet_id,
                workerId=self.worker_id,
                updatedSessionActions=_deserialize_timestamps(
                    updated_session_actions or {}
                ),
            )
        except client.exceptions.ResourceNotFoundException as exc:
            raise WorkerDeletedError(f"Worker {self.worker_id} no longer exists") from exc

    def delete_worker(self) -> None:
        """Deregister the worker. Safe to call when the worker is already gone."""
        client = self._client(use_worker_credentials=True)
        try:
            client.delete_worker(
                farmId=self.farm_id, fleetId=self.fleet_id, workerId=self.worker_id
            )
            logger.info("Deleted worker %s", self.worker_id)
        except client.exceptions.ResourceNotFoundException:
            logger.info("Worker %s was already deleted", self.worker_id)


def _deserialize_timestamps(
    updated_session_actions: dict[str, Any],
) -> dict[str, Any]:
    """Convert ISO-8601 timestamp strings back into datetime objects for botocore.

    Session action results travel through durable checkpoints, which hold JSON, so
    timestamps are carried as strings. The Deadline Cloud API models `startedAt` and
    `endedAt` as timestamps, and botocore will not serialize a bare string for a
    timestamp member, so they are converted back here at the boundary.
    """
    from datetime import datetime

    converted: dict[str, Any] = {}
    for action_id, update in updated_session_actions.items():
        entry = dict(update)
        for field in ("startedAt", "endedAt", "updatedAt"):
            value = entry.get(field)
            if isinstance(value, str):
                entry[field] = datetime.fromisoformat(value)
        converted[action_id] = entry
    return converted


def default_capabilities() -> dict[str, Any]:
    """Capabilities describing a Lambda-hosted worker.

    The amounts are deliberately modest: this worker forwards API calls rather than
    rendering locally, so what matters is the custom `durable-lambda` attribute that
    job templates target to ensure their steps land on this kind of worker.
    """
    return {
        "amounts": [
            {"name": "amount.worker.vcpu", "value": 1},
            {"name": "amount.worker.memory", "value": 2048},
            {"name": "amount.worker.disk.scratch", "value": 0},
            {"name": "amount.worker.gpu", "value": 0},
        ],
        "attributes": [
            {"name": "attr.worker.os.family", "values": ["linux"]},
            {"name": "attr.worker.cpu.arch", "values": ["x86_64"]},
            {"name": "attr.durable.lambda", "values": ["true"]},
        ],
    }
