# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Minimal AWS Deadline Cloud worker protocol client.

`deadline-cloud-worker-agent` assumes a long-lived process on a host it owns, so this
implements the wire protocol directly instead: CreateWorker, AssumeFleetRoleForWorker,
UpdateWorker, UpdateWorkerSchedule, BatchGetJobEntity, DeleteWorker. Every method is a
plain request/response call, which leaves callers free to wrap each one in a durable step.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import boto3
import botocore.session
from botocore.config import Config
from botocore.credentials import RefreshableCredentials

logger = logging.getLogger(__name__)

# Few attempts on purpose: botocore's backoff sleeps inside the call, and in Lambda that
# sleep is billed compute. Long backoff belongs behind a durable wait, which is unbilled.
DEADLINE_BOTOCORE_CONFIG = Config(retries={"max_attempts": 2, "mode": "standard"})


class WorkerProtocolError(Exception):
    """A worker protocol call failed in a way the caller cannot recover from."""


class WorkerNotUsableError(WorkerProtocolError):
    """This worker can no longer do work, whatever the reason.

    Covers a deleted worker and one the service has taken out of STARTED. Both have the
    same remedy, which is to stop cleanly and deregister rather than keep polling.
    """


class WorkerDeletedError(WorkerNotUsableError):
    """The service no longer recognizes this worker."""


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
        self._client_cache: dict[bool, Any] = {}
        self._worker_credentials: Optional[RefreshableCredentials] = None
        if credentials:
            self.set_credentials(credentials)

    # -- clients ---------------------------------------------------------------

    def _bootstrap_client(self):
        """A client using the Lambda execution role, which holds only CreateWorker and
        AssumeFleetRoleForWorker."""
        client = self._client_cache.get(False)
        if client is None:
            client = boto3.Session(region_name=self.region).client(
                "deadline", config=DEADLINE_BOTOCORE_CONFIG
            )
            self._client_cache[False] = client
        return client

    def _worker_client(self):
        """A client signing with the worker-scoped fleet role, assumed on first use.

        Cached: with refreshable credentials a client stays usable for the worker's whole
        life, and building one costs roughly 80ms.
        """
        if self._worker_credentials is None:
            if not self.worker_id:
                raise WorkerProtocolError(
                    "A worker ID is required before the fleet role can be assumed."
                )
            self.assume_fleet_role()
        client = self._client_cache.get(True)
        if client is None:
            botocore_session = botocore.session.get_session()
            botocore_session._credentials = self._worker_credentials
            client = boto3.Session(
                botocore_session=botocore_session, region_name=self.region
            ).client("deadline", config=DEADLINE_BOTOCORE_CONFIG)
            self._client_cache[True] = client
        return client

    def _client(self, *, use_worker_credentials: bool):
        return self._worker_client() if use_worker_credentials else self._bootstrap_client()

    def _fetch_fleet_role_credentials(self) -> dict[str, Any]:
        """Call AssumeFleetRoleForWorker, keyed the way RefreshableCredentials expects."""
        response = self._bootstrap_client().assume_fleet_role_for_worker(
            farmId=self.farm_id, fleetId=self.fleet_id, workerId=self.worker_id
        )
        credentials = response["credentials"]
        logger.info("Obtained fleet role credentials for worker %s", self.worker_id)
        return {
            "access_key": credentials["accessKeyId"],
            "secret_key": credentials["secretAccessKey"],
            "token": credentials["sessionToken"],
            "expiry_time": credentials["expiration"].isoformat(),
        }

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

    def assume_fleet_role(self) -> None:
        """Obtain worker-scoped credentials that botocore renews itself.

        Never checkpointed: credentials are far shorter-lived than a durable execution,
        so a replayed copy would usually be expired.
        """
        self._worker_credentials = RefreshableCredentials.create_from_metadata(
            metadata=self._fetch_fleet_role_credentials(),
            refresh_using=self._fetch_fleet_role_credentials,
            method="deadline-assume-fleet-role-for-worker",
        )

    def set_credentials(self, credentials: dict[str, Any]) -> None:
        """Adopt credentials in the API's own spelling, keeping them refreshable."""
        self._worker_credentials = RefreshableCredentials.create_from_metadata(
            metadata={
                "access_key": credentials["accessKeyId"],
                "secret_key": credentials["secretAccessKey"],
                "token": credentials["sessionToken"],
                "expiry_time": _as_iso(credentials["expiration"]),
            },
            refresh_using=self._fetch_fleet_role_credentials,
            method="deadline-assume-fleet-role-for-worker",
        )
        self._client_cache.pop(True, None)

    def update_worker_status(
        self, *, status: str, capabilities: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Move the worker to STARTED, STOPPING, or STOPPED.

        `capabilities` is required on the transition to STARTED.
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

        The response carries `assignedSessions`, `cancelSessionActions`, an optional
        `desiredWorkerStatus`, and `updateIntervalSeconds`.
        """
        client = self._client(use_worker_credentials=True)
        try:
            return client.update_worker_schedule(
                farmId=self.farm_id,
                fleetId=self.fleet_id,
                workerId=self.worker_id,
                # botocore serializes ISO-8601 strings for timestamp members, so results
                # can cross a JSON checkpoint boundary without conversion.
                updatedSessionActions=updated_session_actions or {},
            )
        except client.exceptions.ResourceNotFoundException as exc:
            raise WorkerDeletedError(f"Worker {self.worker_id} no longer exists") from exc
        except client.exceptions.ConflictException as exc:
            raise WorkerNotUsableError(
                f"Worker {self.worker_id} is no longer in the STARTED status"
            ) from exc

    def get_environment_details(self, *, job_id: str, environment_id: str) -> dict[str, Any]:
        """Fetch the Open Job Description template for a queue environment.

        Only environment details are requested: this worker runs no OpenJD scripts and
        stages no files, so it needs neither job details nor step templates.
        """
        client = self._client(use_worker_credentials=True)
        response = client.batch_get_job_entity(
            farmId=self.farm_id,
            fleetId=self.fleet_id,
            workerId=self.worker_id,
            identifiers=[
                {"environmentDetails": {"jobId": job_id, "environmentId": environment_id}}
            ],
        )
        for error in response.get("errors", []):
            details = error.get("environmentDetails")
            if details:
                raise WorkerProtocolError(
                    f"Could not get environment {environment_id}: "
                    f"{details['code']}: {details['message']}"
                )
        for entity in response.get("entities", []):
            if "environmentDetails" in entity:
                return entity["environmentDetails"]
        raise WorkerProtocolError(
            f"Environment {environment_id} was neither returned nor reported as an error"
        )

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


def _as_iso(expiration: Any) -> str:
    """Accept an expiry as either a datetime or an ISO-8601 string."""
    return expiration if isinstance(expiration, str) else expiration.isoformat()


def default_capabilities() -> dict[str, Any]:
    """Capabilities describing a Lambda-hosted worker.

    The amounts are modest because this worker forwards API calls. What matters is the
    custom `attr.durable.lambda` attribute that job templates target.
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
