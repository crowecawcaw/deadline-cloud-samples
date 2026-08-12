# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Minimal AWS Deadline Cloud worker protocol client.

`deadline-cloud-worker-agent` assumes a long-lived process on a host it owns, so this
implements the wire protocol directly instead: CreateWorker, AssumeFleetRoleForWorker,
AssumeQueueRoleForWorker, UpdateWorker, UpdateWorkerSchedule, BatchGetJobEntity,
DeleteWorker. Every method is a plain request/response call, which leaves callers free to
wrap each one in a durable step.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import boto3
import botocore.session
from botocore.config import Config
from botocore.credentials import RefreshableCredentials

logger = logging.getLogger(__name__)

# The session directory lives in /tmp, whose size Lambda configures separately from memory
# and does not report to the function.
SCRATCH_MIB = int(os.environ.get("EPHEMERAL_STORAGE_MIB", "512"))

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

    def assume_queue_role(self, *, queue_id: str) -> dict[str, Any]:
        """Obtain the queue role credentials a job's own scripts run with.

        This is how a real worker keeps a task's code off the worker's own identity. Not
        checkpointed: these are shorter-lived than a durable execution.
        """
        client = self._client(use_worker_credentials=True)
        try:
            response = client.assume_queue_role_for_worker(
                farmId=self.farm_id,
                fleetId=self.fleet_id,
                workerId=self.worker_id,
                queueId=queue_id,
            )
        except client.exceptions.ResourceNotFoundException as exc:
            raise WorkerDeletedError(f"Worker {self.worker_id} no longer exists") from exc
        return response["credentials"]

    def get_job_entities(self, *, identifiers: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch job, step, and environment entities in one call, keyed by their kind.

        One call rather than several because each is a round trip inside a billed
        invocation, and the identifiers a single action needs always fit the API's limit.
        """
        client = self._client(use_worker_credentials=True)
        limit = _max_identifiers(client)
        if not 1 <= len(identifiers) <= limit:
            raise WorkerProtocolError(
                f"BatchGetJobEntity accepts 1 to {limit} identifiers, got {len(identifiers)}"
            )

        entities, errors = self._batch_get_job_entity(client, identifiers)
        for kind, error in errors.items():
            if error["code"] != "MaxPayloadSizeExceeded":
                raise WorkerProtocolError(
                    f"Could not get {kind}: {error['code']}: {error['message']}"
                )
            # A step whose template carries large embedded files does not fit in a response
            # alongside anything else, so ask for that one entity on its own.
            alone, alone_errors = self._batch_get_job_entity(
                client, [i for i in identifiers if kind in i]
            )
            if alone_errors:
                raise WorkerProtocolError(
                    f"Could not get {kind} even on its own: "
                    f"{alone_errors[kind]['code']}: {alone_errors[kind]['message']}"
                )
            entities.update(alone)

        missing = {kind for identifier in identifiers for kind in identifier} - set(entities)
        if missing:
            raise WorkerProtocolError(
                f"{', '.join(sorted(missing))} was neither returned nor reported as an error"
            )
        return entities

    def _batch_get_job_entity(
        self, client: Any, identifiers: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = client.batch_get_job_entity(
            farmId=self.farm_id,
            fleetId=self.fleet_id,
            workerId=self.worker_id,
            identifiers=identifiers,
        )
        entities = {
            kind: entity
            for wrapper in response.get("entities", [])
            for kind, entity in wrapper.items()
        }
        errors = {
            kind: error
            for wrapper in response.get("errors", [])
            for kind, error in wrapper.items()
        }
        return entities, errors

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


def _max_identifiers(client: Any) -> int:
    """Read BatchGetJobEntity's identifier limit from the model rather than assuming it."""
    shape = client.meta.service_model.operation_model("BatchGetJobEntity").input_shape
    return int(shape.members["identifiers"].metadata["max"])


# Deadline Cloud tags each parameter with its type. `chunkInt` exists only on task
# parameters, so a job parameter carrying one is a wire-format error rather than a value.
JOB_PARAMETER_TYPES = {"string": "STRING", "int": "INT", "float": "FLOAT", "path": "PATH"}
TASK_PARAMETER_TYPES = {**JOB_PARAMETER_TYPES, "chunkInt": "CHUNK[INT]"}


def unwrap_parameters(
    tagged_values: dict[str, dict[str, str]], *, task: bool
) -> dict[str, dict[str, str]]:
    """Restate tagged parameters in Open Job Description's own spelling of the types.

    Plain JSON, so the result can cross a checkpoint boundary on its way to a session.
    """
    types = TASK_PARAMETER_TYPES if task else JOB_PARAMETER_TYPES
    parameters = {}
    for name, tagged_value in (tagged_values or {}).items():
        for tag, value_type in types.items():
            if tag in tagged_value:
                parameters[name] = {"type": value_type, "value": str(tagged_value[tag])}
                break
        else:
            raise WorkerProtocolError(
                f"Parameter {name} has no value this worker recognizes: "
                f"{sorted(tagged_value)}"
            )
    return parameters


def _as_iso(expiration: Any) -> str:
    """Accept an expiry as either a datetime or an ISO-8601 string."""
    return expiration if isinstance(expiration, str) else expiration.isoformat()


def default_capabilities() -> dict[str, Any]:
    """Capabilities describing a Lambda-hosted worker.

    Read from the runtime's own settings so a host requirement is judged against what the
    function was actually given. What matters most is the custom `attr.durable.lambda`
    attribute that job templates target.
    """
    memory_mib = int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "1769"))
    return {
        "amounts": [
            # Lambda gives a function one vCPU per 1769 MB, and nothing exposes the number.
            {"name": "amount.worker.vcpu", "value": max(1, memory_mib // 1769)},
            {"name": "amount.worker.memory", "value": memory_mib},
            {"name": "amount.worker.disk.scratch", "value": SCRATCH_MIB},
            {"name": "amount.worker.gpu", "value": 0},
        ],
        "attributes": [
            {"name": "attr.worker.os.family", "values": ["linux"]},
            {"name": "attr.worker.cpu.arch", "values": ["x86_64"]},
            {"name": "attr.durable.lambda", "values": ["true"]},
        ],
    }
