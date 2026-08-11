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

Every method is a plain request/response call, which keeps the caller free to wrap each
one in a durable step. See the module docstring in `durable_worker.py` for how that
interacts with checkpoint replay.

Credentials are the one piece of state held here. `AssumeFleetRoleForWorker` is wrapped
in botocore's `RefreshableCredentials`, so botocore tracks expiry and re-assumes the
role itself while signing. Callers therefore never assume the role, check an expiry, or
checkpoint a credential blob: the role is assumed on first use, and a worker that wakes
from a long suspension signs with credentials botocore has already renewed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import boto3
import botocore.session
from botocore.config import Config
from botocore.credentials import RefreshableCredentials

logger = logging.getLogger(__name__)

# The worker agent uses adaptive retries with several attempts, because on a host the
# time it spends backing off is free. In Lambda that sleep is billed compute, so the
# attempt count is kept low and long backoff is left to the durable step, which
# suspends the execution between attempts at no cost. A couple of quick in-process
# retries are still worth having: they absorb a transient blip without paying for a
# whole replay of the step.
DEADLINE_BOTOCORE_CONFIG = Config(retries={"max_attempts": 2, "mode": "standard"})


class WorkerProtocolError(Exception):
    """A worker protocol call failed in a way the caller cannot recover from."""


class WorkerNotUsableError(WorkerProtocolError):
    """This worker can no longer do work, whatever the reason.

    Raised for both a deleted worker and one the service has taken out of STARTED,
    which is what a ConflictException on UpdateWorkerSchedule means. Both have the same
    remedy, which is to stop cleanly and deregister rather than keep polling, so the
    caller does not benefit from telling them apart. Treating a conflict as an
    unhandled error instead would fail the execution before it deregistered, leaving a
    registry row that counts against fleet capacity forever.
    """


class WorkerDeletedError(WorkerNotUsableError):
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
        self._client_cache: dict[bool, Any] = {}
        self._worker_credentials: Optional[RefreshableCredentials] = None
        if credentials:
            self.set_credentials(credentials)

    # -- clients ---------------------------------------------------------------

    def _bootstrap_client(self):
        """A client using the Lambda execution role.

        That role holds only `deadline:CreateWorker` and
        `deadline:AssumeFleetRoleForWorker`, mirroring the least-privilege split the
        worker agent uses on EC2.
        """
        client = self._client_cache.get(False)
        if client is None:
            client = boto3.Session(region_name=self.region).client(
                "deadline", config=DEADLINE_BOTOCORE_CONFIG
            )
            self._client_cache[False] = client
        return client

    def _worker_client(self):
        """A client signing with the worker-scoped fleet role.

        The credentials are botocore `RefreshableCredentials`, so botocore tracks
        expiry and calls `AssumeFleetRoleForWorker` again itself when they are close to
        expiring. That matters here because a durable worker can stay suspended for
        longer than a credential's lifetime; the alternative is re-assuming the role
        before every call and reasoning about expiry by hand.

        Clients are cached because building one costs roughly 80ms, and with
        refreshable credentials a client stays usable for the worker's whole life, so
        there is nothing to invalidate.
        """
        if self._worker_credentials is None:
            if not self.worker_id:
                raise WorkerProtocolError(
                    "A worker ID is required before the fleet role can be assumed."
                )
            # Assume the role on first use rather than making every caller remember to
            # do it. Each durable step runs in its own invocation with a fresh instance,
            # so this happens once per step that talks to the service.
            self.assume_fleet_role()
        client = self._client_cache.get(True)
        if client is None:
            # Hand botocore the refreshable credentials directly; it invokes their
            # refresh callback as needed while signing.
            botocore_session = botocore.session.get_session()
            botocore_session._credentials = self._worker_credentials
            client = boto3.Session(
                botocore_session=botocore_session, region_name=self.region
            ).client("deadline", config=DEADLINE_BOTOCORE_CONFIG)
            self._client_cache[True] = client
        return client

    def _client(self, *, use_worker_credentials: bool):
        """Return the client for the requested identity."""
        return self._worker_client() if use_worker_credentials else self._bootstrap_client()

    def _fetch_fleet_role_credentials(self) -> dict[str, Any]:
        """Call AssumeFleetRoleForWorker, shaped for botocore.

        This is the refresh callback botocore invokes, so the keys are the ones
        `RefreshableCredentials` expects rather than the API's own spelling.
        """
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
        """Obtain worker-scoped credentials that refresh themselves.

        Called once after the worker ID is known. Afterwards botocore keeps the
        credentials current, so callers do not re-assume the role before each request
        and nothing needs to be checkpointed: credentials are far shorter-lived than a
        durable execution, so a replayed copy would usually be expired anyway.
        """
        self._worker_credentials = RefreshableCredentials.create_from_metadata(
            metadata=self._fetch_fleet_role_credentials(),
            refresh_using=self._fetch_fleet_role_credentials,
            method="deadline-assume-fleet-role-for-worker",
        )

    def set_credentials(self, credentials: dict[str, Any]) -> None:
        """Adopt credentials obtained elsewhere, keeping them refreshable.

        Accepts the API's own spelling, as returned by `AssumeFleetRoleForWorker`.
        Refresh still goes through this worker's own callback, so credentials adopted
        here stay current for as long as the worker runs.
        """
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
                # Timestamps are passed as ISO-8601 strings. botocore accepts those for
                # timestamp members and serializes them itself, so results can cross a
                # JSON checkpoint boundary without conversion here.
                updatedSessionActions=updated_session_actions or {},
            )
        except client.exceptions.ResourceNotFoundException as exc:
            raise WorkerDeletedError(f"Worker {self.worker_id} no longer exists") from exc
        except client.exceptions.ConflictException as exc:
            # The worker is no longer STARTED, usually because it stopped heartbeating
            # for long enough that the service took it out of service.
            raise WorkerNotUsableError(
                f"Worker {self.worker_id} is no longer in the STARTED status"
            ) from exc

    def get_environment_details(self, *, job_id: str, environment_id: str) -> dict[str, Any]:
        """Fetch the Open Job Description template for a queue environment.

        `BatchGetJobEntity` is how a worker retrieves the details behind the ids it is
        given. This worker asks only for environment details; job attachments, step
        templates, and job details are not needed because it does not run OpenJD
        scripts or stage files.

        Raises WorkerProtocolError if the service reports an error for the entity, so
        the caller fails the action rather than proceeding without the environment.
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
    """Return an expiry as an ISO-8601 string.

    botocore parses `expiration` into a datetime, but a caller restoring credentials
    from JSON will have a string. Accept either.
    """
    return expiration if isinstance(expiration, str) else expiration.isoformat()


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
