# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""This worker's Deadline Cloud identity, and the protocol calls it makes on its own behalf.

The protocol itself comes from `deadline-cloud-worker-agent`: its `api_models` request and
response shapes, its `aws.deadline` call wrappers with their error taxonomy, and its
`JobEntities` fetching, batching, and caching. What is left in this module is only what the
agent has no notion of, which is a worker whose every call has to fit inside one Lambda
invocation.

Deliberately unused from that package: its scheduler, its entrypoint, its `Worker`, and
`log_sync.cloudwatch`. Importing any of its modules loads all of them anyway, because
`deadline_worker_agent/__init__.py` imports them itself.
"""

from __future__ import annotations

import copy
import functools
import inspect
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import boto3
import botocore.session
from botocore.config import Config
from botocore.credentials import RefreshableCredentials

# The agent builds a TelemetryClient the first time one of its record_*_telemetry_event
# functions is called, and that client reads this variable once, when it is constructed.
# Nothing below reaches one, but a public sample must not be one refactor away from emitting
# telemetry nobody asked for. setdefault leaves a deployer free to opt back in.
os.environ.setdefault("DEADLINE_CLOUD_TELEMETRY_OPT_OUT", "true")

from deadline_worker_agent.api_models import (  # noqa: E402  (after the opt-out above)
    AwsCredentials,
    EnvironmentDetailsIdentifier,
    EnvironmentDetailsIdentifierFields,
    HostProperties,
    JobDetailsIdentifier,
    JobDetailsIdentifierFields,
    StepDetailsIdentifier,
    StepDetailsIdentifierFields,
    UpdatedSessionActionInfo,
    UpdateWorkerScheduleResponse,
    WorkerStatus,
)
from deadline_worker_agent.aws import deadline as protocol  # noqa: E402
from deadline_worker_agent.aws.deadline import (  # noqa: E402
    DeadlineRequestConditionallyRecoverableError,
    DeadlineRequestError,
    DeadlineRequestInterrupted,
    DeadlineRequestRecoverableError,
    DeadlineRequestUnrecoverableError,
    DeadlineRequestWorkerNotFound,
    DeadlineRequestWorkerOfflineError,
)
from deadline_worker_agent.boto import DeadlineClient  # noqa: E402
from deadline_worker_agent.capabilities import Capabilities  # noqa: E402
from deadline_worker_agent.sessions.job_entities import EnvironmentDetails, JobEntities  # noqa: E402
from deadline_worker_agent.sessions.job_entities.job_details import (  # noqa: E402
    parameters_from_api_response,
)

logger = logging.getLogger(__name__)

# A deleted worker and one the service has taken out of STARTED have the same remedy here,
# which is to stop cleanly and deregister rather than keep polling.
WORKER_UNUSABLE = (DeadlineRequestWorkerNotFound, DeadlineRequestWorkerOfflineError)

# UpdateWorker reports a missing worker as conditionally recoverable rather than as
# DeadlineRequestWorkerNotFound, so a drain has to read that class as "already gone" too.
WORKER_UNDRAINABLE = (
    DeadlineRequestWorkerNotFound,
    DeadlineRequestConditionallyRecoverableError,
)

# The session directory lives in /tmp, whose size Lambda configures separately from memory
# and does not report to the function.
SCRATCH_MIB = int(os.environ.get("EPHEMERAL_STORAGE_MIB", "512"))

# Bound on the agent's own retry loops, which are `while True` with no attempt cap. In
# Lambda an unbounded retry consumes the whole invocation timeout and then replays the
# step, so every call below runs under this budget instead.
REQUEST_RETRY_BUDGET_SECONDS = float(os.environ.get("REQUEST_RETRY_BUDGET_SECONDS", "30"))

# One attempt on purpose: the agent's wrappers classify and retry Deadline Cloud errors
# themselves, and a second botocore attempt would only add a sleep inside the call, which
# in Lambda is billed compute. The timeouts matter as much: the budget above cannot
# interrupt a request already in flight, so botocore's own default 60s is what would
# actually bound one.
DEADLINE_BOTOCORE_CONFIG = Config(
    retries={"max_attempts": 1, "mode": "standard"}, connect_timeout=5, read_timeout=15
)


# -- bounded requests -------------------------------------------------------------


_budget: Optional[threading.Event] = None


@contextmanager
def _retry_budget() -> Iterator[threading.Event]:
    """Give the agent's retry loops a deadline, and leave no thread behind.

    A nested call shares the outermost budget: a credential refresh happens inside another
    request, and two independent budgets would allow twice the intended delay.
    """
    global _budget
    if _budget is not None:
        yield _budget
        return

    event = threading.Event()
    timer = threading.Timer(REQUEST_RETRY_BUDGET_SECONDS, event.set)
    timer.daemon = True
    timer.start()
    # Four of the seven wrappers accept no interrupt_event, so the `sleep` they hold as a
    # module global is the only place their loop can be stopped.
    unbounded_sleep = protocol.sleep
    protocol.sleep = _budgeted_sleep(event)
    _budget = event
    try:
        yield event
    finally:
        _budget = None
        protocol.sleep = unbounded_sleep
        # A live timer thread is frozen with this invocation and thaws inside a later one,
        # so it has to be gone before the handler returns or suspends.
        timer.cancel()
        timer.join()


def _budgeted_sleep(event: threading.Event) -> Callable[[float], None]:
    """A `sleep` that gives up rather than waiting past the budget."""

    def sleep(delay: float) -> None:
        if event.wait(delay):
            raise DeadlineRequestInterrupted(
                f"Gave up retrying after {REQUEST_RETRY_BUDGET_SECONDS}s"
            )

    return sleep


@functools.lru_cache(maxsize=None)
def _accepts_interrupt(call: Callable[..., Any]) -> bool:
    return "interrupt_event" in inspect.signature(call).parameters


def request(call: Callable[..., Any], **kwargs: Any) -> Any:
    """Make one protocol call, retrying no longer than the request budget allows.

    Raises:
        DeadlineRequestInterrupted: the budget ran out while the agent was still retrying.
    """
    with _retry_budget() as event:
        if _accepts_interrupt(call):
            kwargs["interrupt_event"] = event
        return call(**kwargs)


@dataclass(frozen=True)
class _FleetLocation:
    """The two attributes the agent's CreateWorker and DeleteWorker read off its config.

    A real `Configuration` is built from worker.toml and the agent's own command line,
    neither of which exists in a function.
    """

    farm_id: str
    fleet_id: str


class DeadlineWorker:
    """A Deadline Cloud worker identity plus the calls it makes on its own behalf."""

    def __init__(
        self,
        *,
        farm_id: str,
        fleet_id: str,
        region: str,
        worker_id: Optional[str] = None,
        credentials: Optional[AwsCredentials] = None,
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
                raise DeadlineRequestUnrecoverableError(
                    ValueError("A worker ID is required before the fleet role can be assumed.")
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

    def _fetch_fleet_role_credentials(self) -> dict[str, Any]:
        """Call AssumeFleetRoleForWorker, keyed the way RefreshableCredentials expects."""
        response = request(
            protocol.assume_fleet_role_for_worker,
            deadline_client=self._bootstrap_client(),
            farm_id=self.farm_id,
            fleet_id=self.fleet_id,
            worker_id=self.worker_id,
        )
        credentials = response["credentials"]
        logger.info("Obtained fleet role credentials for worker %s", self.worker_id)
        return {
            "access_key": credentials["accessKeyId"],
            "secret_key": credentials["secretAccessKey"],
            "token": credentials["sessionToken"],
            "expiry_time": _as_iso(credentials["expiration"]),
        }

    # -- lifecycle -------------------------------------------------------------

    def create_worker(self, *, host_name: str) -> str:
        """Register a new worker with the fleet and return its worker ID."""
        response = request(
            protocol.create_worker,
            deadline_client=self._bootstrap_client(),
            config=_FleetLocation(self.farm_id, self.fleet_id),
            host_properties=HostProperties(hostName=host_name),
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

    def set_credentials(self, credentials: AwsCredentials) -> None:
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
        self, *, status: WorkerStatus, capabilities: Optional[Capabilities] = None
    ) -> dict[str, Any]:
        """Move the worker to STARTED, STOPPING, or STOPPED.

        `capabilities` is required on the transition to STARTED.
        """
        return request(
            protocol.update_worker,
            deadline_client=self._worker_client(),
            farm_id=self.farm_id,
            fleet_id=self.fleet_id,
            worker_id=self.worker_id,
            status=status,
            capabilities=capabilities,
        )

    def update_worker_schedule(
        self, *, updated_session_actions: Optional[dict[str, UpdatedSessionActionInfo]] = None
    ) -> UpdateWorkerScheduleResponse:
        """Heartbeat, report action progress, and receive assigned work.

        The response carries `assignedSessions`, `cancelSessionActions`, an optional
        `desiredWorkerStatus`, and `updateIntervalSeconds`.
        """
        return request(
            protocol.update_worker_schedule,
            deadline_client=self._worker_client(),
            farm_id=self.farm_id,
            fleet_id=self.fleet_id,
            worker_id=self.worker_id,
            # botocore serializes ISO-8601 strings for timestamp members, so results can
            # cross a JSON checkpoint boundary without conversion.
            updated_session_actions=updated_session_actions or {},
        )

    def assume_queue_role(self, *, queue_id: str) -> AwsCredentials:
        """Obtain the queue role credentials a job's own scripts run with.

        This is how a real worker keeps a task's code off the worker's own identity. Not
        checkpointed: these are shorter-lived than a durable execution.
        """
        response = request(
            protocol.assume_queue_role_for_worker,
            deadline_client=self._worker_client(),
            farm_id=self.farm_id,
            fleet_id=self.fleet_id,
            worker_id=self.worker_id,
            queue_id=queue_id,
        )
        return response["credentials"]

    def job_entities(self, *, job_id: str) -> JobEntities:
        """The agent's entity fetcher, which batches, caches, and validates for us.

        `DeadlineClient` earns its place here alone: the batch size comes from the botocore
        model through a private attribute only that wrapper exposes. Its UpdateWorkerSchedule
        reshaping is not wanted, so no other call goes through it.
        """
        client = self._worker_client()
        if not hasattr(client, "batch_get_job_entity"):
            # Rather than let DeadlineClient fabricate the hard-coded response it returns
            # for an API that is missing from the model.
            raise DeadlineRequestUnrecoverableError(
                ValueError("The installed botocore has no BatchGetJobEntity operation.")
            )
        return JobEntities(
            farm_id=self.farm_id,
            fleet_id=self.fleet_id,
            worker_id=self.worker_id,
            job_id=job_id,
            deadline_client=DeadlineClient(client),
            windows_credentials_resolver=None,
            job_run_as_user_override=None,
        )

    def delete_worker(self) -> None:
        """Deregister the worker. Safe to call when the worker is already gone."""
        try:
            request(
                protocol.delete_worker,
                deadline_client=self._worker_client(),
                config=_FleetLocation(self.farm_id, self.fleet_id),
                worker_id=self.worker_id,
            )
            logger.info("Deleted worker %s", self.worker_id)
        except DeadlineRequestRecoverableError as exc:
            # The service still has the worker in a running status. Retrying would only
            # delay a deregistration the registry has already recorded.
            logger.warning("Worker %s could not be deleted yet: %s", self.worker_id, exc)
        except DeadlineRequestUnrecoverableError as exc:
            # DeleteWorker maps a missing worker to the generic unrecoverable error rather
            # than to DeadlineRequestWorkerNotFound, so the code has to be read back out.
            if error_code(exc) != "ResourceNotFoundException":
                raise
            logger.info("Worker %s was already deleted", self.worker_id)


def error_code(exc: DeadlineRequestError) -> Optional[str]:
    """The service error code the agent wrapped, when there is one."""
    response = getattr(exc.inner_exc, "response", None) or {}
    return response.get("Error", {}).get("Code")


# -- job entities ----------------------------------------------------------------


def action_identifiers(
    *, job_id: str, step_id: Optional[str] = None, environment_id: Optional[str] = None
) -> list[Any]:
    """Everything one session action needs, so a single warmed call fetches all of it."""
    identifiers: list[Any] = [
        JobDetailsIdentifier(jobDetails=JobDetailsIdentifierFields(jobId=job_id))
    ]
    if step_id is not None:
        identifiers.append(
            StepDetailsIdentifier(
                stepDetails=StepDetailsIdentifierFields(jobId=job_id, stepId=step_id)
            )
        )
    if environment_id is not None:
        identifiers.append(
            EnvironmentDetailsIdentifier(
                environmentDetails=EnvironmentDetailsIdentifierFields(
                    jobId=job_id, environmentId=environment_id
                )
            )
        )
    return identifiers


def environment_template(
    entities: JobEntities, *, job_id: str, environment_id: str, exiting: bool
) -> Any:
    """Fetch and parse one environment, neutering its `onEnter` when only the exit is wanted.

    Parsed here rather than through `JobEntities.environment_details` so that both the
    `onEnter` replacement and the nested authored shape are handled before validation.
    """
    data = copy.deepcopy(
        entities.request(
            identifier=EnvironmentDetailsIdentifier(
                environmentDetails=EnvironmentDetailsIdentifierFields(
                    jobId=job_id, environmentId=environment_id
                )
            )
        )
    )
    # BatchGetJobEntity returns the definition unwrapped from the `environment` key an
    # authored template nests it under. Both shapes are accepted because the nested form is
    # what a reader sees in `queue_environments/`.
    data["template"] = data["template"].get("environment") or data["template"]
    if exiting:
        data["template"] = _with_noop_on_enter(data["template"])
    details = EnvironmentDetails.from_boto(EnvironmentDetails.validate_entity_data(data))
    return details.environment


# Stands in for an `onEnter` that already ran in an earlier invocation. /bin/sh rather than
# /bin/true because openjd-sessions already execs a #!/bin/sh wrapper, so this adds no new
# assumption about what the runtime image contains.
NOOP_ACTION = {"command": "/bin/sh", "args": ["-c", "exit 0"]}


def _with_noop_on_enter(definition: dict[str, Any]) -> dict[str, Any]:
    """Return the environment with its `onEnter` replaced by a command that does nothing.

    Replaced rather than removed because `onEnter` is required in newer revisions of the
    model. The real `onEnter` already ran in an earlier invocation, and its variables come
    back through `os_env_vars`, so running it again would only repeat its side effects.
    """
    script = definition.get("script")
    if not script or not script.get("actions"):
        return definition
    actions = {**script["actions"], "onEnter": dict(NOOP_ACTION)}
    return {**definition, "script": {**script, "actions": actions}}


def unwrap_parameters(tagged_values: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Restate tagged parameters in Open Job Description's own spelling of the types."""
    return parameters_from_api_response(tagged_values or {})


def _as_iso(expiration: Any) -> str:
    """Accept an expiry as either a datetime or an ISO-8601 string."""
    return expiration if isinstance(expiration, str) else expiration.isoformat()


def default_capabilities() -> Capabilities:
    """Capabilities describing a Lambda-hosted worker.

    Read from the runtime's own settings so a host requirement is judged against what the
    function was actually given. What matters most is the custom `attr.durable.lambda`
    attribute that job templates target.
    """
    memory_mib = int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "1769"))
    return Capabilities(
        amounts={
            # Lambda gives a function one vCPU per 1769 MB, and nothing exposes the number.
            "amount.worker.vcpu": max(1, memory_mib // 1769),
            "amount.worker.memory": memory_mib,
            "amount.worker.disk.scratch": SCRATCH_MIB,
            "amount.worker.gpu": 0,
        },
        attributes={
            "attr.worker.os.family": ["linux"],
            "attr.worker.cpu.arch": ["x86_64"],
            "attr.durable.lambda": ["true"],
        },
    )
