# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the scaling handler's reconciliation decisions.

Scaling is the part of this sample with no service-side safety net: Deadline Cloud
never starts or stops workers for a customer-managed fleet, so a mistake here either
strands the fleet at zero workers or asks for more workers than `maxWorkerCount`
allows. These tests pin down the decisions that are invisible until a fleet is under
load: how scale-out is capped, which workers scale-in picks, and what happens when the
fleet count cannot be read.

Run from the parent directory with:

    python3 -m unittest discover -s tests

Every AWS client is a mock, so the tests need no credentials, no network, and no
DynamoDB table.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
import unittest.mock as mock
from pathlib import Path

import boto3  # noqa: F401  (imported so the patches below can resolve `boto3.client`)
from botocore.exceptions import ClientError

LAMBDA_DIR = Path(__file__).resolve().parents[1] / "lambda"
sys.path.insert(0, str(LAMBDA_DIR))

# Required at import time by the scaling handler.
os.environ.setdefault(
    "WORKER_FUNCTION_ARN", "arn:aws:lambda:us-west-2:123456789012:function:durable-worker"
)
os.environ.setdefault("REGISTRY_TABLE", "test-registry")

# The handler builds its Lambda client and DynamoDB resource at module scope. Patching
# boto3 across the import keeps that from constructing anything that could reach AWS or
# demand a region, and every test replaces those module globals with its own mocks
# anyway.
with mock.patch("boto3.client"), mock.patch("boto3.resource"):
    import scaling_handler  # noqa: E402

FARM_ID = "farm-" + "0" * 32
FLEET_ID = "fleet-" + "0" * 32


def _event(new_fleet_size: int, old_fleet_size: int = 0) -> dict:
    """A `Fleet Size Recommendation Change` event as EventBridge delivers it."""
    return {
        "detail-type": "Fleet Size Recommendation Change",
        "source": "aws.deadline",
        "detail": {
            "farmId": FARM_ID,
            "fleetId": FLEET_ID,
            "oldFleetSize": old_fleet_size,
            "newFleetSize": new_fleet_size,
        },
    }


def _registry_row(worker_id: str, started_at: str, **extra) -> dict:
    return {"fleetId": FLEET_ID, "workerId": worker_id, "startedAt": started_at, **extra}


def _client_error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": f"simulated {code}"}}, operation)


class _ScalingHandlerTestCase(unittest.TestCase):
    """Shared harness: a mock registry table, Lambda client, and fleet worker count."""

    def _run_handler(
        self,
        *,
        new_fleet_size: int,
        registry: list[dict],
        fleet_worker_count=0,
        max_workers: int = 10,
    ):
        table = mock.MagicMock()
        table.query.return_value = {"Items": list(registry)}
        count_patch = (
            {"side_effect": fleet_worker_count}
            if isinstance(fleet_worker_count, Exception)
            else {"return_value": fleet_worker_count}
        )
        with mock.patch.object(scaling_handler, "dynamodb") as dynamodb, mock.patch.object(
            scaling_handler, "lambda_client"
        ) as lambda_client, mock.patch.object(
            scaling_handler, "MAX_WORKERS", max_workers
        ), mock.patch.object(
            scaling_handler, "_fleet_worker_count", **count_patch
        ):
            dynamodb.Table.return_value = table
            response = scaling_handler.lambda_handler(_event(new_fleet_size), None)
        return response, table, lambda_client


class TestScaleOut(_ScalingHandlerTestCase):
    """A recommendation above the current size has to become new executions."""

    def test_starts_one_execution_per_missing_worker(self):
        response, table, lambda_client = self._run_handler(
            new_fleet_size=3,
            registry=[_registry_row("worker-1", "2026-01-01T00:00:00+00:00")],
        )
        # One worker is one durable execution, so the shortfall is the invoke count.
        self.assertEqual(lambda_client.invoke.call_count, 2)
        self.assertEqual(json.loads(response["body"])["action"], "started 2 worker(s)")
        table.update_item.assert_not_called()

    def test_recommendation_above_max_workers_is_clamped(self):
        # The recommendation is Deadline Cloud's view of demand and ignores this
        # sample's own ceiling, so the handler has to apply MAX_WORKERS itself.
        _, _, lambda_client = self._run_handler(
            new_fleet_size=50, registry=[], max_workers=4
        )
        self.assertEqual(lambda_client.invoke.call_count, 4)

    def test_each_execution_gets_a_unique_name_and_an_async_invoke(self):
        _, _, lambda_client = self._run_handler(new_fleet_size=3, registry=[])

        names = set()
        for call in lambda_client.invoke.call_args_list:
            kwargs = call.kwargs
            # RequestResponse would cap the worker at the 15-minute synchronous invoke
            # limit; a durable execution needs the asynchronous path to outlive that.
            self.assertEqual(kwargs["InvocationType"], "Event")
            self.assertEqual(kwargs["FunctionName"], scaling_handler.WORKER_FUNCTION_ARN)
            # A reused execution name would make a redelivered EventBridge event a
            # no-op instead of a second worker, so names must not collide.
            names.add(kwargs["DurableExecutionName"])
            self.assertEqual(
                json.loads(kwargs["Payload"])["hostName"], kwargs["DurableExecutionName"]
            )
        self.assertEqual(len(names), 3)


class TestScaleOutHeadroom(_ScalingHandlerTestCase):
    """Scale-out is bounded by the fleet, not by the registry."""

    def test_fleet_worker_count_caps_the_number_started(self):
        # The registry is empty but the fleet still holds 8 workers, so only 2 of the
        # 5 requested workers can be started. Starting all 5 would fail at
        # CreateWorker with a ConflictException.
        _, _, lambda_client = self._run_handler(
            new_fleet_size=5, registry=[], fleet_worker_count=8, max_workers=10
        )
        self.assertEqual(lambda_client.invoke.call_count, 2)

    def test_full_fleet_starts_nothing(self):
        _, _, lambda_client = self._run_handler(
            new_fleet_size=5, registry=[], fleet_worker_count=10, max_workers=10
        )
        lambda_client.invoke.assert_not_called()

    def test_unreadable_fleet_count_does_not_drop_the_event(self):
        # Deliberately permissive: CreateWorker enforces maxWorkerCount on its own, so
        # a failed count must not stall the fleet at zero workers. The scaling event is
        # still honored up to MAX_WORKERS.
        with self.assertLogs(level="WARNING"):
            _, _, lambda_client = self._run_handler(
                new_fleet_size=3,
                registry=[],
                fleet_worker_count=_client_error("AccessDeniedException", "ListWorkers"),
                max_workers=10,
            )
        self.assertEqual(lambda_client.invoke.call_count, 3)

    def test_headroom_falls_back_to_max_workers_on_a_client_error(self):
        with mock.patch.object(
            scaling_handler,
            "_fleet_worker_count",
            side_effect=_client_error("ThrottlingException", "ListWorkers"),
        ), mock.patch.object(scaling_handler, "MAX_WORKERS", 7):
            with self.assertLogs(level="WARNING"):
                headroom = scaling_handler._headroom(farm_id=FARM_ID, fleet_id=FLEET_ID)
        self.assertEqual(headroom, 7)


class TestScaleIn(_ScalingHandlerTestCase):
    """A recommendation below the current size drains workers instead of killing them."""

    def test_marks_the_surplus_workers_to_drain(self):
        registry = [
            _registry_row("worker-1", "2026-01-01T00:00:00+00:00"),
            _registry_row("worker-2", "2026-01-01T00:10:00+00:00"),
            _registry_row("worker-3", "2026-01-01T00:20:00+00:00"),
        ]
        response, table, lambda_client = self._run_handler(
            new_fleet_size=1, registry=registry
        )
        self.assertEqual(table.update_item.call_count, 2)
        self.assertEqual(json.loads(response["body"])["action"], "marked 2 worker(s) to drain")
        # Draining is a flag, not a stop: nothing is invoked and no execution is killed,
        # so a worker holding a request in flight finishes it first.
        lambda_client.invoke.assert_not_called()

    def test_newest_workers_are_drained_first(self):
        registry = [
            _registry_row("worker-oldest", "2026-01-01T00:00:00+00:00"),
            _registry_row("worker-newest", "2026-01-01T00:20:00+00:00"),
            _registry_row("worker-middle", "2026-01-01T00:10:00+00:00"),
        ]
        _, table, _ = self._run_handler(new_fleet_size=1, registry=registry)

        drained = [call.kwargs["Key"]["workerId"] for call in table.update_item.call_args_list]
        # Newest first, because a recently started worker is least likely to be holding
        # a long-running Bedrock request, so draining it sheds capacity soonest.
        self.assertEqual(drained, ["worker-newest", "worker-middle"])

    def test_already_draining_workers_are_not_counted_or_re_flagged(self):
        registry = [
            _registry_row("worker-1", "2026-01-01T00:00:00+00:00"),
            _registry_row("worker-2", "2026-01-01T00:10:00+00:00", drain=True),
        ]
        response, table, _ = self._run_handler(new_fleet_size=1, registry=registry)
        # Only one worker is still active, so the fleet is already the right size.
        # Counting the draining worker would drain the last healthy one as well.
        self.assertEqual(json.loads(response["body"])["current"], 1)
        table.update_item.assert_not_called()

    def test_drain_flag_is_set_to_true(self):
        _, table, _ = self._run_handler(
            new_fleet_size=0,
            registry=[_registry_row("worker-1", "2026-01-01T00:00:00+00:00")],
        )
        kwargs = table.update_item.call_args.kwargs
        self.assertEqual(kwargs["Key"], {"fleetId": FLEET_ID, "workerId": "worker-1"})
        # The worker polls this exact attribute on every heartbeat.
        self.assertEqual(kwargs["ExpressionAttributeValues"], {":true": True})


class TestNoChange(_ScalingHandlerTestCase):
    def test_matching_size_touches_nothing(self):
        registry = [
            _registry_row("worker-1", "2026-01-01T00:00:00+00:00"),
            _registry_row("worker-2", "2026-01-01T00:10:00+00:00"),
        ]
        response, table, lambda_client = self._run_handler(
            new_fleet_size=2, registry=registry
        )
        # Deadline Cloud re-emits the recommendation periodically, not only when it
        # changes, so the steady state has to be free of side effects.
        lambda_client.invoke.assert_not_called()
        table.update_item.assert_not_called()
        self.assertEqual(json.loads(response["body"])["action"], "no change")


if __name__ == "__main__":
    unittest.main()
