# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the live-worker registry.

The registry sits between the two halves of the sample: the scaling handler counts rows
to decide how many workers exist, and each worker polls its own row to learn whether it
has been asked to drain. That makes its failure behavior more interesting than its
success behavior, because the registry is a bookkeeping aid rather than the source of
truth for either side. Every call therefore has to fail in the direction that keeps a
healthy worker working, and these tests pin those directions down.

Run from the parent directory with:

    python3 -m unittest discover -s tests

The DynamoDB table is a mock, so the tests need no credentials and no network.
"""

from __future__ import annotations

import os
import sys
import unittest
import unittest.mock as mock
from pathlib import Path

from botocore.exceptions import ClientError

LAMBDA_DIR = Path(__file__).resolve().parents[1] / "lambda"
sys.path.insert(0, str(LAMBDA_DIR))

os.environ.setdefault("REGISTRY_TABLE", "test-registry")

import worker_registry  # noqa: E402

FLEET_ID = "fleet-" + "0" * 32
WORKER_ID = "worker-" + "0" * 32


def _client_error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": f"simulated {code}"}}, operation)


class TestRegister(unittest.TestCase):
    def test_writes_the_fields_both_halves_of_the_sample_read(self):
        table = mock.MagicMock()
        with mock.patch.object(worker_registry, "_table", return_value=table):
            worker_registry.register(
                fleet_id=FLEET_ID,
                worker_id=WORKER_ID,
                started_at="2026-01-01T00:00:00+00:00",
            )
        item = table.put_item.call_args.kwargs["Item"]
        self.assertEqual(item["fleetId"], FLEET_ID)
        self.assertEqual(item["workerId"], WORKER_ID)
        # startedAt is how scale-in ranks drain candidates, so a row without it would
        # sort as the oldest worker and never be picked.
        self.assertEqual(item["startedAt"], "2026-01-01T00:00:00+00:00")
        # drain is written explicitly rather than left absent, so the attribute exists
        # from the moment the worker is visible to a scaling event.
        self.assertIs(item["drain"], False)
        # An abandoned row counts against fleet capacity forever and would quietly stop
        # the fleet from scaling out, so every row carries a DynamoDB TTL as a backstop.
        self.assertIn("expiresAt", item)
        self.assertIsInstance(item["expiresAt"], int)
        self.assertGreater(item["expiresAt"], 0)

    def test_a_write_failure_does_not_reach_the_caller(self):
        # register() runs inside the same durable step as CreateWorker and the STARTED
        # transition. Raising here would fail a worker that the service has already
        # accepted, so the registry write is best-effort.
        table = mock.MagicMock()
        table.put_item.side_effect = _client_error("ProvisionedThroughputExceededException", "PutItem")
        with mock.patch.object(worker_registry, "_table", return_value=table):
            with self.assertLogs(worker_registry.logger, level="ERROR"):
                worker_registry.register(
                    fleet_id=FLEET_ID,
                    worker_id=WORKER_ID,
                    started_at="2026-01-01T00:00:00+00:00",
                )


class TestShouldDrain(unittest.TestCase):
    def test_true_when_the_scaling_handler_has_set_the_flag(self):
        table = mock.MagicMock()
        table.get_item.return_value = {"Item": {"drain": True}}
        with mock.patch.object(worker_registry, "_table", return_value=table):
            self.assertTrue(
                worker_registry.should_drain(fleet_id=FLEET_ID, worker_id=WORKER_ID)
            )

    def test_false_when_the_flag_is_absent_or_unset(self):
        for item in ({"drain": False}, {}, None):
            with self.subTest(item=item):
                table = mock.MagicMock()
                table.get_item.return_value = {} if item is None else {"Item": item}
                with mock.patch.object(worker_registry, "_table", return_value=table):
                    self.assertFalse(
                        worker_registry.should_drain(fleet_id=FLEET_ID, worker_id=WORKER_ID)
                    )

    def test_a_read_failure_keeps_the_worker_working(self):
        # Draining is irreversible for this worker, so a transient read error must not
        # trigger it: that would shrink the fleet for a reason unrelated to demand.
        table = mock.MagicMock()
        table.get_item.side_effect = _client_error("ThrottlingException", "GetItem")
        with mock.patch.object(worker_registry, "_table", return_value=table):
            with self.assertLogs(worker_registry.logger, level="ERROR"):
                self.assertFalse(
                    worker_registry.should_drain(fleet_id=FLEET_ID, worker_id=WORKER_ID)
                )

    def test_the_read_is_strongly_consistent(self):
        table = mock.MagicMock()
        table.get_item.return_value = {"Item": {"drain": False}}
        with mock.patch.object(worker_registry, "_table", return_value=table):
            worker_registry.should_drain(fleet_id=FLEET_ID, worker_id=WORKER_ID)
        kwargs = table.get_item.call_args.kwargs
        self.assertEqual(kwargs["Key"], {"fleetId": FLEET_ID, "workerId": WORKER_ID})
        # An eventually consistent read can miss a flag the scaling handler just wrote,
        # and the worker only looks once per heartbeat interval, so a stale answer
        # delays scale-in by a whole poll.
        self.assertIs(kwargs["ConsistentRead"], True)


class TestDeregister(unittest.TestCase):
    def test_deletes_the_row_by_its_full_key(self):
        table = mock.MagicMock()
        with mock.patch.object(worker_registry, "_table", return_value=table):
            worker_registry.deregister(fleet_id=FLEET_ID, worker_id=WORKER_ID)
        table.delete_item.assert_called_once_with(
            Key={"fleetId": FLEET_ID, "workerId": WORKER_ID}
        )

    def test_a_delete_failure_does_not_reach_the_caller(self):
        # The worker has already stopped with Deadline Cloud by this point. Raising
        # would fail the durable execution over a row that only inflates the handler's
        # worker count until the next successful pass.
        table = mock.MagicMock()
        table.delete_item.side_effect = _client_error("ThrottlingException", "DeleteItem")
        with mock.patch.object(worker_registry, "_table", return_value=table):
            with self.assertLogs(worker_registry.logger, level="ERROR"):
                worker_registry.deregister(fleet_id=FLEET_ID, worker_id=WORKER_ID)


class TestUtcNowIso(unittest.TestCase):
    def test_returns_a_timezone_aware_iso_string(self):
        from datetime import datetime

        value = worker_registry.utc_now_iso()
        # UpdateWorkerSchedule timestamps travel as strings through checkpoints and are
        # parsed back with fromisoformat, so the value has to round-trip and carry an
        # offset rather than being a naive local time.
        parsed = datetime.fromisoformat(value)
        self.assertIsNotNone(parsed.tzinfo)


if __name__ == "__main__":
    unittest.main()
