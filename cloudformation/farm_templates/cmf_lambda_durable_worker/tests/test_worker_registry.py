# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the live-worker registry.

The registry is a bookkeeping aid rather than the source of truth for either half of the
sample, so every call has to fail in the direction that keeps a healthy worker working.

Run from the parent directory with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import unittest
import unittest.mock as mock
from datetime import datetime

from harness import FLEET_ID, WORKER_ID, client_error

import worker_registry


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
        # Scale-in ranks drain candidates by startedAt, so a row without it would sort as
        # the oldest worker and never be picked.
        self.assertEqual(item["startedAt"], "2026-01-01T00:00:00+00:00")
        self.assertIs(item["drain"], False)
        # An abandoned row would count against fleet capacity forever, so every row
        # carries a TTL as a backstop.
        self.assertIsInstance(item["expiresAt"], int)
        self.assertGreater(item["expiresAt"], 0)

    def test_a_write_failure_does_not_reach_the_caller(self):
        # This runs in the same step as CreateWorker and the STARTED transition, so
        # raising would fail a worker the service has already accepted.
        table = mock.MagicMock()
        table.put_item.side_effect = client_error("ThrottlingException", "PutItem")
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
        # trigger it and shrink the fleet for a reason unrelated to demand.
        table = mock.MagicMock()
        table.get_item.side_effect = client_error("ThrottlingException", "GetItem")
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
        # The worker looks once per heartbeat, so a stale answer delays scale-in by a
        # whole poll.
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
        # The worker has already stopped with Deadline Cloud by this point.
        table = mock.MagicMock()
        table.delete_item.side_effect = client_error("ThrottlingException", "DeleteItem")
        with mock.patch.object(worker_registry, "_table", return_value=table):
            with self.assertLogs(worker_registry.logger, level="ERROR"):
                worker_registry.deregister(fleet_id=FLEET_ID, worker_id=WORKER_ID)


class TestUtcNowIso(unittest.TestCase):
    def test_returns_a_timezone_aware_iso_string(self):
        # Timestamps travel as strings through checkpoints and are parsed back with
        # fromisoformat, so the value has to round-trip and carry an offset.
        parsed = datetime.fromisoformat(worker_registry.utc_now_iso())
        self.assertIsNotNone(parsed.tzinfo)


if __name__ == "__main__":
    unittest.main()
