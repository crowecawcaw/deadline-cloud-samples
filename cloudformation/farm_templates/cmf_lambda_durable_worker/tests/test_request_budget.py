# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the bound this worker puts on the worker agent's retry loops.

Those loops are `while True` with no attempt cap. Unbounded, one throttled request can spend
a whole billed invocation and then have the durable step replay it, so the budget below is
the difference between a slow service and a worker that makes no progress at all.

The timer is the other half: a thread still alive when the handler returns is frozen with the
invocation and thaws inside a later one, whose step results it knows nothing about.

Run from the parent directory with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import threading
import time
import unittest
import unittest.mock as mock

import harness
from harness import FARM_ID, FLEET_ID, WORKER_ID, client_error

import worker_protocol


def _interruptible(*, interrupt_event=None, **kwargs):
    """Stands in for a wrapper that accepts an interrupt_event."""
    return interrupt_event


def _plain(**kwargs):
    """Stands in for one of the four wrappers that does not."""
    return "done"


class TestRequestBudget(unittest.TestCase):
    def test_a_wrapper_that_takes_an_interrupt_event_is_given_one(self):
        event = worker_protocol.request(_interruptible)
        self.assertIsInstance(event, threading.Event)

    def test_a_wrapper_that_takes_no_interrupt_event_is_not_given_one(self):
        self.assertEqual(worker_protocol.request(_plain), "done")

    def test_no_thread_outlives_the_request(self):
        before = set(threading.enumerate())
        worker_protocol.request(_plain)
        self.assertEqual(set(threading.enumerate()), before)

    def test_the_budget_stops_a_loop_that_keeps_sleeping(self):
        def never_succeeds(**kwargs):
            while True:
                worker_protocol.protocol.sleep(0.01)

        with mock.patch.object(worker_protocol, "REQUEST_RETRY_BUDGET_SECONDS", 0.05):
            with self.assertRaises(worker_protocol.DeadlineRequestInterrupted):
                worker_protocol.request(never_succeeds)

    def test_the_agents_own_sleep_is_put_back_afterwards(self):
        original = worker_protocol.protocol.sleep
        worker_protocol.request(_plain)
        self.assertIs(worker_protocol.protocol.sleep, original)

    def test_the_sleep_is_put_back_even_when_the_call_raises(self):
        original = worker_protocol.protocol.sleep

        def explodes(**kwargs):
            raise ZeroDivisionError("boom")

        with self.assertRaises(ZeroDivisionError):
            worker_protocol.request(explodes)
        self.assertIs(worker_protocol.protocol.sleep, original)
        self.assertEqual(threading.active_count(), 1)

    def test_a_nested_request_shares_the_outer_budget(self):
        # A credential refresh happens inside another request, and two independent budgets
        # would let one step wait for twice as long as configured.
        def outer(*, interrupt_event=None, **kwargs):
            return interrupt_event, worker_protocol.request(_interruptible)

        outer_event, inner_event = worker_protocol.request(outer)
        self.assertIs(outer_event, inner_event)

    def test_the_budget_leaves_room_inside_the_function_timeout(self):
        # The stack gives the function a 300s Timeout, and one invocation has to fit a whole
        # session action as well as its API calls.
        self.assertLess(worker_protocol.REQUEST_RETRY_BUDGET_SECONDS, 60)


class AlwaysThrottled:
    """A deadline client that throttles every call, as a sustained throttle would."""

    def __init__(self) -> None:
        self.attempts = 0
        self._real_client = harness.deadline_client()

    def __getattr__(self, operation):
        def throttle(**kwargs):
            self.attempts += 1
            raise client_error("ThrottlingException", operation)

        return throttle


class TestASustainedThrottle(unittest.TestCase):
    """The case the budget exists for: retrying can never succeed, so it must stop."""

    def _under_budget(self, call, **kwargs):
        client = AlwaysThrottled()
        started = time.monotonic()
        # Long enough for the agent's first backoff, which is under a second, so the call is
        # bounded rather than merely never retried.
        with mock.patch.object(worker_protocol, "REQUEST_RETRY_BUDGET_SECONDS", 1.5):
            with self.assertRaises(worker_protocol.DeadlineRequestInterrupted):
                worker_protocol.request(call, deadline_client=client, **kwargs)
        elapsed = time.monotonic() - started
        self.assertGreater(client.attempts, 1, "gave up before retrying at all")
        # Generous, but far short of the 300s a `while True` would have spent.
        self.assertLess(elapsed, 10)
        return client

    def test_a_wrapper_with_an_interrupt_event_gives_up(self):
        self._under_budget(
            worker_protocol.protocol.update_worker_schedule,
            farm_id=FARM_ID,
            fleet_id=FLEET_ID,
            worker_id=WORKER_ID,
            updated_session_actions={},
        )

    def test_a_wrapper_without_one_gives_up_too(self):
        # BatchGetJobEntity takes no interrupt_event, so only the sleep the budget replaces
        # can stop it. Four of the seven wrappers are in this shape.
        self._under_budget(
            worker_protocol.protocol.batch_get_job_entity,
            farm_id=FARM_ID,
            fleet_id=FLEET_ID,
            worker_id=WORKER_ID,
            identifiers=worker_protocol.action_identifiers(job_id="job-1"),
        )


class TestTelemetry(unittest.TestCase):
    def test_the_worker_agents_telemetry_is_opted_out_of(self):
        # A public sample must not emit telemetry nobody asked for. Importing worker_protocol
        # is what sets this, before anything can construct the agent's TelemetryClient.
        import os

        self.assertEqual(os.environ["DEADLINE_CLOUD_TELEMETRY_OPT_OUT"], "true")

    def test_importing_the_agent_starts_no_telemetry_client(self):
        self.assertIsNone(worker_protocol.protocol._telemetry_client)


if __name__ == "__main__":
    unittest.main()
