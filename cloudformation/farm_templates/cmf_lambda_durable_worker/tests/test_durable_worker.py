# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the worker's wire-format handling and credentials.

Run from the parent directory with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import unittest
import unittest.mock as mock
from datetime import datetime, timedelta, timezone

import harness  # noqa: F401  (stubs the durable execution SDK, puts lambda/ on sys.path)

import durable_worker
import worker_protocol


class TestParameterUnwrapping(unittest.TestCase):
    """Deadline Cloud tags each task parameter with its type."""

    def test_unwraps_each_supported_type(self):
        self.assertEqual(durable_worker._unwrap_parameter({"string": "hello"}), "hello")
        self.assertEqual(durable_worker._unwrap_parameter({"int": "42"}), "42")
        self.assertEqual(durable_worker._unwrap_parameter({"float": "1.5"}), "1.5")
        self.assertEqual(durable_worker._unwrap_parameter({"path": "/tmp/x"}), "/tmp/x")

    def test_unknown_tag_yields_none(self):
        # One unfamiliar parameter must not fail the whole assignment.
        self.assertIsNone(durable_worker._unwrap_parameter({"unexpected": "value"}))


class TestSessionSummarization(unittest.TestCase):
    """The worker shrinks the schedule response before checkpointing it."""

    def test_task_run_is_flattened_to_plain_values(self):
        assigned = {
            "session-1": {
                "queueId": "queue-abc",
                "jobId": "job-abc",
                "logConfiguration": {"logDriver": "awslogs", "options": {"noise": "x"}},
                "sessionActions": [
                    {
                        "sessionActionId": "action-1",
                        "definition": {
                            "taskRun": {
                                "taskId": "task-1",
                                "stepId": "step-1",
                                "parameters": {
                                    "Provider": {"string": "bedrock-async"},
                                    "Request": {"string": '{"modelId": "m"}'},
                                },
                            }
                        },
                    }
                ],
            }
        }

        summary = durable_worker._summarize_sessions(assigned)
        parameters = summary["session-1"]["sessionActions"][0]["definition"]["taskRun"][
            "parameters"
        ]

        self.assertEqual(summary["session-1"]["jobId"], "job-abc")
        self.assertEqual(parameters["Provider"], "bedrock-async")
        # The request body is carried through as an opaque string.
        self.assertEqual(parameters["Request"], '{"modelId": "m"}')
        # Checkpoint payloads are size-limited and this worker streams no session logs.
        self.assertNotIn("logConfiguration", summary["session-1"])

    def test_environment_actions_are_preserved(self):
        assigned = {
            "session-1": {
                "queueId": "queue-abc",
                "jobId": "job-abc",
                "sessionActions": [
                    {
                        "sessionActionId": "action-1",
                        "definition": {"envEnter": {"environmentId": "env-1"}},
                    }
                ],
            }
        }
        summary = durable_worker._summarize_sessions(assigned)
        definition = summary["session-1"]["sessionActions"][0]["definition"]
        self.assertEqual(definition, {"envEnter": {"environmentId": "env-1"}})

    def test_missing_session_actions_is_tolerated(self):
        summary = durable_worker._summarize_sessions(
            {"session-1": {"queueId": "q", "jobId": "j"}}
        )
        self.assertEqual(summary["session-1"]["sessionActions"], [])


class TestCapabilities(unittest.TestCase):
    def test_declares_the_targeting_attribute(self):
        capabilities = worker_protocol.default_capabilities()
        attributes = {a["name"]: a["values"] for a in capabilities["attributes"]}
        # Job templates target this attribute, and the fleet must declare it too.
        self.assertEqual(attributes["attr.durable.lambda"], ["true"])

    def test_standard_capability_names_are_used(self):
        capabilities = worker_protocol.default_capabilities()
        amounts = {a["name"] for a in capabilities["amounts"]}
        attributes = {a["name"] for a in capabilities["attributes"]}
        # Open Job Description reserves the `amount.worker` and `attr.worker` prefixes.
        self.assertIn("amount.worker.vcpu", amounts)
        self.assertIn("amount.worker.memory", amounts)
        self.assertIn("attr.worker.os.family", attributes)
        self.assertIn("attr.worker.cpu.arch", attributes)


class TestTimestampSerialization(unittest.TestCase):
    def test_botocore_accepts_iso_strings_for_timestamp_members(self):
        import boto3
        from botocore.stub import Stubber

        client = boto3.client(
            "deadline",
            region_name="us-west-2",
            aws_access_key_id="a",
            aws_secret_access_key="b",
        )
        stubber = Stubber(client)
        stubber.add_response(
            "update_worker_schedule",
            {"assignedSessions": {}, "cancelSessionActions": {}, "updateIntervalSeconds": 15},
        )
        stubber.activate()
        # Checkpoints hold JSON, so timestamps arrive as strings. This pins the reason no
        # conversion is needed: botocore serializes ISO-8601 strings for timestamp members
        # itself. If that ever changes it fails here rather than against a live worker.
        client.update_worker_schedule(
            farmId=harness.FARM_ID,
            fleetId=harness.FLEET_ID,
            workerId=harness.WORKER_ID,
            updatedSessionActions={
                "action-1": {
                    "completedStatus": "SUCCEEDED",
                    "startedAt": "2026-01-01T00:00:00+00:00",
                    "endedAt": "2026-01-01T00:05:00+00:00",
                }
            },
        )
        stubber.assert_no_pending_responses()


class TestWorkerCredentials(unittest.TestCase):
    """Credentials are obtained on first use and kept current by botocore."""

    @staticmethod
    def _api_credentials(expiration):
        return {
            "accessKeyId": "AKIAEXAMPLE",
            "secretAccessKey": "secret",
            "sessionToken": "token",
            "expiration": expiration,
        }

    def _worker(self, **kwargs):
        return worker_protocol.DeadlineWorker(
            farm_id=harness.FARM_ID,
            fleet_id=harness.FLEET_ID,
            region="us-west-2",
            **kwargs,
        )

    def test_a_worker_id_is_required_before_the_role_can_be_assumed(self):
        worker = self._worker()
        with self.assertRaises(worker_protocol.WorkerProtocolError):
            worker.update_worker_schedule()

    def test_the_fleet_role_is_assumed_on_first_use(self):
        worker = self._worker(worker_id=harness.WORKER_ID)
        fetched = {
            "access_key": "AKIAFETCHED",
            "secret_key": "secret",
            "token": "token",
            "expiry_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        with mock.patch.object(
            worker, "_fetch_fleet_role_credentials", return_value=fetched
        ) as fetch:
            worker._worker_client()
            fetch.assert_called_once()
            # Cached: a second call must not re-assume or rebuild.
            worker._worker_client()
            fetch.assert_called_once()

    def test_credentials_are_refreshable_rather_than_static(self):
        from botocore.credentials import RefreshableCredentials

        worker = self._worker(worker_id=harness.WORKER_ID)
        worker.set_credentials(
            self._api_credentials(datetime.now(timezone.utc) + timedelta(hours=1))
        )
        # A durable worker can stay suspended for longer than a credential's lifetime, so
        # botocore has to be able to renew them itself. A checkpointed copy would expire.
        self.assertIsInstance(worker._worker_credentials, RefreshableCredentials)
        self.assertEqual(worker._worker_credentials.access_key, "AKIAEXAMPLE")

    def test_an_expiry_string_is_accepted_as_well_as_a_datetime(self):
        worker = self._worker(worker_id=harness.WORKER_ID)
        expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        worker.set_credentials(self._api_credentials(expiry))
        self.assertEqual(worker._worker_credentials.access_key, "AKIAEXAMPLE")

    def test_expiring_credentials_are_renewed_through_the_worker_callback(self):
        worker = self._worker(worker_id=harness.WORKER_ID)
        renewed = {
            "access_key": "AKIARENEWED",
            "secret_key": "secret2",
            "token": "token2",
            "expiry_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        # Patch before setting credentials: the refresh callback is bound when the
        # credentials are created, so patching afterwards would hit the network.
        with mock.patch.object(
            worker, "_fetch_fleet_role_credentials", return_value=renewed
        ) as fetch:
            worker.set_credentials(
                self._api_credentials(datetime.now(timezone.utc) + timedelta(seconds=1))
            )
            frozen = worker._worker_credentials.get_frozen_credentials()
        fetch.assert_called()
        self.assertEqual(frozen.access_key, "AKIARENEWED")


if __name__ == "__main__":
    unittest.main()
