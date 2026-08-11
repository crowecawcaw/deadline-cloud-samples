# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the durable Lambda worker's pure logic.

These cover the parts that are easy to get wrong and hard to notice in a live fleet:
the translation between Deadline Cloud's wire format and the worker's internal
representation, and the mapping from task parameters to a Bedrock request.

Run from this directory with:

    python3 -m unittest discover -s tests

The durable execution SDK is stubbed out so the tests need no AWS credentials and no
Lambda runtime. Only module-level imports are stubbed; the functions under test are
plain data transformations.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

LAMBDA_DIR = Path(__file__).resolve().parents[1] / "lambda"
sys.path.insert(0, str(LAMBDA_DIR))

# The worker imports the durable execution SDK at module scope, and the SDK is only
# present inside the Lambda runtime. Stub the three names it uses so the module
# imports here. The decorators are identity functions because these tests exercise the
# helpers, not the checkpointing behavior, which belongs to Lambda.
_sdk = types.ModuleType("aws_durable_execution_sdk_python")
_sdk.DurableContext = object
_sdk.durable_execution = lambda fn: fn
_sdk.durable_step = lambda fn: fn
_sdk_config = types.ModuleType("aws_durable_execution_sdk_python.config")


class _Duration:
    @staticmethod
    def from_seconds(seconds):
        return seconds


_sdk_config.Duration = _Duration
sys.modules.setdefault("aws_durable_execution_sdk_python", _sdk)
sys.modules.setdefault("aws_durable_execution_sdk_python.config", _sdk_config)

# Required at import time by the worker module.
os.environ.setdefault("FARM_ID", "farm-" + "0" * 32)
os.environ.setdefault("FLEET_ID", "fleet-" + "0" * 32)
os.environ.setdefault("OUTPUT_BUCKET", "test-bucket")

import bedrock_task  # noqa: E402
import durable_worker  # noqa: E402
import worker_protocol  # noqa: E402


class TestParameterUnwrapping(unittest.TestCase):
    """Deadline Cloud tags each task parameter with its type."""

    def test_unwraps_each_supported_type(self):
        self.assertEqual(durable_worker._unwrap_parameter({"string": "hello"}), "hello")
        self.assertEqual(durable_worker._unwrap_parameter({"int": "42"}), "42")
        self.assertEqual(durable_worker._unwrap_parameter({"float": "1.5"}), "1.5")
        self.assertEqual(durable_worker._unwrap_parameter({"path": "/tmp/x"}), "/tmp/x")

    def test_unknown_tag_yields_none(self):
        # A parameter type this worker does not understand must not raise, because one
        # unfamiliar parameter should not fail the whole assignment.
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
                                    "Prompt": {"string": "a red car"},
                                    "Frames": {"int": "10"},
                                },
                            }
                        },
                    }
                ],
            }
        }

        summary = durable_worker._summarize_sessions(assigned)
        action = summary["session-1"]["sessionActions"][0]

        self.assertEqual(summary["session-1"]["jobId"], "job-abc")
        self.assertEqual(action["definition"]["taskRun"]["parameters"]["Prompt"], "a red car")
        self.assertEqual(action["definition"]["taskRun"]["parameters"]["Frames"], "10")
        # Log configuration is dropped: this worker does not stream session logs, and
        # checkpoint payloads are size-limited.
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


class TestBedrockModelInput(unittest.TestCase):
    """Task parameters describe the request; this is that mapping."""

    def test_defaults_apply_when_parameters_are_absent(self):
        model_input = bedrock_task.build_model_input({})
        self.assertEqual(model_input["duration"], "5s")
        self.assertEqual(model_input["resolution"], "540p")
        self.assertEqual(model_input["aspect_ratio"], "16:9")
        self.assertTrue(model_input["prompt"])

    def test_task_parameters_override_defaults(self):
        model_input = bedrock_task.build_model_input(
            {
                "Prompt": "a misty forest",
                "Duration": "9s",
                "Resolution": "720p",
                "AspectRatio": "1:1",
            }
        )
        self.assertEqual(model_input["prompt"], "a misty forest")
        self.assertEqual(model_input["duration"], "9s")
        self.assertEqual(model_input["resolution"], "720p")
        self.assertEqual(model_input["aspect_ratio"], "1:1")

    def test_empty_prompt_falls_back_to_default(self):
        # An empty string would be rejected by the model, so it must not pass through.
        model_input = bedrock_task.build_model_input({"Prompt": ""})
        self.assertTrue(model_input["prompt"])


class TestCapabilities(unittest.TestCase):
    """Capabilities are what keep Bedrock steps on these workers."""

    def test_declares_the_targeting_attribute(self):
        capabilities = worker_protocol.default_capabilities()
        attributes = {a["name"]: a["values"] for a in capabilities["attributes"]}
        self.assertEqual(attributes["attr.durable.lambda"], ["true"])

    def test_standard_capability_names_are_used(self):
        capabilities = worker_protocol.default_capabilities()
        amounts = {a["name"] for a in capabilities["amounts"]}
        attributes = {a["name"] for a in capabilities["attributes"]}
        # Open Job Description reserves the `amount.worker` and `attr.worker`
        # prefixes for standard capabilities, so these names must match the spec.
        self.assertIn("amount.worker.vcpu", amounts)
        self.assertIn("amount.worker.memory", amounts)
        self.assertIn("attr.worker.os.family", attributes)
        self.assertIn("attr.worker.cpu.arch", attributes)


class TestThrottleHandling(unittest.TestCase):
    """Throttles must retry; bad requests must fail the task."""

    @staticmethod
    def _client_error(code):
        from botocore.exceptions import ClientError

        return ClientError(
            {"Error": {"Code": code, "Message": f"simulated {code}"}}, "StartAsyncInvoke"
        )

    def test_throttling_is_reported_as_retryable(self):
        import unittest.mock as mock

        with mock.patch.object(bedrock_task, "_client") as fake:
            fake.return_value.start_async_invoke.side_effect = self._client_error(
                "ThrottlingException"
            )
            result = bedrock_task.start_generation(task_parameters={"Prompt": "x"})
        # Reported rather than raised, so the caller can retry behind a durable wait.
        # Raising would leave it to the step's finite, closely spaced retries, which a
        # throttling window outlasts.
        self.assertIsNone(result["invocationArn"])
        self.assertTrue(result["throttled"])

    def test_validation_error_fails_the_task(self):
        import unittest.mock as mock

        with mock.patch.object(bedrock_task, "_client") as fake:
            fake.return_value.start_async_invoke.side_effect = self._client_error(
                "ValidationException"
            )
            result = bedrock_task.start_generation(task_parameters={"Prompt": "x"})
        # A malformed request will never succeed, so it is reported and not retried.
        self.assertIsNone(result["invocationArn"])
        self.assertFalse(result["throttled"])
        self.assertIn("ValidationException", result["error"])


class TestTimestampSerialization(unittest.TestCase):
    """Session action results carry timestamps as strings across a checkpoint."""

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
        # Checkpoints hold JSON, so timestamps arrive as strings. This pins the reason
        # no conversion is needed: botocore serializes ISO-8601 strings for timestamp
        # members itself. If that ever stopped being true, this fails loudly here
        # rather than as a ParamValidationError against a live worker.
        client.update_worker_schedule(
            farmId="farm-" + "0" * 32,
            fleetId="fleet-" + "0" * 32,
            workerId="worker-" + "0" * 32,
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

    FARM = "farm-" + "0" * 32
    FLEET = "fleet-" + "0" * 32
    WORKER = "worker-" + "0" * 32

    @staticmethod
    def _api_credentials(expiration):
        """A credentials block shaped the way AssumeFleetRoleForWorker returns one."""
        return {
            "accessKeyId": "AKIAEXAMPLE",
            "secretAccessKey": "secret",
            "sessionToken": "token",
            "expiration": expiration,
        }

    def _worker(self, **kwargs):
        return worker_protocol.DeadlineWorker(
            farm_id=self.FARM, fleet_id=self.FLEET, region="us-west-2", **kwargs
        )

    def test_a_worker_id_is_required_before_the_role_can_be_assumed(self):
        # Without a worker ID there is nothing to assume the role for, and saying so is
        # clearer than letting the request fail inside botocore.
        worker = self._worker()
        with self.assertRaises(worker_protocol.WorkerProtocolError):
            worker.update_worker_schedule()

    def test_the_fleet_role_is_assumed_on_first_use(self):
        import unittest.mock as mock
        from datetime import datetime, timedelta, timezone

        worker = self._worker(worker_id=self.WORKER)
        fetched = {
            "access_key": "AKIAFETCHED",
            "secret_key": "secret",
            "token": "token",
            "expiry_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        # Callers no longer assume the role themselves, so the first client build has
        # to do it. Patching the fetch keeps this off the network.
        with mock.patch.object(
            worker, "_fetch_fleet_role_credentials", return_value=fetched
        ) as fetch:
            worker._worker_client()
            fetch.assert_called_once()
            # Cached: a second call must not re-assume or rebuild.
            worker._worker_client()
            fetch.assert_called_once()

    def test_credentials_are_refreshable_rather_than_static(self):
        from datetime import datetime, timedelta, timezone
        from botocore.credentials import RefreshableCredentials

        worker = self._worker(worker_id=self.WORKER)
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        worker.set_credentials(self._api_credentials(expiry))

        # Refreshable, not static: a durable worker can stay suspended for longer than
        # a credential's lifetime, so botocore has to be able to renew them itself.
        self.assertIsInstance(worker._worker_credentials, RefreshableCredentials)
        self.assertEqual(worker._worker_credentials.access_key, "AKIAEXAMPLE")

    def test_an_expiry_string_is_accepted_as_well_as_a_datetime(self):
        from datetime import datetime, timedelta, timezone

        worker = self._worker(worker_id=self.WORKER)
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        # botocore parses expiration into a datetime, but credentials restored from
        # JSON arrive as a string. Both have to work.
        worker.set_credentials(self._api_credentials(expiry.isoformat()))
        self.assertEqual(worker._worker_credentials.access_key, "AKIAEXAMPLE")

    def test_expiring_credentials_are_renewed_through_the_worker_callback(self):
        import unittest.mock as mock
        from datetime import datetime, timedelta, timezone

        worker = self._worker(worker_id=self.WORKER)
        renewed = {
            "access_key": "AKIARENEWED",
            "secret_key": "secret2",
            "token": "token2",
            "expiry_time": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
        }
        # Patch before setting credentials: the refresh callback is bound when the
        # credentials are created, so patching afterwards would leave the real method
        # wired in and the refresh would hit the network.
        with mock.patch.object(
            worker, "_fetch_fleet_role_credentials", return_value=renewed
        ) as fetch:
            # Already inside the mandatory refresh window, so reading the credentials
            # must renew them instead of handing back a nearly expired key.
            worker.set_credentials(
                self._api_credentials(datetime.now(timezone.utc) + timedelta(seconds=1))
            )
            frozen = worker._worker_credentials.get_frozen_credentials()
        fetch.assert_called()
        self.assertEqual(frozen.access_key, "AKIARENEWED")


if __name__ == "__main__":
    unittest.main()
