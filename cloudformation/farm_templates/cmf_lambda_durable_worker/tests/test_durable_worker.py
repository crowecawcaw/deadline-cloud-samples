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


class TestTimestampConversion(unittest.TestCase):
    """Timestamps cross a JSON checkpoint boundary as strings."""

    def test_iso_strings_become_datetimes(self):
        from datetime import datetime

        converted = worker_protocol._deserialize_timestamps(
            {
                "action-1": {
                    "completedStatus": "SUCCEEDED",
                    "startedAt": "2026-01-01T00:00:00+00:00",
                    "endedAt": "2026-01-01T00:05:00+00:00",
                }
            }
        )
        entry = converted["action-1"]
        self.assertIsInstance(entry["startedAt"], datetime)
        self.assertIsInstance(entry["endedAt"], datetime)
        # Non-timestamp fields pass through untouched.
        self.assertEqual(entry["completedStatus"], "SUCCEEDED")

    def test_absent_timestamps_are_left_alone(self):
        converted = worker_protocol._deserialize_timestamps(
            {"action-1": {"completedStatus": "SUCCEEDED"}}
        )
        self.assertEqual(converted["action-1"], {"completedStatus": "SUCCEEDED"})

    def test_input_is_not_mutated(self):
        # The caller keeps these results for its next heartbeat, so converting must
        # not turn the checkpointed strings into datetimes in place.
        original = {"action-1": {"startedAt": "2026-01-01T00:00:00+00:00"}}
        worker_protocol._deserialize_timestamps(original)
        self.assertIsInstance(original["action-1"]["startedAt"], str)


class TestWorkerProtocolGuards(unittest.TestCase):
    def test_worker_credentials_are_required_for_scheduling_calls(self):
        # Calling a worker-credentialed API before AssumeFleetRoleForWorker is a
        # programming error, and should say so rather than fail inside botocore.
        worker = worker_protocol.DeadlineWorker(
            farm_id="farm-" + "0" * 32,
            fleet_id="fleet-" + "0" * 32,
            region="us-west-2",
            worker_id="worker-" + "0" * 32,
        )
        with self.assertRaises(worker_protocol.WorkerProtocolError):
            worker.update_worker_schedule()


if __name__ == "__main__":
    unittest.main()
