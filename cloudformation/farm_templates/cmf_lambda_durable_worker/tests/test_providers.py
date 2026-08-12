# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the provider seam.

Run from the parent directory with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import subprocess
import sys
import time
import unittest
import unittest.mock as mock

import harness  # noqa: F401  (puts lambda/ on sys.path)

import providers
from providers import sleep as sleep_provider


class TestResolution(unittest.TestCase):
    def test_every_registered_name_resolves_to_the_two_contract_functions(self):
        for name in providers.known_providers():
            with self.subTest(provider=name):
                module = providers.resolve(name)
                self.assertTrue(callable(module.submit))
                self.assertTrue(callable(module.poll))

    def test_an_unknown_name_raises_and_lists_the_known_ones(self):
        with self.assertRaises(providers.UnknownProviderError) as caught:
            providers.resolve("seedance")
        message = str(caught.exception)
        self.assertIn("seedance", message)
        for name in providers.known_providers():
            self.assertIn(name, message)

    def test_resolving_sleep_does_not_import_an_aws_sdk(self):
        # Lazy imports are the point of the registry: a provider that needs no
        # credentials must not drag in another provider's client library. Checked in a
        # fresh interpreter, because this one has already imported boto3.
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import providers; providers.resolve('sleep');"
            "assert 'boto3' not in sys.modules;"
            "assert 'providers.bedrock_async' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", script, str(harness.LAMBDA_DIR)], check=True
        )


class TestSleepProvider(unittest.TestCase):
    def test_a_finished_job_reports_succeeded_with_its_output(self):
        submitted = sleep_provider.submit(
            {"seconds": 0, "outputUri": "s3://bucket/key"}, task_id="task-1"
        )
        status = sleep_provider.poll(submitted["handle"])
        self.assertEqual(status["state"], "SUCCEEDED")
        self.assertEqual(status["outputUri"], "s3://bucket/key")

    def test_an_unfinished_job_reports_running(self):
        submitted = sleep_provider.submit({"seconds": 600}, task_id="task-1")
        self.assertEqual(sleep_provider.poll(submitted["handle"])["state"], "RUNNING")

    def test_the_handle_is_a_plain_json_value(self):
        # Handles cross a checkpoint boundary, so a provider cannot hand back an object.
        handle = sleep_provider.submit({"seconds": 5}, task_id="task-1")["handle"]
        self.assertIsInstance(handle, dict)
        self.assertGreater(handle["finishAt"], time.time() - 1)

    def test_a_failure_can_be_requested_so_the_worker_failure_path_is_reachable(self):
        submitted = sleep_provider.submit(
            {"seconds": 0, "failMessage": "out of pixels"}, task_id="task-1"
        )
        status = sleep_provider.poll(submitted["handle"])
        self.assertEqual(status["state"], "FAILED")
        self.assertEqual(status["message"], "out of pixels")

    def test_a_malformed_request_is_rejected_without_being_retried(self):
        result = sleep_provider.submit({"seconds": "soon"}, task_id="task-1")
        self.assertNotIn("handle", result)
        self.assertFalse(result["retryable"])


class TestBedrockAsyncProvider(unittest.TestCase):
    """The one Bedrock-aware module. It passes the model request through untouched."""

    def setUp(self):
        from providers import bedrock_async

        self.bedrock_async = bedrock_async
        self.client = mock.MagicMock()
        patcher = mock.patch.object(bedrock_async, "_client", return_value=self.client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_model_id_and_model_input_are_passed_through_verbatim(self):
        # This is what lets a job template change model or generation settings with no
        # code change, so nothing here may add, rename, or default a field.
        self.client.start_async_invoke.return_value = {"invocationArn": "arn:invoke/1"}
        model_input = {"prompt": "a red car", "duration": "9s", "anything": {"nested": 1}}

        result = self.bedrock_async.submit(
            {"modelId": "vendor.model-v9:0", "modelInput": model_input}, task_id="task-1"
        )

        self.assertEqual(result, {"handle": "arn:invoke/1"})
        kwargs = self.client.start_async_invoke.call_args.kwargs
        self.assertEqual(kwargs["modelId"], "vendor.model-v9:0")
        self.assertEqual(kwargs["modelInput"], model_input)

    def test_output_goes_to_a_prefix_named_for_the_task(self):
        # Concurrent workers must not collide in S3, and the location has to be identical
        # across replays, which the task ID gives for free.
        self.client.start_async_invoke.return_value = {"invocationArn": "arn:invoke/1"}
        self.bedrock_async.submit(
            {"modelId": "m", "modelInput": {}}, task_id="task-7"
        )
        s3_uri = self.client.start_async_invoke.call_args.kwargs["outputDataConfig"][
            "s3OutputDataConfig"
        ]["s3Uri"]
        self.assertTrue(s3_uri.endswith("/task-7/"))

    def test_a_request_missing_its_model_fields_is_rejected(self):
        result = self.bedrock_async.submit({"modelInput": {}}, task_id="task-1")
        self.assertNotIn("handle", result)
        self.assertFalse(result["retryable"])
        self.client.start_async_invoke.assert_not_called()

    def test_a_throttle_is_reported_as_retryable(self):
        # Per-account generation concurrency is low enough that a fleet scaling out will
        # collide with it, and the worker's retry wait is unbilled.
        self.client.start_async_invoke.side_effect = harness.client_error(
            "ThrottlingException", "StartAsyncInvoke"
        )
        result = self.bedrock_async.submit({"modelId": "m", "modelInput": {}}, task_id="t")
        self.assertTrue(result["retryable"])

    def test_a_validation_error_is_not_retryable(self):
        self.client.start_async_invoke.side_effect = harness.client_error(
            "ValidationException", "StartAsyncInvoke"
        )
        result = self.bedrock_async.submit({"modelId": "m", "modelInput": {}}, task_id="t")
        self.assertFalse(result["retryable"])
        self.assertIn("ValidationException", result["error"])

    def test_invocation_status_maps_onto_the_three_provider_states(self):
        cases = {
            "InProgress": "RUNNING",
            "Completed": "SUCCEEDED",
            "Failed": "FAILED",
        }
        for status, expected in cases.items():
            with self.subTest(status=status):
                self.client.get_async_invoke.return_value = {
                    "status": status,
                    "failureMessage": "model said no",
                    "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/k/"}},
                }
                self.assertEqual(self.bedrock_async.poll("arn:invoke/1")["state"], expected)

    def test_an_unreadable_status_keeps_the_request_running(self):
        # A failed status read is not evidence the invocation failed, and reporting FAILED
        # would abandon a request that is probably still going.
        self.client.get_async_invoke.side_effect = harness.client_error(
            "ThrottlingException", "GetAsyncInvoke"
        )
        with self.assertLogs(self.bedrock_async.logger, level="WARNING"):
            self.assertEqual(self.bedrock_async.poll("arn:invoke/1")["state"], "RUNNING")


if __name__ == "__main__":
    unittest.main()
