# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the provider seam.

A provider now only polls: starting the request is the job template's own business. That makes
the contract one function, and these tests hold it there.

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
    def test_every_registered_name_resolves_to_the_one_contract_function(self):
        for name in providers.known_providers():
            with self.subTest(provider=name):
                module = providers.resolve(name)
                self.assertTrue(callable(module.poll))
                # `submit` moved into the job template's script. A provider that still had
                # one would mean two ways to start the same request.
                self.assertFalse(hasattr(module, "submit"))

    def test_an_unknown_name_raises_and_lists_the_known_ones(self):
        with self.assertRaises(providers.UnknownProviderError) as caught:
            providers.resolve("seedance")
        message = str(caught.exception)
        self.assertIn("seedance", message)
        for name in providers.known_providers():
            self.assertIn(name, message)

    def test_resolving_sleep_does_not_import_an_aws_sdk(self):
        # Lazy imports are the point of the registry: a provider that needs no credentials
        # must not drag in another provider's client library. Checked in a fresh interpreter,
        # because this one has already imported boto3.
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import providers; providers.resolve('sleep');"
            "assert 'boto3' not in sys.modules;"
            "assert 'providers.bedrock_async' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", script, str(harness.LAMBDA_DIR)], check=True)


class TestSleepProvider(unittest.TestCase):
    """A handle a job template can produce with `date`, so the await path needs no SDK."""

    def test_a_finished_handle_reports_succeeded(self):
        status = sleep_provider.poll({"finishAt": time.time() - 1})
        self.assertEqual(status["state"], "SUCCEEDED")

    def test_an_unfinished_handle_reports_running_with_the_time_left(self):
        status = sleep_provider.poll({"finishAt": time.time() + 600})
        self.assertEqual(status["state"], "RUNNING")
        self.assertIn("remaining", status["message"])

    def test_a_handle_needs_nothing_but_finish_at(self):
        # The whole point: a shell script can print this with `date +%s`.
        self.assertEqual(sleep_provider.poll({"finishAt": 0})["state"], "SUCCEEDED")

    def test_a_string_epoch_is_accepted_because_a_shell_produces_strings(self):
        self.assertEqual(sleep_provider.poll({"finishAt": "0"})["state"], "SUCCEEDED")

    def test_an_output_uri_is_passed_through_when_given(self):
        status = sleep_provider.poll({"finishAt": 0, "outputUri": "s3://bucket/key"})
        self.assertEqual(status["outputUri"], "s3://bucket/key")

    def test_a_failure_can_be_requested_so_the_worker_failure_path_is_reachable(self):
        status = sleep_provider.poll({"finishAt": 0, "failMessage": "out of pixels"})
        self.assertEqual(status["state"], "FAILED")
        self.assertEqual(status["message"], "out of pixels")


class TestBedrockAsyncProvider(unittest.TestCase):
    """The one Bedrock-aware module, and now only its status read."""

    def setUp(self):
        from providers import bedrock_async

        self.bedrock_async = bedrock_async
        self.client = mock.MagicMock()
        patcher = mock.patch.object(bedrock_async, "_client", return_value=self.client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_handle_is_the_invocation_arn_the_task_script_printed(self):
        self.client.get_async_invoke.return_value = {"status": "InProgress"}
        self.bedrock_async.poll("arn:invoke/1")
        self.assertEqual(
            self.client.get_async_invoke.call_args.kwargs, {"invocationArn": "arn:invoke/1"}
        )

    def test_invocation_status_maps_onto_the_three_provider_states(self):
        cases = {"InProgress": "RUNNING", "Completed": "SUCCEEDED", "Failed": "FAILED"}
        for status, expected in cases.items():
            with self.subTest(status=status):
                self.client.get_async_invoke.return_value = {
                    "status": status,
                    "failureMessage": "model said no",
                    "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/k/"}},
                }
                self.assertEqual(self.bedrock_async.poll("arn:invoke/1")["state"], expected)

    def test_a_completed_invocation_reports_where_its_output_went(self):
        self.client.get_async_invoke.return_value = {
            "status": "Completed",
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/k/"}},
        }
        self.assertEqual(self.bedrock_async.poll("arn:invoke/1")["outputUri"], "s3://b/k/")

    def test_an_unrecognized_status_keeps_the_request_running(self):
        self.client.get_async_invoke.return_value = {"status": "Submitted"}
        self.assertEqual(self.bedrock_async.poll("arn:invoke/1")["state"], "RUNNING")

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
