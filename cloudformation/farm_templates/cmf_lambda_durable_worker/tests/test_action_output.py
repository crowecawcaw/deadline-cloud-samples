# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the two stdout line protocols the worker reads.

These are the whole contract between a job template and this worker's extension, so a change
here silently changes what every job template has to print.

Run from the parent directory with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

import harness  # noqa: F401  (puts lambda/ on sys.path)

import action_output


class TestAwaitTokenPrefix(unittest.TestCase):
    def test_the_prefix_stays_out_of_the_specification_namespace(self):
        # `openjd_` belongs to Open Job Description, which could add a colliding token.
        self.assertEqual(action_output.AWAIT_PREFIX, "durable_lambda_await:")
        self.assertFalse(action_output.AWAIT_PREFIX.startswith("openjd"))


class TestAwaitTokenHarvesting(unittest.TestCase):
    def test_no_token_is_an_ordinary_action(self):
        lines = ["building frame 1", "openjd_progress: 50", "done"]
        self.assertEqual(action_output.await_tokens(lines), [])

    def test_one_token_is_returned_with_its_provider_and_handle(self):
        lines = ['durable_lambda_await: {"provider": "bedrock-async", "handle": "arn:invoke/1"}']
        self.assertEqual(
            action_output.await_tokens(lines),
            [{"provider": "bedrock-async", "handle": "arn:invoke/1"}],
        )

    def test_more_than_one_token_is_reported_in_order_for_the_caller_to_reject(self):
        lines = [
            'durable_lambda_await: {"provider": "sleep", "handle": 1}',
            'durable_lambda_await: {"provider": "sleep", "handle": 2}',
        ]
        tokens = action_output.await_tokens(lines)
        self.assertEqual([token["handle"] for token in tokens], [1, 2])

    def test_a_handle_may_be_any_json_value(self):
        # The worker never inspects a handle, so a provider is free to shape it.
        lines = ['durable_lambda_await: {"provider": "sleep", "handle": {"finishAt": 12}}']
        self.assertEqual(action_output.await_tokens(lines)[0]["handle"], {"finishAt": 12})

    def test_leading_whitespace_does_not_hide_a_token(self):
        lines = ['   durable_lambda_await: {"provider": "sleep", "handle": 1}']
        self.assertEqual(len(action_output.await_tokens(lines)), 1)

    def test_unparseable_json_is_rejected_rather_than_ignored(self):
        # Ignoring it would report the task succeeded while its request was still running.
        with self.assertRaises(action_output.MalformedOutputError):
            action_output.await_tokens(["durable_lambda_await: not json"])

    def test_a_token_without_a_provider_is_rejected(self):
        with self.assertRaises(action_output.MalformedOutputError):
            action_output.await_tokens(['durable_lambda_await: {"handle": "h"}'])

    def test_a_token_without_a_handle_is_rejected(self):
        with self.assertRaises(action_output.MalformedOutputError):
            action_output.await_tokens(['durable_lambda_await: {"provider": "sleep"}'])


class TestEnvironmentDeltas(unittest.TestCase):
    def test_sets_and_unsets_are_collected_separately(self):
        lines = [
            "openjd_env: CONDA_PREFIX=/opt/conda",
            "openjd_env: PATH=/opt/conda/bin:/usr/bin",
            "openjd_unset_env: PYTHONHOME",
        ]
        self.assertEqual(
            action_output.env_delta(lines),
            {
                "set": {"CONDA_PREFIX": "/opt/conda", "PATH": "/opt/conda/bin:/usr/bin"},
                "unset": ["PYTHONHOME"],
            },
        )

    def test_a_value_may_contain_an_equals_sign(self):
        delta = action_output.env_delta(["openjd_env: OPTIONS=a=1,b=2"])
        self.assertEqual(delta["set"], {"OPTIONS": "a=1,b=2"})

    def test_an_empty_value_is_a_set_not_an_unset(self):
        delta = action_output.env_delta(["openjd_env: EMPTY="])
        self.assertEqual(delta["set"], {"EMPTY": ""})
        self.assertEqual(delta["unset"], [])

    def test_an_unset_of_a_name_this_action_also_set_wins(self):
        # Ordering is the caller's to apply, but the delta has to record both halves.
        delta = action_output.env_delta(
            ["openjd_env: TOOL=/a", "openjd_unset_env: TOOL"]
        )
        self.assertEqual(delta["set"], {"TOOL": "/a"})
        self.assertEqual(delta["unset"], ["TOOL"])

    def test_output_that_is_not_a_protocol_line_is_ignored(self):
        self.assertEqual(
            action_output.env_delta(["Setting: A=1", "export B=2"]),
            {"set": {}, "unset": []},
        )

    def test_a_malformed_assignment_is_rejected(self):
        with self.assertRaises(action_output.MalformedOutputError):
            action_output.env_delta(["openjd_env: NOEQUALS"])

    def test_an_unset_with_no_name_is_rejected(self):
        with self.assertRaises(action_output.MalformedOutputError):
            action_output.env_delta(["openjd_unset_env:   "])

    def test_a_redacted_value_is_refused_with_a_reason(self):
        # The library replaces the value with asterisks before the worker can see it, so
        # carrying it across a wait would set the wrong value rather than the secret.
        with self.assertRaises(action_output.MalformedOutputError) as caught:
            action_output.env_delta(["openjd_redacted_env: SECRET=********"])
        self.assertIn("redacted", str(caught.exception))


class TestCapturedPrefixes(unittest.TestCase):
    def test_every_prefix_the_worker_acts_on_is_captured(self):
        # The session runner keeps only lines starting with these, so a prefix missing here
        # would be parsed from output that was never collected.
        for prefix in (
            action_output.AWAIT_PREFIX,
            action_output.ENV_PREFIX,
            action_output.UNSET_ENV_PREFIX,
            action_output.REDACTED_ENV_PREFIX,
        ):
            self.assertIn(prefix, action_output.CAPTURED_PREFIXES)


if __name__ == "__main__":
    unittest.main()
