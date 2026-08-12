# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the environment a session action runs in.

A session cannot outlive one invocation here, so this bookkeeping is the only thing carrying a
queue environment's variables to the actions that follow it. Getting the layering wrong is
invisible until a task runs unprepared.

Run from the parent directory with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone

import harness  # noqa: F401  (puts lambda/ on sys.path)

import session_env

CREDENTIALS = {
    "accessKeyId": "AKIAQUEUEROLE",
    "secretAccessKey": "secret",
    "sessionToken": "token",
    "expiration": datetime(2026, 1, 1, tzinfo=timezone.utc),
}


class TestBaseEnvironment(unittest.TestCase):
    def test_the_region_is_given_in_both_spellings_boto3_reads(self):
        env = session_env.base_env(region="eu-west-1")
        self.assertEqual(env["AWS_REGION"], "eu-west-1")
        self.assertEqual(env["AWS_DEFAULT_REGION"], "eu-west-1")

    def test_the_interpreter_directory_is_on_path_so_a_template_can_run_python3(self):
        env = session_env.base_env(region="us-west-2")
        self.assertIn(os.path.dirname(sys.executable), env["PATH"].split(os.pathsep))

    def test_queue_credentials_are_passed_to_the_task(self):
        env = session_env.base_env(region="us-west-2", credentials=CREDENTIALS)
        self.assertEqual(env["AWS_ACCESS_KEY_ID"], "AKIAQUEUEROLE")
        self.assertEqual(env["AWS_SECRET_ACCESS_KEY"], "secret")
        self.assertEqual(env["AWS_SESSION_TOKEN"], "token")

    def test_the_credential_expiry_is_replaced_along_with_the_keys(self):
        # Leaving the runtime's own expiry behind would have botocore try to refresh these
        # credentials with it.
        env = session_env.base_env(region="us-west-2", credentials=CREDENTIALS)
        self.assertEqual(env["AWS_CREDENTIAL_EXPIRATION"], "2026-01-01T00:00:00+00:00")

    def test_an_expiry_string_is_accepted_as_well_as_a_datetime(self):
        credentials = {**CREDENTIALS, "expiration": "2026-01-01T00:00:00+00:00"}
        env = session_env.base_env(region="us-west-2", credentials=credentials)
        self.assertEqual(env["AWS_CREDENTIAL_EXPIRATION"], "2026-01-01T00:00:00+00:00")

    def test_without_a_queue_role_every_credential_variable_is_removed(self):
        # The subprocess starts from a copy of the worker's own environment, so leaving these
        # alone would let a task's script act as the worker.
        env = session_env.base_env(region="us-west-2")
        for name in session_env.CREDENTIAL_VARIABLES:
            self.assertIsNone(env[name], name)


class TestLayering(unittest.TestCase):
    def test_layers_apply_in_entry_order_so_a_later_one_wins(self):
        base = {"TOOL": "/base"}
        layers = [["env-1", {"set": {"TOOL": "/one"}}], ["env-2", {"set": {"TOOL": "/two"}}]]
        self.assertEqual(session_env.compose(base, layers)["TOOL"], "/two")

    def test_an_unset_becomes_a_removal_rather_than_an_absence(self):
        # An absent key would be inherited from the worker's own environment instead.
        composed = session_env.compose({"PYTHONHOME": "/usr"}, [["env-1", {"unset": ["PYTHONHOME"]}]])
        self.assertIn("PYTHONHOME", composed)
        self.assertIsNone(composed["PYTHONHOME"])

    def test_a_later_layer_can_set_what_an_earlier_one_unset(self):
        layers = [["env-1", {"unset": ["TOOL"]}], ["env-2", {"set": {"TOOL": "/two"}}]]
        self.assertEqual(session_env.compose({}, layers)["TOOL"], "/two")

    def test_composing_does_not_mutate_the_base(self):
        base = {"A": "1"}
        session_env.compose(base, [["env-1", {"set": {"A": "2"}}]])
        self.assertEqual(base, {"A": "1"})

    def test_a_missing_set_or_unset_key_is_tolerated(self):
        self.assertEqual(session_env.compose({"A": "1"}, [["env-1", {}]]), {"A": "1"})


class TestUnlayering(unittest.TestCase):
    def test_dropping_one_environment_keeps_the_rest_in_order(self):
        layers = [
            ["env-1", {"set": {"A": "1"}}],
            ["env-2", {"set": {"B": "2"}}],
            ["env-3", {"set": {"C": "3"}}],
        ]
        remaining = session_env.drop(layers, "env-2")
        self.assertEqual([layer[0] for layer in remaining], ["env-1", "env-3"])

    def test_an_exit_undoes_only_its_own_environments_variables(self):
        layers = [["env-1", {"set": {"TOOL": "/one"}}], ["env-2", {"set": {"TOOL": "/two"}}]]
        # Merged layers could not do this: after env-2 exits, env-1's value has to come back.
        self.assertEqual(session_env.compose({}, session_env.drop(layers, "env-2"))["TOOL"], "/one")

    def test_dropping_an_environment_that_was_never_entered_changes_nothing(self):
        layers = [["env-1", {"set": {"A": "1"}}]]
        self.assertEqual(session_env.drop(layers, "env-9"), layers)


if __name__ == "__main__":
    unittest.main()
