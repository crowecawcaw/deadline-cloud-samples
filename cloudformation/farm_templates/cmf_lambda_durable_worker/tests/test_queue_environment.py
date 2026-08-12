# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for queue environment handling.

Reporting success for an environment whose script never ran is the silent failure these
tests exist to prevent: the task would go on to run unprepared, and nothing would say so.

Run from the parent directory with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

import harness  # noqa: F401  (puts lambda/ on sys.path)

import queue_environment


def _details(environment: dict) -> dict:
    """An `environmentDetails` entity in the shape BatchGetJobEntity returns.

    Verified live: an environment authored as `{"environment": {...}}` comes back with
    `name` and `variables` at the top level of `template`.
    """
    return {
        "jobId": "job-" + "0" * 32,
        "environmentId": "env-" + "0" * 32,
        "schemaVersion": "environment-2023-09",
        "template": dict(environment),
    }


class TestVariablesOnlyEnvironments(unittest.TestCase):
    def test_variables_are_returned(self):
        applied = queue_environment.apply(
            environment_details=_details(
                {"name": "Config", "variables": {"RENDER_QUALITY": "high", "SEED": "7"}}
            )
        )
        self.assertEqual(applied, {"RENDER_QUALITY": "high", "SEED": "7"})

    def test_non_string_values_are_coerced(self):
        applied = queue_environment.apply(
            environment_details=_details({"name": "Config", "variables": {"SAMPLES": 16}})
        )
        self.assertEqual(applied, {"SAMPLES": "16"})

    def test_an_environment_with_nothing_to_apply_succeeds(self):
        self.assertEqual(
            queue_environment.apply(environment_details=_details({"name": "Empty"})), {}
        )


class TestScriptedEnvironmentsAreRefused(unittest.TestCase):
    def test_a_script_raises_rather_than_being_ignored(self):
        with self.assertRaises(queue_environment.UnsupportedEnvironmentError) as caught:
            queue_environment.apply(
                environment_details=_details(
                    {
                        "name": "Conda",
                        "script": {
                            "actions": {"onEnter": {"command": "conda-queue-env-enter"}}
                        },
                    }
                )
            )
        message = str(caught.exception)
        # The message is what a user sees on the failed action in the monitor, so it has
        # to name the environment and say what to do about it.
        self.assertIn("Conda", message)
        self.assertIn("script", message)

    def test_a_script_is_refused_even_when_variables_are_also_present(self):
        # Applying the variables and skipping the script would be the worst outcome:
        # partially prepared, and reported as fine.
        with self.assertRaises(queue_environment.UnsupportedEnvironmentError):
            queue_environment.apply(
                environment_details=_details(
                    {
                        "name": "Hybrid",
                        "variables": {"A": "1"},
                        "script": {"actions": {"onEnter": {"command": "setup"}}},
                    }
                )
            )

    def test_malformed_variables_are_refused(self):
        with self.assertRaises(queue_environment.UnsupportedEnvironmentError):
            queue_environment.apply(
                environment_details=_details({"name": "Bad", "variables": ["not", "a", "map"]})
            )


class TestTemplateShapes(unittest.TestCase):
    """Both the unwrapped shape the service returns and the nested authored shape work."""

    def test_the_unwrapped_shape_the_service_returns(self):
        applied = queue_environment.apply(
            environment_details={
                "environmentId": "env-1",
                "template": {"name": "Config", "variables": {"A": "1"}},
            }
        )
        self.assertEqual(applied, {"A": "1"})

    def test_the_nested_shape_an_authored_template_uses(self):
        # What a reader sees in queue_environments/. Reading only the unwrapped shape
        # silently found no variables and reported success.
        applied = queue_environment.apply(
            environment_details={
                "environmentId": "env-1",
                "template": {"environment": {"name": "Config", "variables": {"A": "1"}}},
            }
        )
        self.assertEqual(applied, {"A": "1"})

    def test_a_script_is_refused_in_the_nested_shape_too(self):
        with self.assertRaises(queue_environment.UnsupportedEnvironmentError):
            queue_environment.apply(
                environment_details={
                    "environmentId": "env-1",
                    "template": {
                        "environment": {"name": "Conda", "script": {"actions": {}}}
                    },
                }
            )


class TestMissingTemplateFields(unittest.TestCase):
    def test_an_absent_template_is_treated_as_empty(self):
        self.assertEqual(
            queue_environment.apply(
                environment_details={"environmentId": "env-1", "jobId": "job-1"}
            ),
            {},
        )

    def test_an_absent_environment_key_is_treated_as_empty(self):
        self.assertEqual(
            queue_environment.apply(
                environment_details={"environmentId": "env-1", "template": {}}
            ),
            {},
        )


if __name__ == "__main__":
    unittest.main()
