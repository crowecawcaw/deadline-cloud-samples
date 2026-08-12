# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for the worker's wire-format handling, entity fetching, and credentials.

Run from the parent directory with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import unittest
import unittest.mock as mock
from datetime import datetime, timedelta, timezone

import boto3
from botocore.stub import ANY, Stubber

import harness  # noqa: F401  (stubs the durable execution SDK, puts lambda/ on sys.path)
from harness import JOB_ID, QUEUE_ID, WORKER_ID, session_outcome, stub_session_runner, task_run_action

import durable_worker
import worker_protocol


class TestParameterUnwrapping(unittest.TestCase):
    """Deadline Cloud tags each parameter with its type, and the two sets differ."""

    def test_task_parameters_are_restated_in_open_job_description_types(self):
        unwrapped = worker_protocol.unwrap_parameters(
            {
                "Prompt": {"string": "a car"},
                "Frame": {"int": "42"},
                "Ratio": {"float": "1.5"},
                "Scene": {"path": "/tmp/x"},
            },
            task=True,
        )
        self.assertEqual(unwrapped["Prompt"], {"type": "STRING", "value": "a car"})
        self.assertEqual(unwrapped["Frame"], {"type": "INT", "value": "42"})
        self.assertEqual(unwrapped["Ratio"], {"type": "FLOAT", "value": "1.5"})
        self.assertEqual(unwrapped["Scene"], {"type": "PATH", "value": "/tmp/x"})

    def test_chunk_int_is_a_task_parameter_only(self):
        unwrapped = worker_protocol.unwrap_parameters({"Frames": {"chunkInt": "1-10"}}, task=True)
        self.assertEqual(unwrapped["Frames"], {"type": "CHUNK[INT]", "value": "1-10"})
        # It is not a member of the job parameter union at all, so a job parameter carrying one
        # is a wire-format error rather than a value to pass along.
        self.assertNotIn("chunkInt", worker_protocol.JOB_PARAMETER_TYPES)
        with self.assertRaises(worker_protocol.WorkerProtocolError):
            worker_protocol.unwrap_parameters({"Frames": {"chunkInt": "1-10"}}, task=False)

    def test_job_parameters_unwrap_the_four_types_the_api_defines(self):
        unwrapped = worker_protocol.unwrap_parameters(
            {"A": {"string": "s"}, "B": {"int": "1"}, "C": {"float": "2.5"}, "D": {"path": "/p"}},
            task=False,
        )
        self.assertEqual(
            {name: value["type"] for name, value in unwrapped.items()},
            {"A": "STRING", "B": "INT", "C": "FLOAT", "D": "PATH"},
        )

    def test_an_unrecognized_tag_is_refused_rather_than_dropped(self):
        # Passing the action a parameter the template asked for, minus its value, would fail
        # inside the session with a far worse message.
        with self.assertRaises(worker_protocol.WorkerProtocolError) as caught:
            worker_protocol.unwrap_parameters({"Odd": {"unexpected": "v"}}, task=True)
        self.assertIn("Odd", str(caught.exception))

    def test_no_parameters_is_not_an_error(self):
        self.assertEqual(worker_protocol.unwrap_parameters(None, task=True), {})


class TestSessionSummarization(unittest.TestCase):
    """The worker shrinks the schedule response before checkpointing it."""

    def test_a_task_run_keeps_its_step_id_and_its_tagged_parameters(self):
        assigned = {
            "session-1": {
                "queueId": QUEUE_ID,
                "jobId": JOB_ID,
                "logConfiguration": {"logDriver": "awslogs", "options": {"noise": "x"}},
                "sessionActions": [
                    {
                        "sessionActionId": "action-1",
                        "definition": {
                            "taskRun": {
                                "taskId": "task-1",
                                "stepId": "step-1",
                                "parameters": {"Prompt": {"string": "a car"}},
                            }
                        },
                    }
                ],
            }
        }

        summary = durable_worker._summarize_sessions(assigned)
        task_run = summary["session-1"]["sessionActions"][0]["definition"]["taskRun"]

        self.assertEqual(summary["session-1"]["jobId"], JOB_ID)
        # The step ID is what BatchGetJobEntity needs to fetch the step template.
        self.assertEqual(task_run["stepId"], "step-1")
        # Left tagged on purpose: unwrapping can fail, and it has to fail inside the step that
        # runs the action rather than the one that collects the schedule.
        self.assertEqual(task_run["parameters"], {"Prompt": {"string": "a car"}})
        # Checkpoint payloads are size-limited and this worker streams no session logs.
        self.assertNotIn("logConfiguration", summary["session-1"])

    def test_environment_actions_are_preserved(self):
        assigned = {
            "session-1": {
                "queueId": QUEUE_ID,
                "jobId": JOB_ID,
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

    def test_a_template_is_never_checkpointed(self):
        # A step template can be far larger than the 256 KiB checkpoint limit, so it is
        # re-fetched inside the step that needs it.
        summary = durable_worker._summarize_sessions(
            {"session-1": {"queueId": QUEUE_ID, "jobId": JOB_ID, "sessionActions": []}}
        )
        self.assertNotIn("template", repr(summary))

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

    def test_memory_and_vcpu_come_from_the_runtimes_own_setting(self):
        with mock.patch.dict("os.environ", {"AWS_LAMBDA_FUNCTION_MEMORY_SIZE": "3538"}):
            amounts = {
                a["name"]: a["value"] for a in worker_protocol.default_capabilities()["amounts"]
            }
        self.assertEqual(amounts["amount.worker.memory"], 3538)
        # Lambda gives one vCPU per 1769 MB.
        self.assertEqual(amounts["amount.worker.vcpu"], 2)

    def test_scratch_space_is_reported_now_that_sessions_write_files(self):
        amounts = {
            a["name"]: a["value"] for a in worker_protocol.default_capabilities()["amounts"]
        }
        self.assertEqual(amounts["amount.worker.disk.scratch"], worker_protocol.SCRATCH_MIB)
        self.assertGreater(worker_protocol.SCRATCH_MIB, 0)


def _deadline_client():
    return boto3.client(
        "deadline", region_name="us-west-2", aws_access_key_id="a", aws_secret_access_key="b"
    )


# The members the API marks required. Spelled out because Stubber validates against the model,
# which is what makes these stubbed responses evidence about the real wire shape.
def _job_details(**extra):
    return {
        "jobDetails": {
            "jobId": JOB_ID,
            "logGroupName": "/aws/deadline/farm-x/queue-y",
            "schemaVersion": "jobtemplate-2023-09",
            **extra,
        }
    }


def _step_details(template=None, **extra):
    return {
        "stepDetails": {
            "jobId": JOB_ID,
            "stepId": "step-1",
            "schemaVersion": "jobtemplate-2023-09",
            "template": template if template is not None else {"name": "Generate"},
            "dependencies": [],
            **extra,
        }
    }


class TestJobEntityFetching(unittest.TestCase):
    """One call per action, because each round trip happens inside a billed invocation."""

    def _worker(self, client):
        worker = worker_protocol.DeadlineWorker(
            farm_id=harness.FARM_ID,
            fleet_id=harness.FLEET_ID,
            region="us-west-2",
            worker_id=WORKER_ID,
        )
        worker._client_cache[True] = client
        worker._worker_credentials = object()
        return worker

    def test_the_api_limit_is_read_from_the_model_rather_than_assumed(self):
        self.assertEqual(worker_protocol._max_identifiers(_deadline_client()), 10)

    def test_job_details_and_a_step_template_arrive_from_a_single_call(self):
        client = _deadline_client()
        stubber = Stubber(client)
        stubber.add_response(
            "batch_get_job_entity",
            {"entities": [_job_details(), _step_details()], "errors": []},
            {"farmId": ANY, "fleetId": ANY, "workerId": ANY, "identifiers": ANY},
        )
        stubber.activate()

        entities = self._worker(client).get_job_entities(
            identifiers=[
                {"jobDetails": {"jobId": JOB_ID}},
                {"stepDetails": {"jobId": JOB_ID, "stepId": "step-1"}},
            ]
        )
        stubber.assert_no_pending_responses()
        self.assertEqual(set(entities), {"jobDetails", "stepDetails"})
        self.assertEqual(entities["stepDetails"]["template"], {"name": "Generate"})

    def test_asking_for_more_identifiers_than_the_api_allows_is_refused(self):
        worker = self._worker(_deadline_client())
        with self.assertRaises(worker_protocol.WorkerProtocolError):
            worker.get_job_entities(
                identifiers=[{"jobDetails": {"jobId": JOB_ID}}] * 11
            )

    def test_an_error_names_the_entity_and_the_service_code(self):
        client = _deadline_client()
        stubber = Stubber(client)
        stubber.add_response(
            "batch_get_job_entity",
            {
                "entities": [],
                "errors": [
                    {
                        "stepDetails": {
                            "jobId": JOB_ID,
                            "stepId": "step-1",
                            "code": "ResourceNotFoundException",
                            "message": "no such step",
                        }
                    }
                ],
            },
            {"farmId": ANY, "fleetId": ANY, "workerId": ANY, "identifiers": ANY},
        )
        stubber.activate()
        with self.assertRaises(worker_protocol.WorkerProtocolError) as caught:
            self._worker(client).get_job_entities(
                identifiers=[{"stepDetails": {"jobId": JOB_ID, "stepId": "step-1"}}]
            )
        self.assertIn("ResourceNotFoundException", str(caught.exception))
        self.assertIn("no such step", str(caught.exception))

    def test_an_oversized_entity_is_re_requested_on_its_own(self):
        # A step template with large embedded files does not fit in a response alongside
        # anything else, and the whole call fails rather than that one entity.
        client = _deadline_client()
        stubber = Stubber(client)
        stubber.add_response(
            "batch_get_job_entity",
            {
                "entities": [_job_details()],
                "errors": [
                    {
                        "stepDetails": {
                            "jobId": JOB_ID,
                            "stepId": "step-1",
                            "code": "MaxPayloadSizeExceeded",
                            "message": "too big",
                        }
                    }
                ],
            },
            {"farmId": ANY, "fleetId": ANY, "workerId": ANY, "identifiers": ANY},
        )
        stubber.add_response(
            "batch_get_job_entity",
            {"entities": [_step_details(template={})], "errors": []},
            {"farmId": ANY, "fleetId": ANY, "workerId": ANY, "identifiers": ANY},
        )
        stubber.activate()

        entities = self._worker(client).get_job_entities(
            identifiers=[
                {"jobDetails": {"jobId": JOB_ID}},
                {"stepDetails": {"jobId": JOB_ID, "stepId": "step-1"}},
            ]
        )
        stubber.assert_no_pending_responses()
        self.assertEqual(set(entities), {"jobDetails", "stepDetails"})

    def test_an_entity_neither_returned_nor_reported_is_an_error(self):
        # Running the action without it would use defaults the job never asked for.
        client = _deadline_client()
        stubber = Stubber(client)
        stubber.add_response(
            "batch_get_job_entity",
            {"entities": [_job_details()], "errors": []},
            {"farmId": ANY, "fleetId": ANY, "workerId": ANY, "identifiers": ANY},
        )
        stubber.activate()
        with self.assertRaises(worker_protocol.WorkerProtocolError) as caught:
            self._worker(client).get_job_entities(
                identifiers=[
                    {"jobDetails": {"jobId": JOB_ID}},
                    {"stepDetails": {"jobId": JOB_ID, "stepId": "step-1"}},
                ]
            )
        self.assertIn("stepDetails", str(caught.exception))


class TestRunActionStep(unittest.TestCase):
    """The step that fetches what one action needs and hands it to the session runner."""

    def _run(self, action_dict, *, job_details=None, env_layers=None):
        worker = mock.MagicMock()
        worker.get_job_entities.return_value = {
            "jobDetails": job_details
            if job_details is not None
            else {"jobId": JOB_ID, "parameters": {}},
            "stepDetails": {"template": {"name": "Generate"}},
            "environmentDetails": {"template": {"name": "Config"}},
        }
        worker.assume_queue_role.return_value = {
            "accessKeyId": "AKIAQUEUE",
            "secretAccessKey": "s",
            "sessionToken": "t",
            "expiration": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
        with stub_session_runner() as runner, mock.patch.object(
            durable_worker, "DeadlineWorker", return_value=worker
        ):
            outcome = durable_worker.run_action(
                WORKER_ID, "session-1", QUEUE_ID, JOB_ID, action_dict, env_layers or []
            )
        return outcome, runner, worker

    def test_job_details_and_the_step_template_are_asked_for_together(self):
        _, _, worker = self._run(task_run_action())
        identifiers = worker.get_job_entities.call_args.kwargs["identifiers"]
        self.assertEqual(len(identifiers), 1 + 1)
        self.assertEqual(
            identifiers,
            [
                {"jobDetails": {"jobId": JOB_ID}},
                {"stepDetails": {"jobId": JOB_ID, "stepId": "step-1"}},
            ],
        )

    def test_an_environment_action_asks_for_the_environment_template(self):
        _, _, worker = self._run(harness.env_enter_action(environment_id="env-7"))
        identifiers = worker.get_job_entities.call_args.kwargs["identifiers"]
        self.assertEqual(
            identifiers[1], {"environmentDetails": {"jobId": JOB_ID, "environmentId": "env-7"}}
        )

    def test_job_parameters_reach_the_session_unwrapped(self):
        _, runner, _ = self._run(
            task_run_action(),
            job_details={"jobId": JOB_ID, "parameters": {"ModelId": {"string": "luma.ray-v2:0"}}},
        )
        self.assertEqual(
            runner.calls[0]["job_parameters"],
            {"ModelId": {"type": "STRING", "value": "luma.ray-v2:0"}},
        )

    def test_task_parameters_reach_the_session_unwrapped(self):
        _, runner, _ = self._run(task_run_action())
        self.assertEqual(
            runner.calls[0]["task_parameters"], {"Prompt": {"type": "STRING", "value": "a car"}}
        )

    def test_path_mapping_rules_are_passed_through_rather_than_reimplemented(self):
        rules = [
            {"sourcePathFormat": "windows", "sourcePath": "Z:\\", "destinationPath": "/mnt/z"}
        ]
        _, runner, _ = self._run(
            task_run_action(), job_details={"jobId": JOB_ID, "pathMappingRules": rules}
        )
        self.assertEqual(runner.calls[0]["path_mapping_rules"], rules)

    def test_the_queue_role_is_assumed_when_the_job_names_one(self):
        _, runner, worker = self._run(
            task_run_action(),
            job_details={"jobId": JOB_ID, "queueRoleArn": "arn:aws:iam::1:role/QueueRole"},
        )
        worker.assume_queue_role.assert_called_once_with(queue_id=QUEUE_ID)
        # This is what keeps a task's script off the worker's own identity.
        self.assertEqual(runner.calls[0]["os_env_vars"]["AWS_ACCESS_KEY_ID"], "AKIAQUEUE")

    def test_without_a_queue_role_the_task_gets_no_credentials_at_all(self):
        _, runner, worker = self._run(task_run_action(), job_details={"jobId": JOB_ID})
        worker.assume_queue_role.assert_not_called()
        # Inherited credentials would be the worker's own, so they are removed instead.
        self.assertIsNone(runner.calls[0]["os_env_vars"]["AWS_ACCESS_KEY_ID"])

    def test_a_protocol_failure_fails_one_action_rather_than_the_execution(self):
        worker = mock.MagicMock()
        worker.get_job_entities.side_effect = worker_protocol.WorkerProtocolError("no such job")
        with stub_session_runner(), mock.patch.object(
            durable_worker, "DeadlineWorker", return_value=worker
        ), self.assertLogs("durable-step", level="ERROR"):
            outcome = durable_worker.run_action(
                WORKER_ID, "session-1", QUEUE_ID, JOB_ID, task_run_action(), []
            )
        self.assertEqual(outcome["state"], "FAILED")
        self.assertIn("no such job", outcome["message"])
        self.assertEqual(outcome["awaitTokens"], [])

    def test_an_unexpected_defect_fails_one_action_rather_than_the_execution(self):
        worker = mock.MagicMock()
        worker.get_job_entities.side_effect = ZeroDivisionError("boom")
        with stub_session_runner(), mock.patch.object(
            durable_worker, "DeadlineWorker", return_value=worker
        ), self.assertLogs("durable-step", level="ERROR"):
            outcome = durable_worker.run_action(
                WORKER_ID, "session-1", QUEUE_ID, JOB_ID, task_run_action(), []
            )
        self.assertEqual(outcome["state"], "FAILED")
        self.assertIn("ZeroDivisionError", outcome["message"])

    def test_every_outcome_carries_the_ended_at_the_service_requires(self):
        outcome, _, _ = self._run(task_run_action())
        self.assertTrue(outcome["endedAt"])


class TestTimestampSerialization(unittest.TestCase):
    def test_botocore_accepts_iso_strings_for_timestamp_members(self):
        client = _deadline_client()
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
        # Patch before setting credentials: the refresh callback is bound when the credentials
        # are created, so patching afterwards would hit the network.
        with mock.patch.object(
            worker, "_fetch_fleet_role_credentials", return_value=renewed
        ) as fetch:
            worker.set_credentials(
                self._api_credentials(datetime.now(timezone.utc) + timedelta(seconds=1))
            )
            frozen = worker._worker_credentials.get_frozen_credentials()
        fetch.assert_called()
        self.assertEqual(frozen.access_key, "AKIARENEWED")


class TestQueueRoleCredentials(unittest.TestCase):
    """The credentials a job's own scripts run with, which are not the worker's."""

    def test_assume_queue_role_for_worker_exists_in_the_installed_model(self):
        # The whole credentials design rests on this operation being available.
        operations = _deadline_client().meta.service_model.operation_names
        self.assertIn("AssumeQueueRoleForWorker", operations)

    def test_the_queue_id_is_what_scopes_the_request(self):
        client = _deadline_client()
        stubber = Stubber(client)
        stubber.add_response(
            "assume_queue_role_for_worker",
            {
                "credentials": {
                    "accessKeyId": "AKIAQUEUE",
                    "secretAccessKey": "s",
                    "sessionToken": "t",
                    "expiration": datetime(2026, 1, 1, tzinfo=timezone.utc),
                }
            },
            {
                "farmId": harness.FARM_ID,
                "fleetId": harness.FLEET_ID,
                "workerId": WORKER_ID,
                "queueId": QUEUE_ID,
            },
        )
        stubber.activate()
        worker = worker_protocol.DeadlineWorker(
            farm_id=harness.FARM_ID,
            fleet_id=harness.FLEET_ID,
            region="us-west-2",
            worker_id=WORKER_ID,
        )
        worker._client_cache[True] = client
        worker._worker_credentials = object()

        credentials = worker.assume_queue_role(queue_id=QUEUE_ID)
        stubber.assert_no_pending_responses()
        self.assertEqual(credentials["accessKeyId"], "AKIAQUEUE")


if __name__ == "__main__":
    unittest.main()
