# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Integration tests that run real Open Job Description session actions.

Everything else in this suite stubs the session seam. These drive `openjd-sessions` for real,
which is the only way to hold the parts that are agreements with a library rather than with
this code: that `user=None` needs no privileges, that a bare `stepDetails.template` is runnable
without a job template, that the specified stdout protocols actually reach the worker, and that
an environment's variables survive being carried between two separate sessions.

They are skipped when `openjd-sessions` is not importable, so the suite still passes without it.
Install it with `pip install openjd-sessions` to run them. No credentials and no network.

Run from the parent directory with:

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from harness import JOB_ID, QUEUE_ID, WORKER_ID, FakeDurableContext, task_run_action

import action_output

try:
    import session_runner

    OPENJD_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the skip itself
    OPENJD_AVAILABLE = False

import durable_worker

def step_template(body: str, *, task_parameters: list | None = None) -> dict:
    """A step template whose onRun runs `body`, in the shape BatchGetJobEntity returns."""
    template = {
        "name": "Generate",
        "script": {
            "actions": {"onRun": {"command": "{{Task.File.Run}}"}},
            "embeddedFiles": [
                {
                    "name": "Run",
                    "type": "TEXT",
                    "runnable": True,
                    "data": "#!/bin/bash\nset -e\n" + body,
                }
            ],
        },
    }
    if task_parameters:
        template["parameterSpace"] = {"taskParameterDefinitions": task_parameters}
    return template


def environment_template(*, on_enter: str | None = None, on_exit: str | None = None, variables=None):
    """An environment template in the shape BatchGetJobEntity returns."""
    definition: dict = {"name": "Config"}
    if variables:
        definition["variables"] = variables
    actions: dict = {}
    embedded = []
    if on_enter is not None:
        actions["onEnter"] = {"command": "{{Env.File.Enter}}"}
        embedded.append(
            {"name": "Enter", "type": "TEXT", "runnable": True, "data": "#!/bin/bash\nset -e\n" + on_enter}
        )
    if on_exit is not None:
        actions["onExit"] = {"command": "{{Env.File.Exit}}"}
        embedded.append(
            {"name": "Exit", "type": "TEXT", "runnable": True, "data": "#!/bin/bash\nset -e\n" + on_exit}
        )
    if actions:
        definition["script"] = {"actions": actions, "embeddedFiles": embedded}
    return definition


@unittest.skipUnless(OPENJD_AVAILABLE, "openjd-sessions is not installed")
class _SessionTestCase(unittest.TestCase):
    """Runs each action under its own session root, so nothing leaks between tests."""

    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.session_root = Path(root.name)
        patcher = mock.patch.object(session_runner, "SESSION_ROOT", self.session_root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_task(self, template, *, task_parameters=None, job_parameters=None, os_env_vars=None):
        return session_runner.run_action(
            kind="taskRun",
            session_id="session-1",
            template=template,
            job_parameters=job_parameters or {},
            task_parameters=task_parameters or {},
            os_env_vars=os_env_vars if os_env_vars is not None else {},
        )


class TestRunningARealTask(_SessionTestCase):
    def test_a_bare_step_template_runs_with_no_job_template_at_all(self):
        # This is what makes the worker possible: the service hands over one step's template.
        outcome = self.run_task(step_template("echo hello"))
        self.assertEqual(outcome["state"], "SUCCESS")
        self.assertEqual(outcome["exitCode"], 0)

    def test_a_nonzero_exit_is_a_failed_action(self):
        outcome = self.run_task(step_template("exit 3"))
        self.assertEqual(outcome["state"], "FAILED")
        self.assertEqual(outcome["exitCode"], 3)

    def test_the_library_needs_no_privileges_to_do_any_of_this(self):
        # user=None short-circuits every chown, sudo, and setsid path, which is the whole
        # reason openjd-sessions works inside Lambda.
        outcome = self.run_task(step_template("test -w \"$OPENJD_SESSION_WORKING_DIR\""))
        self.assertEqual(outcome["state"], "SUCCESS")

    def test_the_working_directory_is_deleted_when_the_action_ends(self):
        # Nothing may be left behind: /tmp is small and a surviving directory belongs to a
        # session that will never resume.
        self.run_task(step_template("echo hello"))
        self.assertEqual(list(self.session_root.iterdir()), [])

    def test_an_invalid_template_is_reported_rather_than_raised_as_a_defect(self):
        with self.assertRaises(session_runner.SessionRunnerError) as caught:
            self.run_task({"name": "Generate", "script": {"actions": {}}})
        self.assertIn("not valid", str(caught.exception))


class TestParametersReachTheScript(_SessionTestCase):
    def test_a_job_parameter_is_interpolated_by_the_library(self):
        # Job parameters never reached this worker before, so this is the new capability.
        outcome = self.run_task(
            step_template("echo model={{Param.ModelId}}"),
            job_parameters={"ModelId": {"type": "STRING", "value": "luma.ray-v2:0"}},
        )
        self.assertEqual(outcome["state"], "SUCCESS")

    def test_a_task_parameter_is_interpolated_by_the_library(self):
        outcome = self.run_task(
            step_template(
                'test "{{Task.Param.Prompt}}" = "a red car"',
                task_parameters=[{"name": "Prompt", "type": "STRING", "range": ["a red car"]}],
            ),
            task_parameters={"Prompt": {"type": "STRING", "value": "a red car"}},
        )
        self.assertEqual(outcome["state"], "SUCCESS")

    def test_a_chunked_integer_is_refused_as_a_job_parameter(self):
        with self.assertRaises(session_runner.SessionRunnerError):
            self.run_task(
                step_template("true"),
                job_parameters={"Frames": {"type": "CHUNK[INT]", "value": "1-10"}},
            )

    def test_the_base_environment_reaches_the_script(self):
        outcome = self.run_task(
            step_template('test "$AWS_REGION" = "us-west-2"'),
            os_env_vars={"AWS_REGION": "us-west-2", "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(outcome["state"], "SUCCESS")

    def test_a_none_value_removes_an_inherited_variable(self):
        # The subprocess starts from a copy of the worker's own environment, so this is the only
        # way an unset actually unsets.
        with mock.patch.dict("os.environ", {"AWS_SECRET_ACCESS_KEY": "worker-secret"}):
            outcome = self.run_task(
                step_template('test -z "${AWS_SECRET_ACCESS_KEY:-}"'),
                os_env_vars={"AWS_SECRET_ACCESS_KEY": None, "PATH": "/usr/bin:/bin"},
            )
        self.assertEqual(outcome["state"], "SUCCESS")


class TestAwaitTokenHarvestingForReal(_SessionTestCase):
    def test_a_token_printed_on_stdout_reaches_the_worker(self):
        # Single closing brace on purpose: `}}` would end an interpolation expression.
        outcome = self.run_task(
            step_template(
                "echo 'durable_lambda_await: "
                '{"provider": "bedrock-async", "handle": "arn:invoke/1"}\''
            )
        )
        self.assertEqual(outcome["state"], "SUCCESS")
        self.assertEqual(
            outcome["awaitTokens"], [{"provider": "bedrock-async", "handle": "arn:invoke/1"}]
        )

    def test_no_token_is_the_ordinary_case(self):
        outcome = self.run_task(step_template("echo just working"))
        self.assertEqual(outcome["awaitTokens"], [])

    def test_two_tokens_are_both_harvested_for_the_caller_to_reject(self):
        outcome = self.run_task(
            step_template(
                "echo 'durable_lambda_await: {\"provider\": \"sleep\", \"handle\": 1}'\n"
                "echo 'durable_lambda_await: {\"provider\": \"sleep\", \"handle\": 2}'"
            )
        )
        self.assertEqual(len(outcome["awaitTokens"]), 2)

    def test_a_failed_action_has_its_output_ignored(self):
        outcome = self.run_task(
            step_template(
                "echo 'durable_lambda_await: {\"provider\": \"sleep\", \"handle\": 1}'\nexit 1"
            )
        )
        self.assertEqual(outcome["state"], "FAILED")
        self.assertEqual(outcome["awaitTokens"], [])

    def test_a_handle_a_shell_can_produce_with_date_is_enough(self):
        # The sleep provider exists to prove a template needs no SDK to use the await path.
        outcome = self.run_task(
            step_template(
                'echo "durable_lambda_await: '
                '{\\"provider\\": \\"sleep\\", \\"handle\\": '
                '{\\"finishAt\\": $(date +%s)} }"'
            )
        )
        self.assertEqual(outcome["state"], "SUCCESS")
        self.assertEqual(outcome["awaitTokens"][0]["provider"], "sleep")
        self.assertIn("finishAt", outcome["awaitTokens"][0]["handle"])

    def test_an_openjd_fail_message_becomes_the_actions_message(self):
        outcome = self.run_task(
            step_template("echo 'openjd_fail: StartAsyncInvoke was denied'\nexit 1")
        )
        self.assertEqual(outcome["state"], "FAILED")
        self.assertIn("denied", outcome["message"])


class TestEnvironmentsForReal(_SessionTestCase):
    def enter(self, template, *, job_parameters=None, os_env_vars=None):
        return session_runner.run_action(
            kind="envEnter",
            session_id="session-enter",
            template=template,
            job_parameters=job_parameters or {},
            os_env_vars=os_env_vars if os_env_vars is not None else {},
            environment_id="env-1",
        )

    def exit(self, template, *, os_env_vars):
        return session_runner.run_action(
            kind="envExit",
            session_id="session-exit",
            template=template,
            job_parameters={},
            os_env_vars=os_env_vars,
            environment_id="env-1",
        )

    def test_an_on_enter_script_actually_runs_now(self):
        # It used to be refused outright, which failed every scripted queue environment.
        template = environment_template(on_enter="echo 'openjd_env: CONDA_PREFIX=/opt/conda'")
        outcome = self.enter(template)
        self.assertEqual(outcome["state"], "SUCCESS")
        self.assertEqual(outcome["envDelta"]["set"], {"CONDA_PREFIX": "/opt/conda"})

    def test_sets_and_unsets_are_both_harvested_from_stdout(self):
        template = environment_template(
            on_enter="echo 'openjd_env: TOOL=/opt/tool'\necho 'openjd_unset_env: PYTHONHOME'"
        )
        outcome = self.enter(template)
        self.assertEqual(outcome["envDelta"]["set"], {"TOOL": "/opt/tool"})
        self.assertEqual(outcome["envDelta"]["unset"], ["PYTHONHOME"])

    def test_static_variables_are_harvested_and_interpolated(self):
        # The library applies these itself but never reports them, and they have to survive the
        # wait along with everything the script printed.
        template = environment_template(
            on_enter="true", variables={"PLAIN": "value", "DERIVED": "model-{{Param.ModelId}}"}
        )
        outcome = self.enter(
            template, job_parameters={"ModelId": {"type": "STRING", "value": "luma"}}
        )
        self.assertEqual(
            outcome["envDelta"]["set"], {"PLAIN": "value", "DERIVED": "model-luma"}
        )

    def test_a_variable_needing_more_than_job_parameters_fails_comprehensibly(self):
        template = environment_template(
            on_enter="true", variables={"SCRATCH": "{{Session.WorkingDirectory}}/scratch"}
        )
        with self.assertRaises(session_runner.SessionRunnerError) as caught:
            self.enter(template)
        self.assertIn("could not be resolved", str(caught.exception))

    def test_a_variables_only_environment_needs_no_script(self):
        outcome = self.enter(environment_template(variables={"A": "1"}))
        self.assertEqual(outcome["state"], "SUCCESS")
        self.assertEqual(outcome["envDelta"]["set"], {"A": "1"})

    def test_the_nested_authored_shape_is_accepted_too(self):
        # What a reader sees in queue_environments/, even though the service unwraps it.
        outcome = self.enter({"environment": environment_template(variables={"A": "1"})})
        self.assertEqual(outcome["envDelta"]["set"], {"A": "1"})

    def test_a_redacted_variable_is_refused_with_a_reason(self):
        template = environment_template(on_enter="echo 'openjd_redacted_env: SECRET=hunter2'")
        with self.assertRaises(action_output.MalformedOutputError) as caught:
            self.enter(template)
        # The library replaces the value with asterisks before the worker can record it.
        self.assertIn("redacted", str(caught.exception))

    def test_on_exit_runs_in_a_later_session_with_the_variables_on_enter_set(self):
        # The whole point of the layer bookkeeping: two separate sessions, in what would be two
        # separate invocations, and onExit still sees what onEnter exported.
        import session_env

        template = environment_template(
            on_enter="echo 'openjd_env: TOOL=/opt/tool'",
            on_exit='test "$TOOL" = "/opt/tool"',
        )
        entered = self.enter(template)
        composed = session_env.compose({}, [["env-1", entered["envDelta"]]])
        exited = self.exit(template, os_env_vars=composed)
        self.assertEqual(exited["state"], "SUCCESS")

    def test_on_enter_is_not_run_a_second_time_by_the_exit(self):
        template = environment_template(
            on_enter="echo 'openjd_env: RAN=yes'", on_exit="true"
        )
        exited = self.exit(template, os_env_vars={})
        # Its stdout is never seen again, so its side effects cannot repeat either.
        self.assertEqual(exited["envDelta"], {"set": {}, "unset": []})

    def test_an_environment_with_no_on_exit_still_exits_cleanly(self):
        exited = self.exit(
            environment_template(on_enter="echo 'openjd_env: A=1'"), os_env_vars={}
        )
        self.assertEqual(exited["state"], "SUCCESS")

    def test_a_file_written_by_on_enter_is_gone_by_the_next_action(self):
        # Not a bug to fix here: the working directory cannot outlive an invocation. The point
        # is that the failure is a plain missing file rather than something inscrutable.
        template = environment_template(
            on_enter='echo ready > "$OPENJD_SESSION_WORKING_DIR/ready.txt"'
        )
        entered = self.enter(template)
        self.assertEqual(entered["state"], "SUCCESS")
        outcome = self.run_task(step_template('cat "$OPENJD_SESSION_WORKING_DIR/ready.txt"'))
        self.assertEqual(outcome["state"], "FAILED")


class TestTheWholeAwaitPathForReal(_SessionTestCase):
    """A real session action, harvested for real, awaited through the real sleep provider."""

    def test_a_task_hands_over_a_request_and_the_worker_reports_its_outcome(self):
        entities = {
            "jobDetails": {"jobId": JOB_ID, "parameters": {}},
            "stepDetails": {
                "template": step_template(
                    'echo "durable_lambda_await: '
                    '{\\"provider\\": \\"sleep\\", \\"handle\\": '
                    '{\\"finishAt\\": $(date +%s)} }"'
                )
            },
        }
        worker = mock.MagicMock()
        worker.get_job_entities.return_value = entities
        context = FakeDurableContext()

        with mock.patch.object(
            durable_worker, "DeadlineWorker", return_value=worker
        ), mock.patch.object(
            durable_worker,
            "heartbeat",
            lambda worker_id, progress: {
                "workerDeleted": False,
                "desiredWorkerStatus": None,
                "cancelSessionActions": {},
            },
        ):
            result = durable_worker._run_session_action(
                context=context,
                worker_id=WORKER_ID,
                session_id="session-1",
                queue_id=QUEUE_ID,
                job_id=JOB_ID,
                action=task_run_action(parameters={}),
                env_layers=[],
            )

        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        # The wait between the action and the provider poll is the unbilled part.
        self.assertIn(durable_worker.TASK_POLL_SECONDS, context.waits)

    def test_a_task_that_needs_no_await_is_reported_without_one(self):
        entities = {
            "jobDetails": {"jobId": JOB_ID, "parameters": {}},
            "stepDetails": {"template": step_template("echo ordinary openjd task")},
        }
        worker = mock.MagicMock()
        worker.get_job_entities.return_value = entities
        context = FakeDurableContext()

        with mock.patch.object(durable_worker, "DeadlineWorker", return_value=worker):
            result = durable_worker._run_session_action(
                context=context,
                worker_id=WORKER_ID,
                session_id="session-1",
                queue_id=QUEUE_ID,
                job_id=JOB_ID,
                action=task_run_action(parameters={}),
                env_layers=[],
            )

        self.assertEqual(result["completedStatus"], "SUCCEEDED")
        self.assertEqual(context.waits, [])


if __name__ == "__main__":
    unittest.main()
