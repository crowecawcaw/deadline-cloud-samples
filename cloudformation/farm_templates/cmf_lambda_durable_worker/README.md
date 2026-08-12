# Customer-managed fleet with Lambda durable function workers

This sample runs a Deadline Cloud customer-managed fleet (CMF) whose workers are AWS
Lambda **durable functions** instead of hosts. Each worker is one durable execution. It
registers with the fleet, heartbeats, runs the session actions it is assigned, and
suspends, unbilled, while a long-running request it started elsewhere finishes.

It is a proof of concept for using durable Lambda as a fleet so that **Deadline Cloud can
orchestrate workflows whose steps are API calls**, alongside the steps that render. Video
generation is the motivating case. Here the service is Amazon Bedrock, but nothing in the
worker knows that: the job template makes the call, so the same fleet can drive an external
service such as Seedance or fal.ai instead.

The worker is a real Open Job Description worker. It executes session actions with
[`openjd-sessions`](https://pypi.org/project/openjd-sessions/), the same library the
[`deadline-cloud-worker-agent`](https://github.com/aws-deadline/deadline-cloud-worker-agent)
uses, so commands, embedded files, parameter interpolation, exit codes, and cancellation
all behave as they do on a conventional worker.

Choose this pattern when a step spends its time waiting on someone else's service. If the
work needs a persistent filesystem, a GPU, or a DCC installation, use a conventional
[Amazon EC2 customer-managed
fleet](https://docs.aws.amazon.com/deadline-cloud/latest/developerguide/create-auto-scaling.html).

> The worker code is illustrative rather than production grade, and the mechanism that
> hands a wait to the worker is a worker-side extension that Open Job Description does not
> define. See [Waiting is off-spec](#waiting-is-off-spec).

## What this sample demonstrates

* Implementing the Deadline Cloud **worker protocol directly**, without the
  `deadline-cloud-worker-agent` package.
* Running **real Open Job Description sessions** inside a Lambda sandbox, with
  `openjd-sessions` driven as an unprivileged single user.
* Driving CMF capacity from **`EVENT_BASED_AUTO_SCALING`** events, where one worker is one
  durable execution rather than one Amazon EC2 instance.
* Using **durable waits**, so a worker blocked on a ten-minute request costs nothing while
  it waits.
* Handing a long wait from a task's own script to the worker, so the **submission is job
  template data** rather than worker code.
* Giving a task the **queue role's** credentials rather than the worker's own.
* Graceful **scale-in that never abandons in-flight work**.

## Why not a conventional worker

A worker that dispatches an API call and then waits does no local computing, yet a
conventional worker still holds a host for the whole wait: an Amazon EC2 instance or a
service-managed fleet worker is billed for every minute a job is assigned to it, including
the minutes it spends blocked on another service. Because a durable function suspends
between calls, it stays registered and keeps heartbeating while being billed only for the
time it runs code. A validation run of three workers generating three clips spent 91% of
its combined 755-second lifetime suspended, and billed 67 seconds of Lambda compute across
26 invocations, on the order of $0.02 per 100 clips. Those figures come from an earlier
revision that submitted requests from worker code, so the exact split differs now that a
real subprocess runs per action, but the shape holds: the waiting dominates and the compute
is a rounding error. The Deadline Cloud CMF worker charge does not change, since it applies
for as long as a worker is registered. Neither does the external service's own bill for
generation, which dominates the total by orders of magnitude. The saving is real only for
work that spends its time waiting, since a worker doing local computation pays for that
computation in Lambda too, and an instance is likely cheaper.

## How it works

```text
 job submitted
      │
      ▼
 Deadline Cloud ──"Fleet Size Recommendation Change"──▶ scaling function
      ▲                                                        │
      │        starts one execution per worker                 │
      │                                                        ▼
      │                                          ┌───────────────────────────┐
      └── CreateWorker / UpdateWorkerSchedule ───┤  durable worker (Lambda)  │
                                                 └─────────────┬─────────────┘
                                                    run_task() │  one action,
                                                               ▼  one invocation
                                                 ┌───────────────────────────┐
                                                 │  the step's own onRun     │
                                                 │  submits, prints a handle │
                                                 └─────────────┬─────────────┘
                                                               │  provider.poll()
                                                               ▼
                                              Amazon Bedrock, or another service
                                              (suspended, unbilled, between polls)
```

A worker's whole life is one durable execution. It registers, then loops: call
`UpdateWorkerSchedule` (simultaneously the heartbeat, the progress report, and the request
for work), act on whatever comes back, and suspend for the interval the service asks for.

**One Lambda invocation runs exactly one session action, start to finish, and never waits
inside it.** That rule is forced rather than chosen. A durable wait ends the invocation, so
a subprocess cannot survive it and neither can the session working directory, which the
durable execution SDK states plainly about Lambda's `/tmp`. Each action builds a fresh
session, runs, reports its result, and lets the sandbox go.

An EventBridge rule routes the fleet's size recommendations to the scaling function, and
scale-out invokes more executions. Scale-in cannot pick which execution to stop, and
`StopDurableExecution` would strand a worker mid-request, so the scaling function instead
sets a `drain` flag on chosen workers in a small DynamoDB registry. A worker reads its flag
on its next heartbeat, finishes any work already assigned to it, then exits through
`STOPPING` → `STOPPED` → `DeleteWorker`. Newest workers drain first, being least likely to
hold a long-running request.

### Handing a wait to the worker

The step's `onRun` runs for real and starts the long-running request itself, with whatever
tool it likes. To hand the waiting over, it prints one line on stdout:

```text
durable_lambda_await: {"provider": "bedrock-async", "handle": "<opaque JSON>"}
```

The worker harvests that line with the same stdout filter it already needs for
`openjd_env`, checkpoints the handle, suspends, and polls until the request reaches a
terminal state. What the handle contains is the provider's business, never the worker's.

| Action outcome | Await lines | Worker reports |
|---|---|---|
| `FAILED`, `CANCELED`, or `TIMEOUT` | any | the mapped status, without parsing output |
| `SUCCEEDED` | none | `SUCCEEDED` at once, as an ordinary Open Job Description task |
| `SUCCEEDED` | one | whatever the provider finally reports |
| `SUCCEEDED` | two or more | `FAILED`, naming the count |

While awaiting, the worker keeps heartbeating with progress and no `completedStatus`, which
is how the protocol says "still running" and is also where it learns the service has
cancelled the action.

Because an ordinary task without an await line reports immediately, **normal Open Job
Description jobs run on this fleet unchanged**.

### Waiting is off-spec

Open Job Description has no concept of an action that suspends. A sweep of the
specification, its RFCs, and its conformance tests finds no async, poll, suspend, or resume
notion, no action beyond `onRun`, `onEnter`, and `onExit`, and no state beyond `RUNNING`,
`CANCELED`, `TIMEOUT`, `FAILED`, and `SUCCESS`. An action `timeout` is a kill deadline that
the specification treats as a failure, and a worker that blocked for the full duration would
be billed for it anyway.

The `durable_lambda_await:` line is an extension this worker invents on its own account. It
deliberately avoids the `openjd_` prefix, which belongs to the specification and could
collide with a future token. An
[RFC-approved](https://github.com/OpenJobDescription/openjd-specifications) `extensions:`
entry is the sanctioned route for such a mechanism, and no existing extension has added an
action or a state.

### Swapping the service

Submitting is the job template's job, so a provider only has to answer "is it done yet":

```python
def poll(handle) -> dict:
    """-> {"state": "RUNNING" | "SUCCEEDED" | "FAILED", "message": ..., "outputUri": ...}"""
```

Adding a service means one new module in [`lambda/providers/`](lambda/providers/) and one
more name in its registry, which imports lazily so resolving one provider never loads
another's SDK. The registry starts with `bedrock-async`, and with `sleep`, a
credential-free fake that polls a `{"finishAt": <epoch>}` handle a template can produce
with `date +%s`. An unrecognized provider name fails one action rather than the execution.

### Credentials

A task's script needs credentials to call the service it is submitting to, and the worker's
own execution role is the wrong answer. `openjd-sessions` starts a subprocess from a copy
of the worker's environment, so the worker explicitly **removes** the credential variables
and puts the **queue role's** credentials in their place, obtained with
`AssumeQueueRoleForWorker`. That is what a conventional worker does, and it means the
permission to call Bedrock belongs to the queue rather than to the fleet.

`AWS_CREDENTIAL_EXPIRATION` has to be replaced along with the keys. Left behind, botocore
would try to refresh the queue credentials using the runtime's own expiry.

### Queue environments

`onEnter` and `onExit` run for real. A `variables` block or an `onEnter` that exports
variables with `openjd_env:` works properly: the worker harvests each environment's delta,
checkpoints it **per environment identifier in entry order**, composes the layers for later
actions, and un-layers one on `envExit`. Merged layers could not be un-layered again.

An `onEnter` that writes files for a later action to read cannot work, because the working
directory does not survive a suspension. That case fails with a missing-file error from the
task itself.

## Not supported

* **Job attachments.** Staging inputs needs a session directory that outlives a suspension,
  and outputs are never uploaded. A `syncInputJobAttachments` action fails with an
  explanation.
* **Anything that needs files to persist between actions.** One invocation is one action
  with its own working directory, so the cross-action scratch space that Open Job
  Description guarantees is absent. Environments that install software depend on it.
* **`openjd_redacted_env:`.** The library replaces the value with `********` before the
  worker can see it, so carrying it across a suspension would set the wrong value. It is
  refused with a message saying so.
* **Session log streaming.** Action output goes to the Lambda function's own log group, not
  to the job's session logs in the Deadline Cloud monitor.
* **`jobRunAsUser`.** Actions run as the single Lambda user. A queue still needs a
  `jobRunAsUser` set to accept a CMF association, but the worker cannot honor it.
* **Open Job Description extensions**, which would need declaring to the session library.

## Prerequisites

* A Deadline Cloud **farm**, and a **queue whose `jobRunAsUser` is set**. A CMF cannot be
  associated with a queue without one, and `--job-run-as-user '{"runAs":"WORKER_AGENT_USER"}'`
  is enough.
* A **queue role**, since a task's script runs with its credentials. The stack emits a
  managed policy for you to attach to it.
* For the Bedrock provider, **model access** for an asynchronous generation model in your
  Region. Asynchronous invocation is what makes a sleeping worker worthwhile, and on
  Bedrock that means the **video** models. Image models such as Amazon Nova Canvas are
  synchronous-only through `InvokeModel` and have nothing to poll. Availability varies. In
  `us-west-2` at the time of writing, `luma.ray-v2:0` is available and Amazon Nova Reel is
  not:

  ```console
  aws bedrock list-foundation-models --region us-west-2 \
    --query "modelSummaries[?contains(outputModalities,'VIDEO')].[modelId]" --output table
  ```

* Permission to create IAM roles and policies, Lambda functions, DynamoDB tables, S3
  buckets, EventBridge rules, SQS queues, and Deadline Cloud fleets.
* The AWS CLI, Python 3, and `zip`.

## Setup

`deploy.sh` builds the deployment package from the Lambda source plus the durable execution
SDK, `openjd-sessions`, and their dependencies. It stages the package in S3 and then
deploys the stack. Use it rather than deploying the template directly: CloudFormation cannot
upload local source, and the worker function needs its real code before a version is
published.

```console
./deploy.sh --farm-id farm-<your-farm-id>
```

Associate the fleet with a queue, using the `FleetId` from the stack outputs:

```console
aws deadline create-queue-fleet-association \
  --farm-id farm-<your-farm-id> \
  --queue-id queue-<your-queue-id> \
  --fleet-id fleet-<from-stack-outputs> \
  --region us-west-2
```

Then attach the stack's `QueueGenerationPolicyArn` to your queue role, so a task's script
can call Bedrock and write its output:

```console
aws iam attach-role-policy \
  --role-name <your-queue-role-name> \
  --policy-arn <QueueGenerationPolicyArn from stack outputs>
```

## Run or submit

Submit the accompanying job bundle. Each prompt becomes one task, and the number of queued
tasks is what drives the fleet's scale-out recommendation.

```console
deadline bundle submit ../../../job_bundles/bedrock_generation_fanout \
  --farm-id farm-<your-farm-id> --queue-id queue-<your-queue-id> \
  -p OutputBucket=<OutputBucketName from stack outputs>
```

Watch a worker's lifecycle, including its suspensions, under **Durable executions** in the
Lambda console, or from the CLI:

```console
aws lambda list-durable-executions-by-function \
  --function-name <WorkerFunction name> --region us-west-2

aws lambda get-durable-execution-history \
  --durable-execution-arn <arn> --region us-west-2
```

`WaitStarted` and `WaitSucceeded` pairs in the history bracket the intervals during which
the worker was suspended and unbilled.

## Parameters and outputs

| Parameter | Default | Purpose |
|---|---|---|
| `FarmId` | *(required)* | Farm to create the fleet in |
| `FleetName` | `DurableLambdaFleet` | Fleet display name |
| `MaxWorkerCount` | `5` | Caps concurrent workers, request concurrency, and cost |
| `TaskPollSeconds` | `30` | Sleep between checks on an awaited request |
| `EphemeralStorageMiB` | `2048` | `/tmp` size, which holds the session working directory |

The worker function runs at **1769 MB**, which is one full vCPU, because it now runs real
subprocesses. The rest of the policy is in the template's `Environment` block rather than in
stack parameters: `ACTION_TIMEOUT_SECONDS` (240) and `CANCEL_GRACE_SECONDS` (20) bound
one action inside the function's 300-second timeout, `MAX_TASK_POLLS` (120) bounds how long
an awaited request may run, `MAX_IDLE_POLLS` (20, roughly five minutes) is how long an idle
worker waits before deleting itself and is the one to raise for bursty jobs, and
`MAX_LOOP_ITERATIONS` (2000) caps a single worker's schedule polls. Raising a wait interval
costs nothing, because the waits are suspended. `REGISTRY_TTL_SECONDS` (48 hours) is a
module default with no entry in the template, and expires the registry row of a worker that
died without deregistering.

Stack outputs give the `FleetId`, the worker function alias ARN, the scaling function name,
the output bucket, the registry table, and `QueueGenerationPolicyArn`.

The fleet declares a custom `attr.durable.lambda` capability and the job template requires
it. Both halves are needed: without the fleet declaring it, tasks are reported
`NOT_COMPATIBLE` and never scheduled.

## Security, cost, and cleanup

Roles follow the split a conventional worker uses on Amazon EC2, with the queue role added.
The function's execution role holds only `deadline:CreateWorker`,
`deadline:AssumeFleetRoleForWorker`, and `bedrock:GetAsyncInvoke` for polling; everything
else uses the worker-scoped fleet role, and a task's own script uses the queue role. The
managed policies leave out grants that the fleet role still needs, which is easy to miss:
`logs:CreateLogStream` (Deadline Cloud creates the worker's log stream during the transition
to `STARTED`, and without it the worker never leaves `CREATED`), `deadline:DeleteWorker` so
a drained worker can deregister, and `deadline:BatchGetJobEntity` to read job, step, and
environment details. Note that `CreateWorker`, `ListWorkers`, and `DeleteWorker` authorize
against the *worker* resource, so their ARNs end in `/worker/*`.

A task's script runs with real AWS credentials and no user isolation, so treat a job
submitted to this fleet as trusted code.

On cost, see [Why not a conventional worker](#why-not-a-conventional-worker): generation
dominates, with Lambda compute for the brief active periods, Deadline Cloud CMF worker
usage, and DynamoDB and S3 at negligible volume. `MaxWorkerCount` is the concurrency and
cost ceiling. Note that a Bedrock throttle now fails the task, because the submission
happens in the template rather than in worker code, and a script that retries in place is
billed for the wait.

To clean up:

```console
# Stop any running workers first, then delete the stack.
aws deadline delete-queue-fleet-association \
  --farm-id farm-xxx --queue-id queue-xxx --fleet-id fleet-xxx --region us-west-2
aws iam detach-role-policy --role-name <queue-role> --policy-arn <QueueGenerationPolicyArn>
aws cloudformation delete-stack --stack-name deadline-durable-lambda-worker --region us-west-2
```

The output bucket is retained deliberately, so generated files outlive the stack. Delete it
and the artifacts bucket by hand when you no longer need them.

## Files

| File | Purpose |
|---|---|
| [`deadline-durable-lambda-worker.yaml`](deadline-durable-lambda-worker.yaml) | Fleet, functions, registry, scaling rule, and IAM |
| [`deploy.sh`](deploy.sh) | Packages the Lambda source and deploys the stack |
| [`lambda/durable_worker.py`](lambda/durable_worker.py) | Registration, heartbeat loop, and the await decision table |
| [`lambda/session_runner.py`](lambda/session_runner.py) | Runs one session action with `openjd-sessions` |
| [`lambda/action_output.py`](lambda/action_output.py) | Harvests the stdout line protocols |
| [`lambda/session_env.py`](lambda/session_env.py) | Base environment and the queue environment layers |
| [`lambda/worker_protocol.py`](lambda/worker_protocol.py) | Deadline Cloud worker protocol client |
| [`lambda/providers/`](lambda/providers/) | Provider registry, the Bedrock poller, and the `sleep` fake |
| [`lambda/scaling_handler.py`](lambda/scaling_handler.py) | Turns scaling events into worker executions |
| [`lambda/worker_registry.py`](lambda/worker_registry.py) | Live-worker registry and drain flag |
| [`tests/`](tests/) | Unit tests, plus real `openjd-sessions` integration tests |

```console
python3 -m unittest discover -s tests
```

The tests need no credentials and make no AWS calls. The integration tests drive
`openjd-sessions` for real and skip themselves when it is not importable, so
`pip install openjd-sessions` is what turns them on. Checkpoint and replay behavior is the
one thing only a live run exercises.

## Related resources

* [Customer-managed fleets](https://docs.aws.amazon.com/deadline-cloud/latest/userguide/manage-cmf.html)
* [Bedrock generation fanout job bundle](../../../job_bundles/bedrock_generation_fanout/) (the companion job)
* [Open Job Description specification](https://github.com/OpenJobDescription/openjd-specifications/wiki)
