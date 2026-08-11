# Customer-managed fleet with Lambda durable function workers

A Deadline Cloud customer-managed fleet (CMF) whose workers are AWS Lambda durable
functions instead of hosts. Each worker registers with the fleet, heartbeats, and
dispatches long-running Amazon Bedrock generation requests, suspending without compute
charges while those requests run.

Choose this sample when your workers spend their time waiting on another service rather
than computing locally: dispatching API calls, orchestrating model inference, or polling
an external job. If your work needs a persistent filesystem, a GPU, or a DCC
installation, use a conventional [Amazon EC2 customer-managed
fleet](https://docs.aws.amazon.com/deadline-cloud/latest/developerguide/create-auto-scaling.html)
instead.

## What this sample demonstrates

* Implementing the Deadline Cloud **worker protocol directly**, without the
  `deadline-cloud-worker-agent` package.
* Driving CMF capacity from **`EVENT_BASED_AUTO_SCALING`** events, where one worker is
  one durable execution rather than one Amazon EC2 instance.
* Using **durable waits** so a worker that is idle, or blocked on a ten-minute request,
  costs nothing while it waits.
* Graceful **scale-in that never abandons in-flight work**.

## The cost model

The motivation is as much economic as architectural. When the actual work happens in
another service, a conventional worker holds compute it is not using: an Amazon EC2
instance or a service-managed fleet worker is billed for the whole time a job is
assigned to it, including the minutes it spends blocked on someone else's API call. A
durable function inverts that. The worker stays registered and keeps heartbeating, but
suspends between calls and is billed only while it is running code.

What you pay either way:

| | Conventional worker | Durable Lambda worker |
|---|---|---|
| Deadline Cloud CMF worker usage | yes, while the worker exists | yes, while the worker exists |
| Compute | the full worker lifetime | only the seconds spent running code |
| The external service | its own charges | its own charges |

The Deadline Cloud customer-managed fleet worker charge still applies and does not
change: a registered worker is a registered worker. What changes is the compute bill
underneath it, and the external service's charges are unaffected either way.

### A measured example

Three workers generating three clips, from the run used to validate this sample:

* combined worker lifetime: **755 seconds**
* suspended in durable waits: **688 seconds (91%)**
* billed Lambda compute: **67 seconds** across 26 invocations, or **9%** of lifetime

At 512 MB in `us-west-2` that is about 33 GB-seconds, so roughly **$0.0006** in Lambda
charges for the whole run, or about **$0.02 per 100 clips**. Compute is not the
interesting line item at this scale, which is the point: the same three clips on a
conventional worker would have billed an instance for all 755 seconds while it waited.

Part of that 9% is the price of staying correct. A busy worker has to keep heartbeating,
so each generation poll also calls `UpdateWorkerSchedule`; skipping those heartbeats
looks cheaper and gets the worker marked `NOT_RESPONDING` with its task reassigned.

Two caveats keep this honest. Bedrock generation dominates the total bill by orders of
magnitude, so this pattern optimizes the smaller half of the cost. And the saving only
appears for work that genuinely waits on another service; a worker doing local
computation is billed for that computation whether it runs in Lambda or on an instance,
and the instance is likely cheaper.

### Two ways to lose the saving

Reaching 9% took two fixes, both worth knowing if you adapt this sample, because both
quietly turn waiting back into billed compute:

* **Do not let botocore retry in process.** Its backoff sleeps inside the API call, and
  that sleep is billed. A throttled `StartAsyncInvoke` was costing about 11 seconds of
  billed time per attempt. Retrying behind a `context.wait()` instead makes the same
  backoff free, and cut billed compute from 166 seconds to 91 on identical work.
* **Cache your clients.** Building a credentialed `boto3.Session` plus client costs
  roughly 80ms and cannot reuse botocore's warm loader cache, so a step that assumed the
  fleet role and made one call paid it twice per network round trip.

## How it works

```text
 job submitted
      │
      ▼
 Deadline Cloud ──"Fleet Size Recommendation Change"──▶ EventBridge ──▶ scaling function
      ▲                                                                      │
      │                                                    starts one durable execution
      │                                                          per additional worker
      │                                                                      ▼
      │                                                          ┌──────────────────────┐
      └──── CreateWorker / UpdateWorker / UpdateWorkerSchedule ───│   durable worker     │
                                                                 │  (Lambda function)   │
                                                                 └──────────┬───────────┘
                                                                            │
                                                    StartAsyncInvoke ──▶ Amazon Bedrock
                                                            │                   │
                                                     context.wait()       generates to S3
                                                    (suspended, unbilled)       │
                                                            │                   ▼
                                                    GetAsyncInvoke ◀──── output.mp4
```

A worker's whole life is one durable execution. It registers, then loops: call
`UpdateWorkerSchedule` (which is simultaneously the heartbeat, the progress report, and
the request for work), act on whatever comes back, and sleep for the interval the
service asks for. When it receives a task it starts a Bedrock request, sleeps between
status checks, and reports the result on its next heartbeat.

### Reporting results

Two rules govern how results go back, and both are enforced by the service rather than
merely recommended.

**One result per call, in the assigned order.** Reporting action 1 before action 0 is
rejected with "comes in a wrong order", and because the rejection fails the whole request,
batching results loses every result in the batch rather than just the offending one. So
each action's result is sent in its own `UpdateWorkerSchedule` call as soon as it
completes.

**One failure stops the rest of its session.** The service will not run further `taskRun`,
`envEnter`, or `syncInputJobAttachments` actions in a session once any action in it has
failed, been canceled, or been interrupted. The remaining actions are reported
`NEVER_ATTEMPTED`, with no timestamps, because they never started. `envExit` still runs: it
is the session's cleanup and has to happen on the failure path too. Ignoring this rule
means paying Bedrock for requests whose results the service discards.

### Heartbeating while busy

A worker must keep calling `UpdateWorkerSchedule` even while it is working. Stop, and the
service concludes the worker is gone: it marks it `NOT_RESPONDING` and reassigns the
task, so a second worker starts the same Bedrock request while the first is still
running it, and the first worker's eventual result is rejected.

This is easy to get wrong here. The real worker agent runs sessions on a thread pool so
its main loop can keep heartbeating independently, but a durable execution is
single-threaded: a wait for a ten-minute Bedrock request is a wait for everything. So the
generation loop heartbeats on every poll, sending the action's progress with **no**
`completedStatus`, which is how the protocol expresses "still running". That is also
where the worker learns the service has cancelled the action, in which case it stops
polling and reports `CANCELED` instead of paying for a result nobody is waiting for.

### Why not use the worker agent

The [`deadline-cloud-worker-agent`](https://github.com/aws-deadline/deadline-cloud-worker-agent)
assumes a long-lived process on a host it controls: it persists worker IDs to disk,
caches credentials in local files, runs jobs as OS users in local sessions, and streams
logs from background threads. None of that survives a function that suspends and
replays. This sample implements only the five calls a worker actually needs —
`CreateWorker`, `AssumeFleetRoleForWorker`, `UpdateWorker`, `UpdateWorkerSchedule`, and
`DeleteWorker` — in [`lambda/worker_protocol.py`](lambda/worker_protocol.py).

The tradeoff is real: you give up job attachments, session log streaming, host
configuration scripts, and running jobs as a specific user. That is why the job template
here describes an API call rather than a command to execute.

**Job parameters do not reach the worker**, only task parameters do. Job-level parameters
arrive via `BatchGetJobEntity`'s `jobDetails`, which this sample does not fetch, so a job
parameter would silently fall back to the function's default. The accompanying job bundle
therefore puts everything in the step's `parameterSpace`.

### Queue environments

Deadline Cloud wraps a task in `envEnter` and `envExit` actions, one pair per queue
environment. The worker fetches each environment's Open Job Description template with
`BatchGetJobEntity` and then does what it can with it:

* An environment that only defines **`variables`** is applied, and the variables are
  available to the task. This covers environments that pass configuration.
* An environment that defines a **`script`** fails the `envEnter` action with an
  explanation naming the environment.

Failing is deliberate. Every environment in
[`queue_environments/`](../../../queue_environments/) is script-based, because their job
is to install software onto a host: a Lambda worker has no persistent session directory,
no writable filesystem outside `/tmp`, and none of the tools those scripts drive. The
alternative to failing is reporting success for setup that never happened and letting the
task run in an environment that was never prepared.

One operational consequence to know about: a failed `envEnter` is retried, so a queue with
a scripted environment and only these workers will cycle through sessions until the job's
retry limits are exhausted rather than failing once. The message on the failed action says
which environment is responsible. Either remove that environment from the queue, or run
those jobs on a fleet whose workers execute scripts.

**Job attachments are also unsupported**, and fail for the same reason: there is no
session directory to stage input files into. Succeeding would let a task run expecting
inputs that never arrived.

### Writing for replay

Lambda resumes a suspended durable execution by re-running the handler from the top,
substituting stored results for completed steps. Determinism is therefore a correctness
requirement, and the worker follows three rules:

1. Everything non-deterministic or side-effecting happens inside `context.step()`, so
   `CreateWorker` cannot register a second worker on replay.
2. Control flow depends only on step results, never on a clock read or random value at
   the top level of the handler.
3. Timestamps are captured inside steps. `UpdateWorkerSchedule` requires `startedAt` on
   any completed action, and a value read outside a checkpoint would drift each replay.

Credentials are deliberately *not* checkpointed. They are far shorter-lived than a
durable execution, so a replayed copy would usually be expired. Instead the worker wraps
`AssumeFleetRoleForWorker` in botocore's `RefreshableCredentials` and lets botocore
track expiry and re-assume the role while signing, which is the same mechanism the AWS
SDKs use for instance and container credentials.

### Scaling in without losing work

Deadline Cloud emits a recommendation of *how many* workers a CMF should have; it never
picks which ones to stop. Calling `StopDurableExecution` would strand a worker
mid-request, leaving the Bedrock call running, the task unreported, and the worker
registered until the service timed it out.

Instead, the scaling function marks a `drain` flag on chosen workers in a small DynamoDB
registry. Each worker reads its flag on its next heartbeat, finishes any work already
assigned to it, and then exits through `STOPPING` → `STOPPED` → `DeleteWorker`. Newest
workers drain first, since they are least likely to hold a long-running request.

## Prerequisites

* A Deadline Cloud **farm**, and a **queue whose `jobRunAsUser` is set**. A CMF cannot
  be associated with a queue that has no `jobRunAsUser`; creating one with
  `--job-run-as-user '{"runAs":"WORKER_AGENT_USER"}'` is sufficient for this sample.
* **Amazon Bedrock model access** for an asynchronous generation model in your Region,
  granted in the Bedrock console.
* Permission to create IAM roles, Lambda functions, DynamoDB tables, S3 buckets,
  EventBridge rules, SQS queues, and Deadline Cloud fleets.
* The AWS CLI, Python 3, and `zip`.

### Model availability

Asynchronous invocation is what makes a sleeping worker worthwhile, and on Bedrock that
means the **video** generation models. Bedrock's **image** models, including Amazon Nova
Canvas, are synchronous-only through `InvokeModel`: they return an image inline in
seconds and have no async invocation to poll.

Availability varies by Region. In `us-west-2` at the time of writing, `luma.ray-v2:0`
is the available video model; Amazon Nova Reel is not. Check before deploying:

```console
aws bedrock list-foundation-models --region us-west-2 \
  --query "modelSummaries[?contains(outputModalities,'VIDEO')].[modelId]" --output table
```

## Setup

`deploy.sh` packages the Lambda source, bundles the durable execution SDK, uploads the
package, and deploys the stack. CloudFormation cannot upload local source itself, and
the worker function needs its real code before a version is published, so use the
script rather than deploying the template directly.

```console
./deploy.sh --farm-id farm-<your-farm-id>
```

Then associate the fleet with a queue, using the `FleetId` from the stack outputs:

```console
aws deadline create-queue-fleet-association \
  --farm-id farm-<your-farm-id> \
  --queue-id queue-<your-queue-id> \
  --fleet-id fleet-<from-stack-outputs> \
  --region us-west-2
```

Useful options:

```console
./deploy.sh --farm-id farm-xxx --model-id luma.ray-v2:0 --max-workers 10 --region us-west-2
```

## Run or submit

Submit the accompanying job bundle. Each prompt in the template becomes one task, and
the number of queued tasks is what drives the fleet's scale-out recommendation.

```console
deadline bundle submit ../../../job_bundles/bedrock_generation_fanout \
  --farm-id farm-<your-farm-id> --queue-id queue-<your-queue-id>
```

Watch a worker's lifecycle, including its suspensions, in the Lambda console under
**Durable executions**, or from the CLI:

```console
aws lambda list-durable-executions-by-function \
  --function-name <WorkerFunction name> --region us-west-2

aws lambda get-durable-execution-history \
  --durable-execution-arn <arn> --region us-west-2
```

`WaitStarted` and `WaitSucceeded` pairs in the history are the intervals during which
the worker was suspended and unbilled.

## Parameters and outputs

| Parameter | Default | Purpose |
|---|---|---|
| `FarmId` | *(required)* | Farm to create the fleet in |
| `FleetName` | `DurableLambdaFleet` | Fleet display name |
| `MaxWorkerCount` | `5` | Caps concurrent workers, Bedrock concurrency, and cost |
| `ModelId` | `luma.ray-v2:0` | Async-capable Bedrock model to invoke |
| `GenerationPollSeconds` | `30` | Sleep between checks on an in-flight request |

Several further limits are code defaults rather than stack parameters, so changing them
means editing the template's `Environment` block: `SUBMIT_RETRY_SECONDS` (60) and
`MAX_SUBMIT_ATTEMPTS` (10) bound how long a throttled submit waits before retrying;
`MAX_GENERATION_POLLS` (120) bounds how long a request may run; `MAX_IDLE_POLLS` (20,
about five minutes) is how long an idle worker waits before deleting itself, which is
the one to raise for bursty jobs; and `REGISTRY_TTL_SECONDS` (48 hours) expires the
registry row of a worker that died without deregistering. Raising any of the wait
intervals costs nothing, because the waits are suspended.

Stack outputs give the `FleetId`, the worker function alias ARN, the scaling function
name, the output bucket, and the registry table. Generated files land in
`s3://<OutputBucket>/generated/<taskId>/<invocationId>/output.mp4`.

The fleet declares a custom `attr.durable.lambda` capability, and the job template
requires it. Both halves are needed: without the fleet declaring it, tasks are reported
`NOT_COMPATIBLE` and never scheduled.

## Security, cost, and cleanup

Roles follow the split the worker agent uses on Amazon EC2. The function's execution
role holds only `deadline:CreateWorker` and `deadline:AssumeFleetRoleForWorker`;
everything afterward uses the worker-scoped fleet role. Two grants are easy to miss
because the managed policies do not include them: the fleet role needs
`logs:CreateLogStream` (Deadline Cloud creates the worker's log stream during the
transition to `STARTED`, and without it the worker never leaves `CREATED`) and
`deadline:DeleteWorker` (so a drained worker can deregister). Note also that
`CreateWorker`, `ListWorkers`, and `DeleteWorker` authorize against the *worker*
resource, so their ARNs end in `/worker/*`.

Costs are broken down in [The cost model](#the-cost-model) above: Bedrock generation
dominates, with Lambda compute for the brief active periods, Deadline Cloud CMF worker
usage, and DynamoDB and S3 at negligible volume. `MaxWorkerCount` is the concurrency and
cost ceiling. Bedrock's per-account concurrency limits for generation models are low, so
several workers starting at once will hit throttling; the worker retries behind a durable
wait rather than failing the task, which is why the retry is unbilled.

To clean up:

```console
# Stop any running workers first, then delete the stack.
aws deadline delete-queue-fleet-association \
  --farm-id farm-xxx --queue-id queue-xxx --fleet-id fleet-xxx --region us-west-2
aws cloudformation delete-stack --stack-name deadline-durable-lambda-worker --region us-west-2
```

The output bucket is retained deliberately so generated files outlive the stack; delete
it and the artifacts bucket by hand when you no longer need them.

## Files

| File | Purpose |
|---|---|
| [`deadline-durable-lambda-worker.yaml`](deadline-durable-lambda-worker.yaml) | Fleet, functions, registry, scaling rule, and IAM |
| [`deploy.sh`](deploy.sh) | Packages the Lambda source and deploys the stack |
| [`lambda/durable_worker.py`](lambda/durable_worker.py) | The durable worker: registration, heartbeat loop, task execution |
| [`lambda/worker_protocol.py`](lambda/worker_protocol.py) | Deadline Cloud worker protocol client |
| [`lambda/bedrock_task.py`](lambda/bedrock_task.py) | Maps task parameters to async Bedrock requests |
| [`lambda/scaling_handler.py`](lambda/scaling_handler.py) | Turns scaling events into worker executions |
| [`lambda/queue_environment.py`](lambda/queue_environment.py) | Applies a queue environment's variables, refuses its scripts |
| [`lambda/worker_registry.py`](lambda/worker_registry.py) | Live-worker registry and drain flag |
| [`tests/`](tests/) | Unit tests: data transformations, scaling arithmetic, registry, and worker loop exit paths |

Run the tests with:

```console
python3 -m unittest discover -s tests
```

The tests need no credentials and make no AWS calls. They cover the scaling
arithmetic, the drain registry, and the worker loop's exit paths, including a
regression guard that a drain finishes work already assigned to it. Because the
durable execution SDK is stubbed, they verify loop control flow and data handling
rather than checkpoint and replay behavior, which only a live run exercises.
