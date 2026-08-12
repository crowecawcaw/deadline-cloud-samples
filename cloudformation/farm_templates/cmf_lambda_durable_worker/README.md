# Customer-managed fleet with Lambda durable function workers

This sample runs a Deadline Cloud customer-managed fleet (CMF) whose workers are AWS
Lambda **durable functions** instead of hosts. Each worker is one durable execution. It
registers with the fleet, heartbeats, and, once it is given a task, dispatches a
long-running API request and suspends, unbilled, until that request finishes.

It is a proof of concept for using durable Lambda as a fleet so that **Deadline Cloud can
orchestrate workflows whose steps are API calls**, alongside the steps that render. Video
generation is the motivating case. Here the service is Amazon Bedrock, but the worker does
not know that: the job template picks the API to call, so the same fleet can drive an
external service such as Seedance or fal.ai instead.

Choose this pattern when a step spends its time waiting on someone else's service. If the
work needs a filesystem, a GPU, or a DCC installation, use a conventional [Amazon EC2
customer-managed
fleet](https://docs.aws.amazon.com/deadline-cloud/latest/developerguide/create-auto-scaling.html).

> The worker code is illustrative rather than production grade. It implements the protocol
> calls the concept needs and no more, and the [Not supported](#not-supported) list below
> says what that leaves out.

## What this sample demonstrates

* Implementing the Deadline Cloud **worker protocol directly**, without the
  `deadline-cloud-worker-agent` package.
* Driving CMF capacity from **`EVENT_BASED_AUTO_SCALING`** events, where one worker is one
  durable execution rather than one Amazon EC2 instance.
* Using **durable waits**, so a worker that is idle or blocked on a ten-minute request
  costs nothing while it waits.
* Graceful **scale-in that never abandons in-flight work**.
* A **service-agnostic worker**: the task says which provider to call and carries that
  provider's request verbatim.

## Why not a conventional worker

A worker that dispatches an API call and then waits does no local computing, yet a
conventional worker still holds a host for the whole wait: an Amazon EC2 instance or a
service-managed fleet worker is billed for every minute a job is assigned to it, including
the minutes it spends blocked on another service. Because a durable function suspends
between calls, it stays registered and keeps heartbeating while being billed only for the
time it runs code. In a validation run of three workers generating three clips, combined
worker lifetime was 755 seconds, of which 688 (91%) was spent suspended in durable waits.
Billed Lambda compute came to 67 seconds across 26 invocations, roughly $0.02 per 100 clips
at 512 MB. The Deadline Cloud CMF worker charge does not change: it applies for as long as
a worker is registered. Neither does the external service's own bill for generation, which
dominates the total by orders of magnitude. The saving is real only for work that spends
its time waiting, since a worker doing local computation pays for that computation in
Lambda too, and an instance is likely cheaper.

## Not supported

The worker implements the worker protocol but no session runtime, so a good deal of
Deadline Cloud does not apply to it:

* **Job attachments.** The worker has no session directory to stage inputs into, and it
  uploads no outputs. A `syncInputJobAttachments` action fails with an explanation.
* **Queue environments that run scripts.** Only an environment's `variables` are applied.
  A `script` fails the `envEnter` action, which stops the session rather than letting a
  task run in an environment that was never prepared. Every environment in
  [`queue_environments/`](../../../queue_environments/) is script-based, so none of them
  work here.
* **Job parameters.** Only task parameters reach the worker. Job-level parameters arrive
  through `BatchGetJobEntity`'s `jobDetails`, which this worker never fetches, so a job
  parameter silently does nothing. Put everything in the step's `parameterSpace`.
* **Commands.** The step's `script`, `onRun`, and embedded files are never executed. The
  worker reads task parameters and makes an API call. It runs no OpenJD actions.
* **Session log streaming.** Worker output goes to the Lambda function's own log group,
  not to the job's session logs in the Deadline Cloud monitor.
* **Host configuration scripts, `jobRunAsUser`, path mapping, and storage profiles.** The
  worker owns no host, acts as its own Lambda execution role, and sees only an ephemeral
  `/tmp`. A queue still needs a `jobRunAsUser` set to accept a CMF association, but the
  worker does not honor it.
* **More than one action at a time.** A durable execution is single-threaded, so a worker
  runs one session action at a time and cannot host concurrent sessions.
* **Conda, Rez, and any other software delivery.** Nothing is installed; `/tmp` is the
  only writable path and the sandbox is discarded.

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
                                                               │  provider.submit()
                                                               ▼  provider.poll()
                                              Amazon Bedrock, or another service
                                              (suspended, unbilled, between polls)
```

A worker's whole life is one durable execution. It registers, then loops: call
`UpdateWorkerSchedule` (simultaneously the heartbeat, the progress report, and the request
for work), act on whatever comes back, and suspend for the interval the service asks for.
Given a task, it submits the request and then suspends between status polls until it can
report the result.

An EventBridge rule routes the fleet's size recommendations to the scaling function, and
scale-out invokes more executions. Scale-in cannot pick which execution to stop, and
`StopDurableExecution` would strand a worker mid-request, so the scaling function instead
sets a `drain` flag on chosen workers in a small DynamoDB registry. A worker reads its
flag on its next heartbeat, finishes any work already assigned to it, then exits through
`STOPPING` → `STOPPED` → `DeleteWorker`. Newest workers drain first, being least likely
to hold a long-running request.

### Swapping the service

The worker core contains no reference to Bedrock. A task names a provider and carries
that provider's request as an opaque JSON string:

```yaml
- name: Provider
  type: STRING
  range: ["bedrock-async"]
- name: Request
  type: STRING
  range:
  - >-
    {"modelInput": {"prompt": "a red sports car on a coastal highway at sunset",
    "duration": "5s", "resolution": "540p"},
    "modelId": "luma.ray-v2:0"}
```

Keep `modelId` last, or otherwise avoid `}}` anywhere in the request: Open Job Description
reads those two braces as the end of an interpolation expression and rejects the template.

The worker resolves the provider name and hands over the request untouched, then drives a
generic submit-wait-poll loop. Providers implement two functions:

```python
def submit(request: dict, *, task_id: str) -> dict:
    """-> {"handle": ...} | {"error": ..., "retryable": bool}"""

def poll(handle) -> dict:
    """-> {"state": "RUNNING" | "SUCCEEDED" | "FAILED", "message": ..., "outputUri": ...}"""
```

Retry and poll timing are the worker's policy, not the provider's; a provider only says
whether a failure is worth retrying. Adding a service means one new module in
[`lambda/providers/`](lambda/providers/) and one more name in its registry. The registry
starts with `bedrock-async`, and with `sleep`, a credential-free fake that lets you
exercise the whole fleet without Bedrock model access.

Because the model and its input are passed through verbatim, changing model, prompt,
resolution, or duration is a job template edit with no code change.

### Notes for reading the code

These constraints shape the worker and are easy to break while adapting it:

* **One result per `UpdateWorkerSchedule` call, in the order the actions were assigned.**
  Reporting out of order is rejected with "comes in a wrong order", and the rejection
  fails the whole request, so a batch loses every result rather than the offending one.
* **One failure stops the rest of its session.** The service runs no further `taskRun`,
  `envEnter`, or `syncInputJobAttachments` actions once one in the session has not
  succeeded; the rest are reported `NEVER_ATTEMPTED` with no timestamps. `envExit` still
  runs.
* **A busy worker must keep heartbeating.** The real worker agent heartbeats from a
  separate thread; a durable execution is single-threaded, so the poll loop heartbeats
  itself with progress and no `completedStatus`. Stop, and the service marks the worker
  `NOT_RESPONDING` and reassigns the task to someone else. That same response is where
  cancellation is observed.
* **Replay demands determinism.** Lambda resumes a suspended execution by re-running the
  handler and substituting stored results, and it matches checkpoints to steps *by call
  order*. Side effects and clock reads live inside `context.step()`, control flow depends
  only on step results, and a retry loop varies its step arguments, or it replays the
  first attempt's result forever. Credentials are deliberately not
  checkpointed: `AssumeFleetRoleForWorker` is wrapped in botocore's
  `RefreshableCredentials` so botocore renews them while signing.

## Prerequisites

* A Deadline Cloud **farm**, and a **queue whose `jobRunAsUser` is set**. A CMF cannot be
  associated with a queue without one; `--job-run-as-user '{"runAs":"WORKER_AGENT_USER"}'`
  is enough.
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

* Permission to create IAM roles, Lambda functions, DynamoDB tables, S3 buckets,
  EventBridge rules, SQS queues, and Deadline Cloud fleets.
* The AWS CLI, Python 3, and `zip`.

## Setup

`deploy.sh` builds the deployment package from the Lambda source plus the durable
execution SDK. It stages the package in S3 and then deploys the stack. Use it rather than
deploying the template directly: CloudFormation cannot upload local source, and the worker
function needs its real code before a version is published.

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
./deploy.sh --farm-id farm-xxx --max-workers 10 --region us-west-2
```

## Run or submit

Submit the accompanying job bundle. Each `Request` in the template becomes one task, and
the number of queued tasks is what drives the fleet's scale-out recommendation.

```console
deadline bundle submit ../../../job_bundles/bedrock_generation_fanout \
  --farm-id farm-<your-farm-id> --queue-id queue-<your-queue-id>
```

To watch the fleet work without Bedrock access, change the bundle's `Provider` range to
`["sleep"]` and submit that.

Watch a worker's lifecycle, including its suspensions, under **Durable executions** in
the Lambda console, or from the CLI:

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
| `TaskPollSeconds` | `30` | Sleep between checks on an in-flight request |

The rest of the retry and poll policy sits in the template's `Environment` block rather
than as stack parameters. `SUBMIT_RETRY_SECONDS` (60) and `MAX_SUBMIT_ATTEMPTS` (10) bound
how long a throttled submit waits before retrying, `MAX_TASK_POLLS` (120) bounds how long
one request may run, `MAX_IDLE_POLLS` (20, roughly five minutes) is how long an idle worker
waits before deleting itself and is the one to raise for bursty jobs, and
`MAX_LOOP_ITERATIONS` (2000) caps a single worker's schedule polls. Raising a wait interval
costs nothing, because the waits are suspended.

`REGISTRY_TTL_SECONDS` (48 hours) and the Bedrock provider's `OUTPUT_PREFIX`
(`generated`) are module defaults with no entry in the template. The first expires the
registry row of a worker that died without deregistering. The second has to match the
output bucket's lifecycle rule if you change it.

Stack outputs give the `FleetId`, the worker function alias ARN, the scaling function
name, the output bucket, and the registry table. Bedrock writes generated files to
`s3://<OutputBucket>/generated/<taskId>/<invocationId>/output.mp4`.

The fleet declares a custom `attr.durable.lambda` capability and the job template
requires it. Both halves are needed: without the fleet declaring it, tasks are reported
`NOT_COMPATIBLE` and never scheduled.

## Security, cost, and cleanup

Roles follow the split the worker agent uses on Amazon EC2. The function's execution role
holds only `deadline:CreateWorker` and `deadline:AssumeFleetRoleForWorker`; everything
afterward uses the worker-scoped fleet role. The managed policies leave out grants that the
fleet role still needs, which is easy to miss: `logs:CreateLogStream` (Deadline Cloud
creates the worker's log stream during the transition to `STARTED`, and without it the
worker never leaves `CREATED`), `deadline:DeleteWorker` so a drained worker can deregister,
and `deadline:BatchGetJobEntity` to read queue environments. Note that `CreateWorker`,
`ListWorkers`, and `DeleteWorker` authorize against the *worker* resource, so their ARNs
end in `/worker/*`.

On cost, see [Why not a conventional worker](#why-not-a-conventional-worker): generation
dominates, with Lambda compute for the brief active periods, Deadline Cloud CMF worker
usage, and DynamoDB and S3 at negligible volume. `MaxWorkerCount` is the concurrency and
cost ceiling. Bedrock's per-account concurrency limits for generation models are low, so
concurrent workers will be throttled. The worker then retries behind a durable wait, which
is unbilled, rather than failing the task.

To clean up:

```console
# Stop any running workers first, then delete the stack.
aws deadline delete-queue-fleet-association \
  --farm-id farm-xxx --queue-id queue-xxx --fleet-id fleet-xxx --region us-west-2
aws cloudformation delete-stack --stack-name deadline-durable-lambda-worker --region us-west-2
```

The output bucket is retained deliberately, so generated files outlive the stack. Delete
it and the artifacts bucket by hand when you no longer need them.

## Files

| File | Purpose |
|---|---|
| [`deadline-durable-lambda-worker.yaml`](deadline-durable-lambda-worker.yaml) | Fleet, functions, registry, scaling rule, and IAM |
| [`deploy.sh`](deploy.sh) | Packages the Lambda source and deploys the stack |
| [`lambda/durable_worker.py`](lambda/durable_worker.py) | Registration, heartbeat loop, and the generic submit-wait-poll driver |
| [`lambda/worker_protocol.py`](lambda/worker_protocol.py) | Deadline Cloud worker protocol client |
| [`lambda/providers/`](lambda/providers/) | Provider registry, the Bedrock provider, and the `sleep` fake |
| [`lambda/scaling_handler.py`](lambda/scaling_handler.py) | Turns scaling events into worker executions |
| [`lambda/queue_environment.py`](lambda/queue_environment.py) | Applies a queue environment's variables, refuses its scripts |
| [`lambda/worker_registry.py`](lambda/worker_registry.py) | Live-worker registry and drain flag |
| [`tests/`](tests/) | Unit tests: provider seam, scaling arithmetic, registry, worker loop exit paths |

```console
python3 -m unittest discover -s tests
```

The tests need no credentials and make no AWS calls. Because the durable execution SDK is
stubbed, they cover loop control flow and data handling rather than checkpoint and replay
behavior, which only a live run exercises.

## Related resources

* [Customer-managed fleets](https://docs.aws.amazon.com/deadline-cloud/latest/userguide/manage-cmf.html)
* [Bedrock generation fanout job bundle](../../../job_bundles/bedrock_generation_fanout/) (the companion job)
* [`deadline-cloud-worker-agent`](https://github.com/aws-deadline/deadline-cloud-worker-agent) (the conventional worker this sample replaces)
