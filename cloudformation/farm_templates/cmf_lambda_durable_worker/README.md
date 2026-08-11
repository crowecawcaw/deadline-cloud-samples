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

In a measured run of three workers over roughly twelve minutes of combined worker
lifetime, the fleet billed about 294 seconds of compute across 47 invocations. Idle
polling and generation waits were not billed.

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

Costs come from Bedrock generation, which dominates, plus Lambda compute for the brief
active periods, DynamoDB and S3 at negligible volume, and Deadline Cloud CMF worker
usage. `MaxWorkerCount` is the concurrency and cost ceiling. Bedrock's per-account
concurrency limits for generation models are low, so several workers starting at once
will hit throttling; the worker treats throttles as retryable and backs off rather than
failing the task.

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
| [`lambda/worker_registry.py`](lambda/worker_registry.py) | Live-worker registry and drain flag |
| [`tests/`](tests/) | Unit tests for the replay-sensitive logic |

Run the tests with:

```console
python3 -m unittest discover -s tests
```
