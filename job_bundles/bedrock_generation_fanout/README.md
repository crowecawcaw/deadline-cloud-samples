# Bedrock generation fanout

Fans out a list of prompts as one video generation request per task. The step's `onRun`
starts an asynchronous Amazon Bedrock invocation itself, then hands the waiting to the
worker so nothing is billed while the model works.

This bundle is the companion job for the [customer-managed fleet with Lambda durable
function workers](../../cloudformation/farm_templates/cmf_lambda_durable_worker/) sample,
and expects that fleet to be deployed and associated with your queue.

## How it differs from a normal job bundle

* **The step requires a custom capability.** `hostRequirements` asks for
  `attr.durable.lambda`, so the scheduler assigns these tasks only to durable Lambda
  workers. The fleet must declare the same attribute. If it does not, tasks are reported
  `NOT_COMPATIBLE` and never run.
* **`onRun` hands off a wait.** It is an ordinary script that runs for real, and it ends by
  printing one line:

  ```text
  durable_lambda_await: {"provider": "bedrock-async", "handle": "<invocation ARN>"}
  ```

  A durable Lambda worker reads that line, suspends, and polls the invocation until it
  finishes, then reports the task's result from the outcome. Open Job Description defines no
  such thing, so the line is a worker-side extension, and it avoids the reserved `openjd_`
  prefix for that reason. Any other worker ignores it and reports the task successful as soon
  as the script exits.
* **The script needs the queue role.** Credentials come from the queue, not the fleet, so
  the fleet stack's `QueueGenerationPolicyArn` has to be attached to your queue role before
  a task can call Bedrock.

## Parameters

Settings are **job** parameters. The only **task** parameter is `Prompt`, so the parameter
space is one task per prompt.

| Parameter | Kind | Default | Purpose |
|---|---|---|---|
| `OutputBucket` | job | *(required)* | Bucket Bedrock writes to. Use the fleet stack's `OutputBucketName`. |
| `OutputPrefix` | job | `generated` | Key prefix. The fleet stack expires this prefix after seven days. |
| `ModelId` | job | `luma.ray-v2:0` | A model that supports `StartAsyncInvoke`. |
| `Duration` | job | `5s` | Clip length, spelled as the model documents. |
| `Resolution` | job | `540p` | Output resolution, spelled as the model documents. |
| `AspectRatio` | job | `16:9` | Output aspect ratio, spelled as the model documents. |
| `Prompt` | task | three prompts | One task per entry. This range is what fans out. |

Add or remove entries in the `Prompt` range to change how many concurrent requests the fleet
is asked to make. The number of queued tasks is what drives scale-out. Keep `Prompt` the
only task parameter, because task parameters form a cross product: a second one would
multiply the task count rather than change a setting.

Use a model that supports `StartAsyncInvoke`, which on Bedrock means the video models. Image
models such as Amazon Nova Canvas are synchronous-only through `InvokeModel` and expose
nothing to poll.

Output goes to `s3://<OutputBucket>/<OutputPrefix>/<hash of the prompt>/`. The prefix is
derived from the prompt so concurrent tasks cannot collide and a retried task overwrites its
own output.

One authoring constraint worth knowing: `}}` cannot appear anywhere Open Job Description
interpolates, because it reads those two braces as the end of an expression. The embedded
Python builds its nested dictionaries one at a time for that reason.

## Submit

```console
deadline bundle submit . \
  --farm-id farm-<your-farm-id> --queue-id queue-<your-queue-id> \
  -p OutputBucket=<OutputBucketName from the fleet stack>
```

Check the template, or run one task locally:

```console
openjd check template.yaml
openjd run template.yaml --step Generate --maximum-tasks 1 -p OutputBucket=<bucket>
```

`openjd run` proves the template is valid and that interpolation, the embedded files, and
the `StartAsyncInvoke` call path all work. It leaves the wait unproven: the CLI ignores
the await line and reports the task successful the moment the script exits. Without
credentials the script fails cleanly with `openjd_fail:` and a Bedrock error.
