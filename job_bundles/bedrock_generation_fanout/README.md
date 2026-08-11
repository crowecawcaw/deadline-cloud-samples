# Bedrock generation fanout

Fans out a list of prompts as one Amazon Bedrock generation request per task. Each task
describes an API call rather than a command to run, which is what lets a sleeping worker
execute it.

This bundle is the companion job for the
[customer-managed fleet with Lambda durable function workers](../../cloudformation/farm_templates/cmf_lambda_durable_worker/)
sample, and it expects that fleet to be deployed and associated with your queue.

## How it differs from a normal job bundle

Two things are unusual:

* **The step requires a custom capability.** `hostRequirements` asks for
  `attr.durable.lambda`, so these tasks only land on durable Lambda workers. The fleet
  must declare the same attribute; if it does not, tasks are reported `NOT_COMPATIBLE`
  and never run.
* **The `onRun` command is not what does the work.** A durable Lambda worker reads the
  task parameters and calls Bedrock itself. The embedded script only echoes what was
  requested, which keeps the template valid, runnable under `openjd run`, and useful for
  comparison on a conventional worker.

## Parameters

Everything is a **task** parameter, set in the step's `parameterSpace`:

| Parameter | Value | Purpose |
|---|---|---|
| `Prompt` | three prompts | One task per entry. This is what fans out. |
| `ModelId` | `luma.ray-v2:0` | Bedrock model to invoke. Must support `StartAsyncInvoke`. |
| `Duration` | `5s` | Length of each generated clip |
| `Resolution` | `540p` | Output resolution; higher takes longer |

Edit the `Prompt` range to change how many concurrent requests the fleet is asked to
make; the number of queued tasks is what drives scale-out. The other three are
single-valued on purpose: task parameters form a cross product, so giving `Resolution`
two values would double the task count rather than change a setting.

There are deliberately **no job parameters**. A worker receives task parameters directly
in the `taskRun` session action, but job parameters must be fetched with
`BatchGetJobEntity`, which this sample does not implement. A job parameter would
therefore never reach the worker and would silently fall back to the function's default,
which is worth knowing before adding one.

Image models such as Amazon Nova Canvas will not work here: they are synchronous-only
through `InvokeModel` and expose no asynchronous invocation to poll.

## Submit

```console
deadline bundle submit . --farm-id farm-<your-farm-id> --queue-id queue-<your-queue-id>
```

Check the template, or run a single task locally to see what a task requests:

```console
openjd check template.yaml
openjd run template.yaml --step Generate \
  -tp Prompt="a slow aerial push over a misty pine forest at dawn" \
  -tp ModelId="luma.ray-v2:0" -tp Duration="5s" -tp Resolution="540p"
```

Generated files are written by Bedrock to the output bucket created by the fleet stack,
under `generated/<taskId>/<invocationId>/`.
