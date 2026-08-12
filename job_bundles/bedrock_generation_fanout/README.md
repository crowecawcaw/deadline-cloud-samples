# Bedrock generation fanout

Fans out a list of video generation requests, one task each. A task here describes an
**API call** rather than a command to run, which is what lets a suspended worker execute
it.

This bundle is the companion job for the [customer-managed fleet with Lambda durable
function workers](../../cloudformation/farm_templates/cmf_lambda_durable_worker/) sample,
and expects that fleet to be deployed and associated with your queue.

## How it differs from a normal job bundle

* **The step requires a custom capability.** `hostRequirements` asks for
  `attr.durable.lambda`, so the scheduler assigns these tasks only to durable Lambda
  workers. The fleet must declare the same attribute. If it does not, tasks are reported
  `NOT_COMPATIBLE` and never run.
* **The `onRun` command is not what calls the service.** A durable Lambda worker never
  runs it. The worker reads the task parameters and calls the service itself. The embedded
  script only echoes the request, which keeps the template valid, runnable under `openjd
  run`, and comparable on a conventional worker.
* **The task carries the request, not a rendering of it.** `Request` is passed to the
  provider verbatim, so the model and every generation setting are template data. The
  worker parses none of it.

## Parameters

Everything is a **task** parameter, set in the step's `parameterSpace`:

| Parameter | Value | Purpose |
|---|---|---|
| `Provider` | `bedrock-async` | Which provider the worker calls. `sleep` is a credential-free fake. |
| `Request` | three JSON requests | The provider's request, passed through unchanged. One task per entry, so this range is what fans out. |

A `Request` for the Bedrock provider is a `modelId` and the `modelInput` that model
expects. Both reach `StartAsyncInvoke` untouched, so `modelInput` follows whatever schema
the chosen model documents:

```json
{"modelInput": {"prompt": "a slow aerial push over a misty pine forest at dawn",
 "duration": "5s", "resolution": "540p"},
 "modelId": "luma.ray-v2:0"}
```

Keep `modelId` last, or otherwise avoid `}}` anywhere in the request: Open Job Description
reads those two braces as the end of an interpolation expression and rejects the template.

Add or remove entries in the `Request` range to change how many concurrent requests the
fleet is asked to make. The number of queued tasks is what drives scale-out. Keep
`Provider` single-valued, because task parameters form a cross product: a second value
would double the task count rather than change a setting.

Use a model that supports `StartAsyncInvoke`, which on Bedrock means the video models.
Image models such as Amazon Nova Canvas are synchronous-only through `InvokeModel` and
expose nothing to poll.

This template deliberately defines **no job parameters**. Task parameters reach the worker
directly in the `taskRun` session action, but job parameters must be fetched with
`BatchGetJobEntity`, which that sample does not implement, so a job parameter would
silently never arrive.

## Submit

```console
deadline bundle submit . --farm-id farm-<your-farm-id> --queue-id queue-<your-queue-id>
```

To exercise the fleet without Bedrock model access, set the `Provider` range to
`["sleep"]` and submit again.

Check the template, or run one task locally to see the request a task carries. `Request`
has an enumerated range, so pick a task rather than passing a value with `-tp`:

```console
openjd check template.yaml
openjd run template.yaml --step Generate --maximum-tasks 1
```

Bedrock writes generated files to the output bucket created by the fleet stack, under
`generated/<taskId>/<invocationId>/`.
