# Repair a Conda channel for service-managed fleets

Make a custom Conda channel readable again by the Conda queue environment on an AWS Deadline Cloud
service-managed fleet, after a recent `rattler-index` or `rattler-build publish` wrote an index the
fleet cannot parse.

Choose this script when your jobs fail as the queue environment starts and the session log ends with:

```text
WARN rattler_repodata_gateway::fetch::with_cache: previous cache state does not contain cache_control header. Assuming out of date...
openjd_fail: failed to load repodata
```

## The channel index, not your package

The channel index, not the packages, is what breaks. `rattler-index` 0.31 and later, and
`rattler-build publish` 0.75 and later, write two things that the Conda queue environment on a
service-managed fleet cannot read:

* An `info.repodata_revisions` field in `repodata.json`, written as a mapping where the fleet expects
  a sequence. The fleet fails to parse `repodata.json` itself.
* A `repodata_shards.msgpack.zst` sharded index, whose embedded channel info has the same field.
  The fleet prefers the sharded index when it is present, so it fails before reading `repodata.json`.

The script removes the field and deletes the sharded index, in every subdirectory of the channel. What
remains is a plain `repodata.json` that both current and older Conda clients read. Packages stay where
they are, and so do the `shards/` objects, which nothing references once the index is gone.

Local package installs succeed against the same channel throughout, because a current `conda`, `pixi`,
or `rattler` reads the newer format. A channel that installs on your workstation and fails on the farm
points at the index rather than at your packages or your permissions.

## Prerequisites

* Python 3.9 or later.
* `boto3`, for a channel in S3: `pip install boto3`.
* Permission to read and write the channel. For a channel in S3, that is `s3:ListBucket`,
  `s3:GetObject`, `s3:PutObject`, and `s3:DeleteObject` on the channel prefix.

## Run

Report what needs repair without changing anything. This exits 1 when a repair is needed, so it also
works as a build pipeline gate:

```console
python repair_conda_channel.py --check s3://my-bucket/Conda/Default
```

Apply the repair:

```console
python repair_conda_channel.py s3://my-bucket/Conda/Default
```

A local channel directory works the same way:

```console
python repair_conda_channel.py ./temp-local-channel
```

Then submit a test job. The queue environment reports `Loading repodata for platforms` followed by
`Loaded repodata in ...` when the channel is readable.

## Parameters and outputs

| Option | Purpose |
|---|---|
| `channel` | The channel to operate on, either an `s3://bucket/prefix` URL or a local directory. |
| `--check` | Report the planned changes, leave the channel untouched, and exit 1 when a repair is needed. |
| `--subdir SUBDIR` | Limit the operation to one subdirectory, such as `linux-64`. Repeatable. The default is every subdirectory that has a `repodata.json`. |
| `--profile` | The AWS named profile to use for a channel in S3. |
| `--region` | The AWS Region of the bucket for a channel in S3. |

The script rewrites `repodata.json`, replaces the `repodata.json.zst` and `repodata.json.bz2`
companions when they exist, and deletes `repodata_shards.msgpack.zst`. It is safe to run repeatedly:
a channel that the fleet already reads reports `Nothing to repair` and exits 0.

The `.zst` companion is rewritten when the Python you run has Zstandard support, which is the standard
library `compression.zstd` module on Python 3.14 and later, or the `zstandard` package. Otherwise the
companion is deleted, because a Conda client prefers a compressed copy and a stale one would keep the
channel broken. Clients then read the uncompressed `repodata.json`.

## Avoid publishing another broken channel

Repair addresses a channel that is already broken. To stop producing broken ones:

* When you index or publish from your own workstation, constrain the tools to
  `rattler-index<0.31` and `rattler-build<0.75`, or run this script after every publish.
* When you build with the [package build job bundles](../../conda_recipes/), the build environment
  already constrains both tools. A worker that already created its `PackageBuildEnv` keeps the newer
  versions it installed, so recreate the environment once with the `ReusePackageBuildEnv` job
  parameter set to `0`.
* Mixing indexers leaves a stale sharded index behind. `conda index` writes `repodata.json` and
  ignores `repodata_shards.msgpack.zst`, so a channel previously indexed by `rattler-index` keeps
  serving the older package list to any client that reads the sharded index. Running this script after
  `conda index` removes that index and resolves the mismatch.

Raise or remove the version constraints once service-managed fleets read the newer index format.

## Related resources

* [Conda recipes and package build jobs](../../conda_recipes/) for the recipes and job bundles that
  build packages into a channel.
* [Conda queue environments](../../queue_environments/) for how a queue provides Conda packages to a
  session.
* [Provide applications with Conda in AWS Deadline Cloud](https://docs.aws.amazon.com/deadline-cloud/latest/developerguide/conda-queue-environments.html)
