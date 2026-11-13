# Isaac Sim Replicator (Conda)

This job bundle generates randomized, annotated synthetic images with
[NVIDIA Isaac Sim Replicator](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/index.html)
on a Linux GPU service-managed fleet. Isaac Sim comes from the `isaacsim` conda package, so workers
need no container runtime, image registry, or host configuration script.

## What this sample demonstrates

* Running Isaac Sim headlessly from the default Conda queue environment, using the package built
  by the [Isaac Sim 6.2 conda recipe](../../conda_recipes/isaacsim-6.2/).
* Splitting frames across tasks with `CHUNK[INT]` task chunking, so each task starts Isaac Sim once
  for several frames.
* Adding a USD asset from the Isaac Sim asset library over HTTPS.

Each task writes one directory of frames. BasicWriter numbers the files in each directory from
`0000`, so `frames_0008-0015/rgb_0000.png` is frame 8.

```text
output/
├── frames_0000-0007/
│   ├── rgb_0000.png
│   ├── bounding_box_2d_tight_0000.npy
│   ├── bounding_box_2d_tight_labels_0000.json
│   ├── semantic_segmentation_0000.png
│   ├── semantic_segmentation_labels_0000.json
│   └── ...
└── frames_0008-0015/
    └── ...
```

## Prerequisites

* A queue with the default Conda queue environment and a Linux GPU service-managed fleet. The job
  requires one GPU per task.
* The `isaacsim` package built into an S3 conda channel that the queue role can read. See
  [Build the package](../../conda_recipes/isaacsim-6.2/README.md#build-the-package).
* The [Deadline Cloud CLI](https://docs.aws.amazon.com/deadline-cloud/latest/developerguide/submit-jobs-how.html).

## Submit the job

Replace the bucket in the channel URL with the job attachments bucket of your package build queue:

```console
$ deadline bundle submit job_bundles/isaacsim_replicator_conda \
    -p CondaChannels="s3://amzn-s3-demo-bucket/Conda/Default conda-forge"
```

Download the frames after the job finishes:

```console
$ deadline job download-output
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `Frames` | `0-15` | Frame indexes to generate. |
| `ChunkSize` | `8` | Frames per task. Larger chunks spend less time starting Isaac Sim. |
| `Width`, `Height` | `640`, `480` | Image resolution. |
| `Asset` | `Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd` | A USD URL, an absolute path, or a path relative to the Isaac Sim asset root. Leave empty for cubes only. |
| `CondaPackages` | `isaacsim=6.2` | Conda packages for the job. |
| `CondaChannels` | `conda-forge` | Conda channels. Put your S3 channel first. |

## Security, cost, and licensing

Submitting this job accepts the NVIDIA Omniverse License Agreement and the
[NVIDIA Isaac Sim Additional Software and Materials License](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/)
on the workers. The job does this by setting `OMNI_KIT_ACCEPT_EULA=YES`.

Workers download the Isaac Sim asset from NVIDIA's public asset server. GPU instances are billed
while tasks run. Each session installs the roughly 23 GiB package before the first task, and the
first render compiles shaders for about 3 minutes.

## Troubleshooting

* **`openjd_fail: failed to load repodata`** when the Conda environment starts: see
  [the conda recipes troubleshooting section](../../conda_recipes/README.md#openjd_fail-failed-to-load-repodata).
* **Kit crashes in `librtx.scenedb.plugin.so`**: Isaac Sim 5.x does this with the R595 driver that
  the `latest` GPU runtime provides, because it supports only R580. Use the Isaac Sim 6.2 package.
* **Errors about `libomniverse_connection.so` or `GLIBC_2.35`** when loading `omniverse://` URLs:
  Nucleus connections aren't supported on glibc 2.34 workers. Use local paths, HTTP(S) URLs, or the
  default asset root.
