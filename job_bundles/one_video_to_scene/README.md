# One video to scene

## Introduction

This job bundle turns a single video into an animated, object-level 3D scene. It finds the
objects in the video, generates a mesh for each one, places the meshes in 3D, tracks the ones
that move, and assembles everything in [Blender](https://www.blender.org/) as an animated GLB
with a ground plane, the recovered camera, and rigid body settings. It also renders the scene
from the video's camera and encodes a side-by-side comparison with the input.

The pipeline is inspired by [OVOW (One Video, One World)](https://github.com/yisuanwang/OVOW), but it
uses only models and libraries under permissive licenses (MIT, Apache-2.0, or BSD). It's a proof of
concept, not a reproduction: the results are rough, and deformable objects are treated as rigid.

| Step | What it does | Models and tools |
|---|---|---|
| Analyze | Samples frames, labels objects with boxes, tracks a mask for each object, estimates depth and camera poses in meters, classifies objects as static or moving, and saves an RGBA crop of each object | [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [SAM 2.1](https://github.com/facebookresearch/sam2), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) `DA3-BASE` and `DA3METRIC-LARGE`, FFmpeg |
| Reconstruct | Generates a mesh with vertex colors for each object, one task per object | [TRELLIS](https://github.com/microsoft/TRELLIS) `TRELLIS-image-large` |
| Register | Fits the ground plane, then fits each mesh to the object's depth points with upright ICP, and tracks moving objects frame to frame | Open3D, SciPy, trimesh |
| Assemble | Builds the animated scene and exports `scene.glb` and `scene.blend` | Blender |
| Render | Renders clay and textured passes from the video's camera path, and encodes `compare.mp4` | Blender Cycles, FFmpeg |

### Licenses of the models

All the model weights the job downloads are MIT or Apache-2.0 licensed. Some popular alternatives
were avoided on purpose:

* TRELLIS's own GLB export bakes textures with `nvdiffrast` and a Gaussian rasterizer, which have
  non-commercial licenses. This job uses only the TRELLIS mesh decoder and its vertex colors.
* TRELLIS.2 depends on DINOv3 (custom license) and RMBG-2.0 (CC BY-NC).
* The large Depth Anything 3 checkpoints (`DA3-LARGE`, `DA3-GIANT`, and the nested models) are CC BY-NC.
  `DA3-BASE` and `DA3METRIC-LARGE` are Apache-2.0.
* OVOW itself uses SAM 3 (SAM license), FoundationPose (NVIDIA non-commercial), and Motion324 (CC BY-NC-SA).

Check the licenses yourself before you use this in production.

## Prerequisites

1. A Deadline Cloud farm with a Linux GPU fleet. The GPU needs at least 24 GB of memory, for example
   an NVIDIA L4, A10G, or L40S. A [service-managed fleet](https://docs.aws.amazon.com/deadline-cloud/latest/userguide/smf-manage.html)
   with a persistent volume is ideal, because the job caches its Python environment and about 30 GB of model
   weights there between sessions.
2. A queue with a Conda queue environment like
   [conda_queue_env_improved_caching.yaml](../../queue_environments/conda_queue_env_improved_caching.yaml),
   and the `conda-forge` channel. The job uses `blender`, `ffmpeg`, `uv`, and `git` from conda.
3. The workers need internet access to download Python packages, source code from GitHub, and model
   weights from Hugging Face.
4. On your workstation, the [Deadline Cloud CLI](https://pypi.org/project/deadline/) and Python 3.9+ with
   PyYAML, or [uv](https://docs.astral.sh/uv/) which installs PyYAML for you.

## Run the pipeline

You need a video of a few objects. Static camera or slow camera motion works best. To make a synthetic
test video with no license terms, render [test_data/make_test_video.py](test_data/make_test_video.py) with Blender:

```bash
blender --background --factory-startup --python test_data/make_test_video.py -- /tmp/tabletop
ffmpeg -framerate 24 -i /tmp/tabletop/frame_%04d.png -pix_fmt yuv420p tabletop.mp4
```

Submit the job with `submit.py`:

```bash
uv run submit.py --video tabletop.mp4
```

Pass any job parameter with `-p NAME=VALUE`, for example `-p MaxObjects=4 -p VlmModel=Qwen/Qwen3-VL-4B-Instruct`,
and the farm, queue, and profile with `--farm-id`, `--queue-id`, and `--profile`. You can also submit the bundle
directly with `deadline bundle submit` or `deadline bundle gui-submit`.

When the job finishes, download the output:

```bash
deadline job download-output --job-id <job-id>
```

The output directory has one subdirectory per step:

```
analyze/      frames/, labels.jpg, objects.json, masks/, depth.npz, cameras.json, crops/
reconstruct/  obj_NN/mesh.ply, obj_NN/mesh.glb, obj_NN/input.png
register/     scene.json, report.json, debug/obj_NN_points.ply
assemble/     scene.glb, scene.blend
render/       clay/, textured/, compare.mp4
```

## Re-run from a later stage

When you're iterating on a later stage, you don't need to re-run the earlier ones. Give `submit.py`
the previous job and the stage to start from:

```bash
uv run submit.py --from-job <job-id> --start-at Register
```

`submit.py` then:

1. Reads the previous job's parameters and reuses them. Override any of them with `-p`.
2. Downloads the previous job's outputs for every stage before `--start-at`.
3. Copies them into a new `<output>_prior` directory, which the new job takes as its `PriorOutputDir` input.
   Job attachments deduplicate by content hash, so files that are already in S3 aren't uploaded again.
4. Submits a copy of the bundle with the earlier steps removed, so they don't wait for a worker.

Each step reads its inputs from the job's `OutputDir` when an earlier step of the same job produced
them, and from `PriorOutputDir` otherwise. This works for restarts of restarts too: if the previous job
skipped a stage, `submit.py` takes that stage's outputs from the previous job's `PriorOutputDir`.

The stages are `Analyze`, `Reconstruct`, `Register`, `Assemble`, and `Render`.

## How it works

* **Labeling.** Qwen3-VL lists the movable objects in the first frame as JSON with boxes, which are relative
  coordinates from 0 to 1000. The job drops boxes that cover most of the frame, removes duplicates, and keeps
  the `MaxObjects` largest. Use `ObjectHint` to steer it, for example "only the items on the desk".
* **Tracking.** SAM 2.1 takes the boxes as prompts on the first frame and propagates the masks through the video.
* **Depth and cameras.** `DA3-BASE` estimates depth, intrinsics, and camera poses for all frames together. Its
  scale is arbitrary, so `DA3METRIC-LARGE` runs on a few frames to find the factor that converts it to meters.
* **Motion.** An object whose 3D centroid travels more than a quarter of its size is "rigid" and gets tracked.
  Everything else is "static".
* **Meshes.** Each object's crop comes from the frame where its mask is largest without touching the image edge.
  TRELLIS turns the RGBA crop into a mesh. The mesh is decimated to `--max-faces` and reduced to its largest
  connected piece.
* **Registration.** The ground plane is the dominant plane among background points near the objects. Objects
  are assumed upright. For an object resting on the ground, its scale comes from its height and its bottom stays
  on the ground, leaving its yaw and position to fit. ICP fits the observed depth points to the mesh from many
  starting yaws and positions, and a silhouette check against the masks breaks ties. Static objects use points
  pooled from several frames.

## Known limitations

* Objects must be visible and unoccluded in the first frame to be labeled and tracked.
* Rotationally symmetric shapes, like a mug seen from the side without its handle, can be fitted at the wrong yaw.
* Moving objects are tracked with yaw and translation only, so a rolling ball slides instead of rolling.
* Deformable objects are treated as rigid.
* Generative steps are seeded, but results can still vary across GPUs and library versions.
