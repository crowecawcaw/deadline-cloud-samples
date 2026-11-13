# Isaac Sim 6.2 conda recipe

This recipe packages [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim) 6.2 as a conda
package. Linux GPU workers on Deadline Cloud service-managed fleets (SMF) can then run Isaac Sim
headlessly through the default Conda queue environment, with no container runtime or host
configuration script.

## What the package contains

The package is the `isaacsim[all,extscache]==6.2.0.0` pip distribution from
<https://pypi.nvidia.com>, installed into the package prefix together with its Python dependencies,
including PyTorch and the CUDA libraries it uses. It depends on conda-forge `python 3.12.*`.

* Package content: about 23 GiB in about 240,000 files. The `.conda` archive built with
  `--fast-build` is about 14 GiB, and installing it into a new environment took about 6 minutes on
  an SMF worker.
* Activation sets `ISAACSIM_PATH` and `ISAACSIM_PYTHON_EXE`, the variables that
  [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and other tools use to find Isaac Sim.
* Run Isaac Sim from the environment's `python` with the `isaacsim` module, for example
  `from isaacsim import SimulationApp`, as in the
  [Isaac Sim Replicator (Conda)](../../job_bundles/isaacsim_replicator_conda/) job bundle.

The build runs pip from the build environment with `--prefix` and `--ignore-installed`, so every
pip dependency, including `packaging` and `setuptools`, is installed into the package.
rattler-build leaves out files that were already present in the host environment, so the host
requirements list only `python`.

The pip-installed packages, such as `numpy` and `torch`, belong to this package. Don't combine it in
one environment with conda packages that provide the same Python modules.

## Compatibility with service-managed fleet workers

SMF Linux workers run Amazon Linux 2023, which has glibc 2.34. Isaac Sim 5.0 and later is built on
glibc 2.35, so the build applies two narrow workarounds. Each one fails the build if its assumptions
stop holding.

1. **Wheel selection.** The Isaac Sim wheels are tagged `manylinux_2_35_x86_64`, so pip refuses to
   install them on a glibc 2.34 host. [`pip_glibc235.py`](recipe/pip_glibc235.py) runs pip and
   reports glibc 2.35 for wheel tag selection only.
2. **Symbol versions.** A scan of the roughly 4,300 ELF files in the installation found that only
   these files need a symbol from glibc 2.35:

   | Symbol | Files | Handling |
   |---|---|---|
   | `hypot@GLIBC_2.35` | Kit's `libpython3.12.so`, `_cumotion`, `lula` | Retargeted to `hypot@GLIBC_2.2.5` |
   | `hypotf@GLIBC_2.35` | OpenCV `cv2.abi3.so` | Retargeted to `hypotf@GLIBC_2.2.5` |
   | `_dl_find_object@GLIBC_2.35` | `omni.client.lib/bin/libomniverse_connection.so` | Left unchanged |

   glibc 2.35 only added faster versions of `hypot` and `hypotf` that behave the same. glibc 2.34
   still exports the `GLIBC_2.2.5` versions, so
   [`patch_glibc_versions.py`](recipe/patch_glibc_versions.py) points those references at them.
   `_dl_find_object` has no older equivalent. It is only used by the library for
   Nucleus (`omniverse://`) connections, so local files, HTTP(S) URLs, and the default Isaac Sim
   asset root still work. The patch script fails the build if a binary needs any other glibc 2.35+
   symbol.

The package declares `__glibc >=2.34`.

### NVIDIA driver

Each Isaac Sim release is tested on one NVIDIA driver branch. Isaac Sim 6.1 and 6.2 were tested on
595.58.03 (R595), see the
[Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html).
On service-managed fleets, the `latest` GPU accelerator runtime provides R595.

The package was tested on an SMF fleet with an NVIDIA L4 GPU, the `latest` runtime (driver
595.91.07), and Amazon Linux 2023.12. It hasn't been tested on the other SMF GPU types.

Isaac Sim 5.1 was tested only on R580 (580.65.06), and its RTX renderer crashes in
`librtx.scenedb.plugin.so` on R595, both from a conda package and from NVIDIA's container
([isaac-sim/IsaacSim#619](https://github.com/isaac-sim/IsaacSim/issues/619)). To run Isaac Sim 5.x,
set the fleet's accelerator `runtime` to `grid:r580`. That configuration hasn't been tested with
this recipe.

## Build the package

Follow [Submitting package build jobs](../README.md#submitting-package-build-jobs). The package build
queue needs a Linux x86_64 fleet, and the build took about 30 minutes on an SMF worker. The build holds the
installed files and the package archive on disk at the same time. It was tested on workers with a
250 GiB root volume.

```console
$ ./submit-package-job isaacsim-6.2 --fast-build
```

`--fast-build` is recommended. Most of the package content is already compressed binary data, so
full compression takes much longer without making the package much smaller.

If the build worker has a persistent volume mounted at `/mnt/persistent`, the build reuses the pip
download cache there, so later builds don't download the wheels again.

## Use the package

Set the job's `CondaPackages` parameter to `isaacsim=6.2`, and list your S3 channel first in
`CondaChannels`:

```console
$ deadline bundle submit ../job_bundles/isaacsim_replicator_conda \
    -p CondaChannels="s3://amzn-s3-demo-bucket/Conda/Default conda-forge"
```

Isaac Sim requires accepting the NVIDIA Omniverse License Agreement. Jobs set
`OMNI_KIT_ACCEPT_EULA=YES` to accept it non-interactively. The package does not set this variable.

The first render in a new environment compiles RTX shaders, which took about 3 minutes on an L4.
Kit stores the shader cache inside the environment, so later tasks that reuse the environment
start rendering sooner.

## License

Isaac Sim is distributed under the
[NVIDIA Isaac Sim Additional Software and Materials License](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/),
and the pip dependencies are distributed under their own licenses. Review the terms before you
distribute a built package outside your organization.
