#!/usr/bin/env bash
# Job environment setup: create (or reuse) the cached Python virtual environment and TRELLIS checkout.
#   setup_env.sh CACHE_DIR SESSION_DIR
# Prints openjd_env lines so that later steps get ML_PY, TRELLIS_SRC, HF_HOME, and TORCH_HOME.
set -euo pipefail

CACHE_DIR=$1
SESSION_DIR=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TRELLIS_COMMIT=442aa1e1afb9014e80681d3bf604e8d728a86ee7
SAM2_SPEC="SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

# Prefer the fleet's persistent volume so environments and model weights survive between sessions.
if [ -z "$CACHE_DIR" ]; then
    PERSISTENT=${DEADLINE_PERSISTENT_MOUNT:-/mnt/persistent}
    if [ -d "$PERSISTENT" ] && [ -w "$PERSISTENT" ]; then
        CACHE_DIR=$PERSISTENT/one_video_to_scene
    else
        CACHE_DIR=$SESSION_DIR/cache
    fi
fi
mkdir -p "$CACHE_DIR"
echo "Cache directory: $CACHE_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "No NVIDIA GPU visible"

export UV_CACHE_DIR=$CACHE_DIR/uv
export UV_PYTHON_INSTALL_DIR=$CACHE_DIR/python
echo "openjd_env: HF_HOME=$CACHE_DIR/huggingface"
echo "openjd_env: TORCH_HOME=$CACHE_DIR/torch"

# Serialize setup between sessions that share the cache.
exec 9>"$CACHE_DIR/.setup.lock"
flock 9

# Virtual environment, keyed by the requirements and this script
HASH=$(cat "$SCRIPT_DIR/requirements.txt" "$SCRIPT_DIR/setup_env.sh" | sha256sum | cut -c1-12)
VENV=$CACHE_DIR/venvs/ml-$HASH
if [ ! -f "$VENV/.complete" ]; then
    echo "openjd_status: Creating Python environment (first run on this worker takes several minutes)"
    rm -rf "$VENV"
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
        -r "$SCRIPT_DIR/requirements.txt"
    # SAM 2 builds against the installed torch. Its optional CUDA extension only fills small mask holes.
    uv pip install --python "$VENV/bin/python" setuptools wheel
    SAM2_BUILD_CUDA=0 uv pip install --python "$VENV/bin/python" --no-build-isolation "$SAM2_SPEC"
    touch "$VENV/.complete"
fi
echo "openjd_env: ML_PY=$VENV/bin/python"

# TRELLIS has no package metadata, so it runs from a pinned source checkout.
TRELLIS_SRC=$CACHE_DIR/src/TRELLIS-$TRELLIS_COMMIT
if [ ! -f "$TRELLIS_SRC/.complete" ]; then
    rm -rf "$TRELLIS_SRC"
    git clone --quiet https://github.com/microsoft/TRELLIS.git "$TRELLIS_SRC"
    git -C "$TRELLIS_SRC" checkout --quiet "$TRELLIS_COMMIT"
    git -C "$TRELLIS_SRC" submodule update --init --recursive --quiet
    touch "$TRELLIS_SRC/.complete"
fi
echo "openjd_env: TRELLIS_SRC=$TRELLIS_SRC"

"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
