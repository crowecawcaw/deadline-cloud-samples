#!/bin/env bash
set -xeuo pipefail

# PIP_EXTRAS comes from context.pip_extras in recipe.yaml.
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
SITE_PACKAGES="$PREFIX/lib/python$PYTHON_VERSION/site-packages"

## Turn off "pip install" customizations from the conda build environment.
#
# This package bundles Isaac Sim with its pip dependencies (including PyTorch), so pip install
# must resolve and install dependencies like normal.
unset PIP_NO_DEPENDENCIES
unset PIP_IGNORE_INSTALLED
unset PIP_NO_INDEX
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Reuse downloaded wheels across builds when the worker has a persistent volume.
if [ -d /mnt/persistent ] && [ -w /mnt/persistent ]; then
    export PIP_CACHE_DIR=/mnt/persistent/pip-cache
else
    export PIP_NO_CACHE_DIR=1
fi

## Install Isaac Sim and all of its pip dependencies into $PREFIX.
#
# rattler-build leaves files that already existed in the host environment out of the package,
# so pip runs from the build environment and --ignore-installed installs every dependency
# (including packaging and setuptools, which Isaac Sim extensions import) into $PREFIX.
echo "openjd_status: pip installing isaacsim[$PIP_EXTRAS]==$PKG_VERSION"
"$BUILD_PREFIX/bin/python" "$RECIPE_DIR/pip_glibc235.py" install \
    --prefix "$PREFIX" \
    --ignore-installed \
    --no-warn-script-location \
    --extra-index-url https://pypi.nvidia.com \
    "isaacsim[$PIP_EXTRAS]==$PKG_VERSION"

# pip wrote the build environment's interpreter into the console scripts. Point them at the
# package's interpreter so rattler-build relocates them with the rest of $PREFIX.
grep -rlI --null "$BUILD_PREFIX/bin/python" "$PREFIX/bin" \
    | xargs -0 --no-run-if-empty sed -i "s|$BUILD_PREFIX/bin/python[0-9.]*|$PREFIX/bin/python|g"
if grep -rlI "$BUILD_PREFIX" "$PREFIX/bin"; then
    echo "ERROR: files in \$PREFIX/bin still reference the build environment"
    exit 1
fi

# Fail the build if a module that Isaac Sim extensions import is missing from the package.
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$PREFIX/bin/python" -c "import packaging, setuptools, numpy, torch; print('imports ok')"

echo "openjd_status: Patching glibc symbol versions"
"$BUILD_PREFIX/bin/python" "$RECIPE_DIR/patch_glibc_versions.py" "$PREFIX" --max-glibc 2.34

# Remove any runtime state that was created during the install.
rm -rf "$SITE_PACKAGES/isaacsim/kit/logs" "$SITE_PACKAGES/isaacsim/kit/data" "$SITE_PACKAGES/isaacsim/kit/cache"

## Activation scripts
#
# ISAACSIM_PATH and ISAACSIM_PYTHON_EXE are the variables that Isaac Lab and other
# Isaac Sim tooling use to locate an Isaac Sim installation.
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
cat <<EOF > "$PREFIX/etc/conda/activate.d/isaacsim-activate.sh"
export ISAACSIM_PATH="\$CONDA_PREFIX/lib/python$PYTHON_VERSION/site-packages/isaacsim"
export ISAACSIM_PYTHON_EXE="\$CONDA_PREFIX/bin/python"
EOF
cat <<EOF > "$PREFIX/etc/conda/deactivate.d/isaacsim-deactivate.sh"
unset ISAACSIM_PATH
unset ISAACSIM_PYTHON_EXE
EOF

du -sh "$PREFIX"
