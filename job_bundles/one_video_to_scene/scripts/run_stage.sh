#!/usr/bin/env bash
# Run one pipeline stage with the right interpreter.
#   run_stage.sh STAGE [ARGS...]
# PipelineEnv (setup_env.sh) sets ML_PY, TRELLIS_SRC, OUTPUT_DIR, PRIOR_OUTPUT_DIR, and SCRIPT_DIR.
set -euo pipefail

STAGE=$1
shift

case "$STAGE" in
  analyze|register)
    exec "$ML_PY" "$SCRIPT_DIR/$STAGE.py" "$@"
    ;;
  reconstruct)
    PYTHONPATH="$TRELLIS_SRC${PYTHONPATH:+:$PYTHONPATH}" exec "$ML_PY" "$SCRIPT_DIR/reconstruct.py" "$@"
    ;;
  assemble)
    exec blender --background --factory-startup --python-exit-code 1 \
      --python "$SCRIPT_DIR/assemble_blender.py" -- "$@"
    ;;
  render)
    blender --background --factory-startup --python-exit-code 1 \
      --python "$SCRIPT_DIR/render_blender.py" -- "$@"
    exec "$ML_PY" "$SCRIPT_DIR/make_comparison.py"
    ;;
  *)
    echo "ERROR: Unknown stage $STAGE" >&2
    exit 1
    ;;
esac
