"""Where each stage reads and writes. Standard library only, so Blender's Python can import it too.

Every stage writes to $OUTPUT_DIR/<stage>. A stage's inputs come from $OUTPUT_DIR/<stage> when an
earlier step of this job produced them, and from $PRIOR_OUTPUT_DIR/<stage> when the job was
submitted as a restart that skipped that step.
"""

import json
import os
from pathlib import Path


def output_dir(stage: str) -> Path:
    path = Path(os.environ["OUTPUT_DIR"]) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def input_dir(stage: str) -> Path:
    candidates = [Path(os.environ["OUTPUT_DIR"]) / stage]
    if os.environ.get("PRIOR_OUTPUT_DIR"):
        candidates.append(Path(os.environ["PRIOR_OUTPUT_DIR"]) / stage)
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path
    raise FileNotFoundError(f"No outputs from the {stage} stage in: {', '.join(map(str, candidates))}")


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        # default= handles numpy scalars and arrays.
        json.dump(data, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
