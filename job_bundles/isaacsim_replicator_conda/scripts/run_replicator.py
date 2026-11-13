"""Generate annotated synthetic frames with NVIDIA Isaac Sim Replicator.

Runs with the Python interpreter of a Conda environment that has the isaacsim package. For a
frame range [start, end] (inclusive), the script builds a scene of randomized cubes on a ground
plane, optionally adds a USD asset from the Isaac Sim asset library, and uses BasicWriter to write
one annotated frame (RGB, 2D bounding boxes, and semantic segmentation) per frame in the range.
BasicWriter numbers the files in each range's output directory from 0000.
"""
import argparse
import os
import random
import re
import sys
import time
import traceback

# SimulationApp must be created before any other omni or isaacsim imports.
from isaacsim import SimulationApp


def parse_contiguous_range(text):
    """Parse a CHUNK[INT] task parameter value like '0-3', '7', or '-5--1' into (start, end)."""
    m = re.fullmatch(r"\s*(-?\d+)\s*(?:-\s*(-?\d+)\s*)?", text)
    if not m:
        raise ValueError(f"Expected a contiguous frame range like '0-7', got {text!r}")
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) is not None else start
    if end < start:
        raise ValueError(f"Frame range {text!r} ends before it starts")
    return start, end


def resolve_asset(asset_path):
    """Return a URL for asset_path, resolving paths relative to the Isaac Sim asset root."""
    if not asset_path:
        return None
    if "://" in asset_path or os.path.isabs(asset_path):
        return asset_path
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if not root:
        raise RuntimeError("Could not find the Isaac Sim asset root")
    return f"{root.rstrip('/')}/{asset_path.lstrip('/')}"


def render(args, output_dir, frame_start, num_frames):
    """Build the randomized scene and write num_frames annotated frames. Returns the RGB file count."""
    import omni.replicator.core as rep

    seed = 1000 + frame_start
    random.seed(seed)
    rep.set_global_seed(seed)

    asset_url = resolve_asset(args.asset)
    if asset_url:
        print(f"[replicator] adding asset {asset_url}", flush=True)
        started = time.time()
        rep.create.from_usd(asset_url, semantics=[("class", "asset")])
        print(f"[replicator] asset referenced in {time.time() - started:.1f}s", flush=True)

    rep.create.plane(scale=10, position=(0, 0, 0), semantics=[("class", "ground")])
    cubes = [
        rep.create.cube(position=(0, 0, 0.5), scale=0.3, semantics=[("class", f"cube_{i}")])
        for i in range(random.randint(3, 5))
    ]
    light = rep.create.light(light_type="distant", intensity=3000, rotation=(-45, 0, 0))
    rep.create.light(light_type="dome", intensity=500)
    camera = rep.create.camera(position=(3, 3, 2), look_at=(0, 0, 0.5))

    # Randomize the cube poses, light intensity, and camera position before each captured frame.
    with rep.trigger.on_frame(max_execs=num_frames):
        with rep.create.group(cubes):
            rep.modify.pose(
                position=rep.distribution.uniform((-2.5, -2.5, 0.15), (2.5, 2.5, 1.0)),
                scale=rep.distribution.uniform(0.15, 0.4),
            )
        with rep.create.group([light]):
            rep.modify.attribute("intensity", rep.distribution.uniform(2000, 5000))
        with rep.create.group([camera]):
            rep.modify.pose(
                position=rep.distribution.uniform((2, 2, 1.5), (4, 4, 3)),
                look_at=(0, 0, 0.5),
            )

    render_product = rep.create.render_product(camera, (args.width, args.height))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        frame_padding=4,
    )
    writer.attach([render_product])

    print(f"openjd_status: Rendering {num_frames} frame(s)", flush=True)
    started = time.time()
    rep.orchestrator.run_until_complete(num_frames=num_frames)
    rep.BackendDispatch.wait_until_done()
    print(f"[replicator] rendered {num_frames} frame(s) in {time.time() - started:.1f}s", flush=True)

    rgb_written = 0
    for name in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, name)
        print(f"  {name} ({os.path.getsize(path)} bytes)", flush=True)
        # Only count files from this run, not ones left by an earlier attempt.
        if name.startswith("rgb_") and name.endswith(".png") and os.path.getmtime(path) >= started:
            rgb_written += 1
    return rgb_written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-range", required=True, help="Inclusive range, e.g. '0-3' or '7'.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--asset",
        default="",
        help="USD asset to add to the scene: a URL, an absolute path, or a path relative to the Isaac Sim asset root.",
    )
    args = parser.parse_args()

    frame_start, frame_end = parse_contiguous_range(args.frame_range)
    num_frames = frame_end - frame_start + 1
    output_dir = os.path.join(args.output_dir, f"frames_{frame_start:04d}-{frame_end:04d}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"openjd_status: Starting Isaac Sim for frames {frame_start}-{frame_end}", flush=True)
    started = time.time()
    sim = SimulationApp({"headless": True, "width": args.width, "height": args.height})
    print(f"[replicator] SimulationApp ready in {time.time() - started:.1f}s", flush=True)

    # After Kit starts, an uncaught exception can hang interpreter shutdown, and
    # SimulationApp.close() can end the process with status 0. Exit directly on failure instead.
    try:
        rgb_written = render(args, output_dir, frame_start, num_frames)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    if rgb_written < num_frames:
        print(f"ERROR: expected {num_frames} RGB frames, found {rgb_written}", flush=True)
        os._exit(1)

    sim.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
