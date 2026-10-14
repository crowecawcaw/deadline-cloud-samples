"""Render stage (runs inside Blender): render the assembled scene from the video's camera path.

Writes render/textured/####.png and render/clay/####.png (a white-model pass).
make_comparison.py then encodes them side by side with the input frames.
"""

import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
import stages  # noqa: E402


def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("OPTIX", "CUDA", "HIP", "METAL"):
        try:
            prefs.compute_device_type = device_type
        except TypeError:
            continue
        prefs.refresh_devices()
        gpus = [d for d in prefs.devices if d.type == device_type]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == device_type
            bpy.context.scene.cycles.device = "GPU"
            print(f"Rendering with {device_type}: {[d.name for d in gpus]}")
            return
    print("No GPU found for Cycles, rendering on CPU")


def render_pass(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = f"{out_dir}/"
    bpy.ops.render.render(animation=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args(argv)

    bpy.ops.wm.open_mainfile(filepath=str(stages.input_dir("assemble") / "scene.blend"))
    out = stages.output_dir("render")
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    enable_gpu()

    render_pass(out / "textured")

    clay = bpy.data.materials.new("Clay")
    clay.use_nodes = True
    clay.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
    bpy.context.view_layer.material_override = clay
    render_pass(out / "clay")


if __name__ == "__main__":
    main()
