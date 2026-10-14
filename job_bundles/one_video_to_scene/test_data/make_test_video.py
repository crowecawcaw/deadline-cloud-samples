"""Render a short synthetic tabletop video to test the one_video_to_scene job bundle.

The scene has a few everyday objects on a table, a slowly orbiting camera, and one
object (a ball) that rolls across the table so the pipeline has a rigid, moving object
to track. Everything is procedural, so the video has no third-party license terms.

Usage:
    blender --background --factory-startup --python make_test_video.py -- OUTPUT_DIR [--frames N]

Writes OUTPUT_DIR/frame_####.png. Encode them with:
    ffmpeg -framerate 24 -i OUTPUT_DIR/frame_%04d.png -pix_fmt yuv420p tabletop.mp4
"""

import argparse
import math
import sys

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--resolution", type=int, nargs=2, default=[768, 512])
    return parser.parse_args(argv)


def make_material(name, color, roughness=0.5, noise_scale=None, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if noise_scale:
        # Subtle procedural variation so surfaces don't look flat.
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = noise_scale
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*[c * 0.7 for c in color], 1.0)
        ramp.color_ramp.elements[1].color = (*color, 1.0)
        links = mat.node_tree.links
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add(obj, mat):
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return obj


def build_scene(frames):
    scene = bpy.context.scene

    # Table top and floor
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -0.025))
    table = bpy.context.object
    table.name = "Table"
    table.scale = (1.6, 1.0, 0.05)
    table.data.materials.append(make_material("Wood", (0.45, 0.28, 0.15), 0.6, noise_scale=40))

    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.8))
    bpy.context.object.data.materials.append(make_material("Floor", (0.5, 0.5, 0.52), 0.9))

    # Mug: a hollow cylinder with a torus handle
    bpy.ops.mesh.primitive_cylinder_add(radius=0.07, depth=0.16, location=(-0.3, 0.12, 0.08), vertices=48)
    mug = bpy.context.object
    mug.name = "Mug"
    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.16, location=(-0.3, 0.12, 0.095), vertices=48)
    inner = bpy.context.object
    boolean = mug.modifiers.new("hollow", "BOOLEAN")
    boolean.object = inner
    boolean.operation = "DIFFERENCE"
    bpy.context.view_layer.objects.active = mug
    bpy.ops.object.modifier_apply(modifier="hollow")
    bpy.data.objects.remove(inner)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.045, minor_radius=0.012, location=(-0.375, 0.12, 0.085), rotation=(math.pi / 2, 0, 0)
    )
    handle = bpy.context.object
    mug_mat = make_material("Ceramic", (0.12, 0.35, 0.75), 0.25)
    for obj in (mug, handle):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mug
    bpy.ops.object.join()
    add(mug, mug_mat)

    # Cereal-style box
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.3, 0.2, 0.13))
    box = bpy.context.object
    box.name = "Box"
    box.scale = (0.18, 0.07, 0.26)
    box.rotation_euler = (0, 0, math.radians(-20))
    box.data.materials.append(make_material("Cardboard", (0.95, 0.55, 0.1), 0.7, noise_scale=8))

    # Apple-ish fruit: a squashed sphere with a stem
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(0.05, 0.25, 0.045), segments=48, ring_count=24)
    apple = bpy.context.object
    apple.name = "Apple"
    apple.scale = (1.0, 1.0, 0.9)
    add(apple, make_material("AppleSkin", (0.75, 0.05, 0.05), 0.35, noise_scale=15))
    bpy.ops.mesh.primitive_cylinder_add(radius=0.004, depth=0.03, location=(0.05, 0.25, 0.098))
    bpy.context.object.data.materials.append(make_material("Stem", (0.25, 0.15, 0.05), 0.8))

    # Rolling ball: the rigid moving object
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.045, location=(-0.5, -0.2, 0.045), segments=48, ring_count=24)
    ball = bpy.context.object
    ball.name = "Ball"
    add(ball, make_material("Rubber", (0.1, 0.65, 0.2), 0.45, noise_scale=25))
    travel = 0.9
    ball.location = (-0.5, -0.2, 0.045)
    ball.rotation_euler = (0, 0, 0)
    ball.keyframe_insert("location", frame=1)
    ball.keyframe_insert("rotation_euler", frame=1)
    ball.location = (-0.5 + travel, -0.15, 0.045)
    ball.rotation_euler = (0, travel / 0.045, 0)
    ball.keyframe_insert("location", frame=frames)
    ball.keyframe_insert("rotation_euler", frame=frames)

    # Lights and world
    bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(30))
    bpy.ops.object.light_add(type="AREA", location=(-1.5, -1.0, 1.5))
    fill = bpy.context.object
    fill.data.energy = 150
    fill.data.size = 2.0
    fill.rotation_euler = (math.radians(50), 0, math.radians(-55))
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.65, 0.7, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

    # Camera orbiting 35 degrees around the table while looking at its center
    bpy.ops.object.empty_add(location=(0, 0.05, 0.05))
    target = bpy.context.object
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.data.lens = 35
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    scene.camera = cam
    radius, height = 1.15, 0.6
    for frame in range(1, frames + 1):
        angle = math.radians(-110 + 35 * (frame - 1) / max(frames - 1, 1))
        cam.location = Vector((radius * math.cos(angle), radius * math.sin(angle), height))
        cam.keyframe_insert("location", frame=frame)

    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for kp in fcurve.keyframe_points:
                    kp.interpolation = "LINEAR"


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    build_scene(args.frames)

    scene.frame_start, scene.frame_end = 1, args.frames
    scene.render.fps = 24
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 2, 0) else "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = f"{args.output_dir.rstrip('/')}/frame_"
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
