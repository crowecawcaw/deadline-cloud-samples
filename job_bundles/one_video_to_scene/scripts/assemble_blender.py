"""Assemble stage (runs inside Blender): build the animated scene and export GLB and .blend.

Reads register/scene.json, where every transform is already in Blender's world frame
(Z up, meters, ground plane at z=0), and the per-object meshes from the reconstruct stage.
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).parent))
import stages  # noqa: E402


def vertex_color_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    color = nodes.new("ShaderNodeVertexColor")
    links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def import_mesh(path: Path, name: str):
    bpy.ops.object.select_all(action="DESELECT")
    # PLY with identity axes keeps the same vertex coordinates that the register stage fitted.
    bpy.ops.wm.ply_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    obj.name = name
    obj.data.materials.clear()
    obj.data.materials.append(vertex_color_material(f"{name}_material"))
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    return obj


def keyframe_matrices(obj, matrices, start_frame=1):
    for i, m in enumerate(matrices):
        obj.matrix_world = Matrix(m)
        obj.keyframe_insert("location", frame=start_frame + i)
        obj.keyframe_insert("rotation_quaternion" if obj.rotation_mode == "QUATERNION" else "rotation_euler",
                            frame=start_frame + i)
        obj.keyframe_insert("scale", frame=start_frame + i)


def main():
    scene_desc = stages.read_json(stages.input_dir("register") / "scene.json")
    reconstruct_dir = stages.input_dir("reconstruct")
    out = stages.output_dir("assemble")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    num_frames = len(scene_desc["camera"]["cam_to_world"])
    scene.frame_start, scene.frame_end = 1, num_frames
    scene.render.fps = round(scene_desc.get("fps", 24))
    scene.render.resolution_x, scene.render.resolution_y = scene_desc["width"], scene_desc["height"]

    # Ground plane, a passive rigid body so the scene is ready for a physics pass.
    size = scene_desc["ground"].get("size", 4.0)
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"
    ground_mat = bpy.data.materials.new("Ground")
    ground_mat.use_nodes = True
    ground_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1)
    ground.data.materials.append(ground_mat)
    bpy.ops.rigidbody.object_add(type="PASSIVE")

    for desc in scene_desc["objects"]:
        mesh_path = reconstruct_dir / desc["mesh"]
        if not mesh_path.exists():
            print(f"WARNING: Missing mesh {mesh_path}, skipping {desc['label']}")
            continue
        obj = import_mesh(mesh_path, f"{desc['id']:02d}_{desc['label']}")
        obj["motion"] = desc["motion"]
        matrices = desc["matrices"]
        if len(matrices) == 1:
            obj.matrix_world = Matrix(matrices[0])
        else:
            keyframe_matrices(obj, matrices)
        # Static objects are passive colliders; tracked objects are animated (kinematic) rigid bodies.
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add(type="PASSIVE" if desc["motion"] == "static" else "ACTIVE")
        if desc["motion"] != "static":
            obj.rigid_body.kinematic = True
        obj.rigid_body.collision_shape = "CONVEX_HULL"

    # Camera following the recovered path. Intrinsics come from the first frame.
    cam_desc = scene_desc["camera"]
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = "VideoCamera"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = cam_desc["fx"] * 36.0 / scene_desc["width"]
    cam.data.shift_x = (scene_desc["width"] / 2 - cam_desc["cx"]) / scene_desc["width"]
    cam.data.shift_y = (cam_desc["cy"] - scene_desc["height"] / 2) / scene_desc["width"]
    cam.data.clip_start, cam.data.clip_end = 0.01, 100
    keyframe_matrices(cam, cam_desc["cam_to_world"])
    scene.camera = cam

    # Simple lighting
    bpy.ops.object.light_add(type="SUN", rotation=(math.radians(40), math.radians(10), math.radians(30)))
    bpy.context.object.data.energy = 3.0
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.7, 0.72, 0.75, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8

    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for kp in fcurve.keyframe_points:
                    kp.interpolation = "LINEAR"

    bpy.context.preferences.filepaths.save_version = 0  # no scene.blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    bpy.ops.export_scene.gltf(
        filepath=str(out / "scene.glb"),
        export_format="GLB",
        export_animations=True,
        export_cameras=True,
        export_lights=True,
        export_apply=False,
    )
    print(f"openjd_status: Assembled {len(scene_desc['objects'])} objects")


if __name__ == "__main__":
    main()
