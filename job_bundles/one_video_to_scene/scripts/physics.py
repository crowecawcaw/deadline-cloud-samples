"""Physics stage: estimate each object's material, mass, and collision shape, and correct the scene scale.

Inputs (analyze):     objects.json, crops/obj_NN.png
Inputs (reconstruct): obj_NN/mesh.ply
Inputs (register):    scene.json
Outputs (physics):    scene.json              register's scene, rescaled, with a "physics" entry per object
                      properties.json         the same physical properties, plus the model's raw answers
                      obj_NN/collision.npz    convex parts in the object's local frame, meters

The vision-language model only classifies: it picks a material from a fixed list, says whether
the object is a thin-walled shell, and gives the typical height and mass range for that kind of
object. The physical constants come from the MATERIALS table, and volume, mass, center of mass,
and inertia come from the mesh at its registered size.

Monocular depth gets the absolute scale of a scene only roughly right, and mass grows with the
cube of the scale. When objects that come in a narrow range of sizes (a mug, an apple, a soda
can) measure outside their typical height, the whole scene is rescaled by the median correction.
"""

import argparse
import gc
import json
import re

import numpy as np
import trimesh

import stages

# Approximate handbook values. density kg/m^3, friction against a typical surface, restitution,
# and the wall thickness in meters used for objects the model calls hollow.
MATERIALS = {
    "ceramic":   {"density": 2400, "static_friction": 0.6, "dynamic_friction": 0.5, "restitution": 0.3, "wall": 0.005},
    "glass":     {"density": 2500, "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.4, "wall": 0.004},
    "metal":     {"density": 7800, "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.3, "wall": 0.001},
    "aluminum":  {"density": 2700, "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.3, "wall": 0.0005},
    "wood":      {"density": 600,  "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.3, "wall": 0.01},
    "plastic":   {"density": 950,  "static_friction": 0.4, "dynamic_friction": 0.3, "restitution": 0.4, "wall": 0.002},
    "rubber":    {"density": 1100, "static_friction": 0.9, "dynamic_friction": 0.8, "restitution": 0.8, "wall": 0.003},
    "cardboard": {"density": 700,  "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.1, "wall": 0.004},
    "paper":     {"density": 800,  "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.1, "wall": 0.001},
    "fabric":    {"density": 300,  "static_friction": 0.7, "dynamic_friction": 0.6, "restitution": 0.05, "wall": 0.003},
    "foam":      {"density": 50,   "static_friction": 0.7, "dynamic_friction": 0.6, "restitution": 0.3, "wall": 0.01},
    "stone":     {"density": 2600, "static_friction": 0.6, "dynamic_friction": 0.5, "restitution": 0.2, "wall": 0.01},
    "fruit":     {"density": 850,  "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.2, "wall": 0.005},
    "food":      {"density": 900,  "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.1, "wall": 0.005},
    # Mostly for environment surfaces.
    "plaster":   {"density": 850,  "static_friction": 0.6, "dynamic_friction": 0.5, "restitution": 0.2, "wall": 0.0125},
    "concrete":  {"density": 2400, "static_friction": 0.7, "dynamic_friction": 0.6, "restitution": 0.2, "wall": 0.05},
    "tile":      {"density": 2300, "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.3, "wall": 0.01},
    "carpet":    {"density": 300,  "static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.05, "wall": 0.01},
}

# The format uses placeholders rather than example values, because small models tend to copy examples.
PROMPT = (
    'This image shows an object cut out from a video, labeled "{label}". Estimate its physical properties. '
    "Answer with only a JSON object with these keys: "
    '{{"object": <string>, "material": <string>, "hollow": <bool>, "typical_height_cm": [<min>, <max>], '
    '"typical_mass_kg": [<min>, <max>], "size_varies": <bool>}}. '
    '"object" says specifically what the object is, like "coffee mug" or "tennis ball". '
    '"material" is the main material, one of: {materials}. '
    '"hollow" is true for thin-walled containers and shells, like a mug, a box, a bottle, a can, or an '
    "inflated ball, and false for solid objects. "
    '"typical_height_cm" and "typical_mass_kg" are the [min, max] range for this kind of object, as it '
    "stands. "
    '"size_varies" is true when objects of this kind come in very different sizes, like boxes, toys, '
    "or furniture, and false when they come in a narrow range, like mugs, apples, or soda cans."
)


def parse_json_object(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def ask_vlm(model_id: str, items):
    """Ask the model about each (label, RGBA crop) item. Returns one parsed dict per item."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_id)
    answers = []
    for label, crop_path in items:
        rgba = Image.open(crop_path).convert("RGBA")
        image = Image.new("RGB", rgba.size, (128, 128, 128))
        image.paste(rgba, mask=rgba.getchannel("A"))
        prompt = PROMPT.format(label=label.replace("_", " "), materials=", ".join(MATERIALS))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        print(f"Model answer for {label}: {text.strip()}")
        answers.append({"raw": text.strip(), **parse_json_object(text)})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return answers


def number_range(value, default):
    """A [min, max] pair of positive floats, or default when the model's answer isn't one."""
    try:
        lo, hi = sorted(float(v) for v in value)
        return [lo, hi] if lo > 0 else default
    except (TypeError, ValueError):
        return default


def scale_correction(heights, answers, min_votes=2, max_factor=3.0):
    """Scene scale factor that brings narrow-range objects inside their typical height range.

    Each object votes for the smallest change that puts its height in range (no change when it
    already is). Objects whose size varies a lot don't vote, and one vote alone doesn't rescale
    the scene. Returns (factor, votes).
    """
    votes = []
    for height, answer in zip(heights, answers):
        lo, hi = number_range(answer.get("typical_height_cm"), [0, 0])
        if hi <= 0 or answer.get("size_varies", True) is not False or hi / lo > 3:
            continue
        lo, hi = lo / 100, hi / 100
        votes.append(lo / height if height < lo else hi / height if height > hi else 1.0)
    if len(votes) < min_votes:
        return 1.0, votes
    factor = float(np.exp(np.median(np.log(votes))))
    return float(np.clip(factor, 1 / max_factor, max_factor)), votes


def mass_properties(mesh: trimesh.Trimesh, hollow: bool, material: dict, resolution=48):
    """Volume, mass, center of mass, and inertia for a mesh in meters, at uniform density.

    TRELLIS meshes are rarely watertight, so the solid comes from a filled voxelization. Hollow
    objects are modeled as a shell of the material's wall thickness over the outer surface, unless
    the voxelized solid (which already has walls when the mesh models them) is lighter. The shell
    uses the convex hull's area, because TRELLIS surfaces are lumpy and sometimes double-walled,
    which inflates the mesh's own area.
    """
    pitch = float(np.max(mesh.extents)) / resolution
    voxels = mesh.voxelized(pitch).fill()
    solid_points = voxels.points
    solid_volume = len(solid_points) * pitch**3
    density = material["density"]
    if hollow:
        shell_volume = mesh.convex_hull.area * material["wall"]
        if shell_volume < solid_volume:
            points, _ = trimesh.sample.sample_surface_even(mesh, 20000)
            return finish_mass_properties(np.asarray(points), shell_volume, density, "shell")
    return finish_mass_properties(solid_points, solid_volume, density, "solid")


def finish_mass_properties(points, volume, density, model):
    mass = volume * density
    com = points.mean(0)
    r = points - com
    # Inertia of point masses: sum m (|r|^2 I - r r^T)
    inertia = (mass / len(points)) * (np.sum(r * r) * np.eye(3) - r.T @ r)
    principal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    return {
        "model": model,
        "volume_m3": float(volume),
        "mass_kg": float(mass),
        "center_of_mass": com.tolist(),
        "diagonal_inertia": principal.tolist(),
        # Columns are the principal axes in the object's local frame.
        "principal_axes": axes.tolist(),
    }


def collision_parts(mesh: trimesh.Trimesh, max_hulls: int, threshold: float):
    """Convex decomposition with CoACD (MIT). Falls back to the convex hull."""
    simple = mesh
    if len(mesh.faces) > 8000:
        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
        ).simplify_quadric_decimation(8000)
        simple = trimesh.Trimesh(np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.triangles))
    try:
        import coacd

        coacd.set_log_level("error")
        parts = coacd.run_coacd(
            coacd.Mesh(simple.vertices, simple.faces), threshold=threshold, max_convex_hull=max_hulls, seed=0
        )
        parts = [trimesh.Trimesh(v, f) for v, f in parts]
        if parts:
            return parts, "coacd"
    except Exception as e:  # noqa: BLE001 - any failure falls back to a hull
        print(f"WARNING: CoACD failed ({e}), using the convex hull")
    return [mesh.convex_hull], "convex_hull"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--scale-from-size-priors", choices=["on", "off"], default="on")
    parser.add_argument("--max-hulls", type=int, default=16)
    parser.add_argument("--hull-threshold", type=float, default=0.05, help="CoACD concavity threshold.")
    args = parser.parse_args()

    analyze = stages.input_dir("analyze")
    reconstruct = stages.input_dir("reconstruct")
    scene = stages.read_json(stages.input_dir("register") / "scene.json")
    out = stages.output_dir("physics")
    crops = {o["id"]: o.get("crop") for o in stages.read_json(analyze / "objects.json")["objects"]}

    # Meshes at their registered size, in the object's local frame. Register only scales uniformly.
    meshes, heights = [], []
    for desc in scene["objects"]:
        M = np.array(desc["matrices"][0])
        scale = float(np.linalg.norm(M[:3, 0]))
        mesh = trimesh.load(reconstruct / desc["mesh"], process=False)
        mesh.apply_scale(scale)
        meshes.append(mesh)
        heights.append(float(mesh.extents[2]))

    print("openjd_status: Estimating materials")
    items = [(d["label"], analyze / crops[d["id"]]) for d in scene["objects"]]
    answers = ask_vlm(args.vlm_model, items) if items else []

    factor, votes = 1.0, []
    if args.scale_from_size_priors == "on":
        factor, votes = scale_correction(heights, answers)
    print(f"Scene scale correction {factor:.3f} from votes {np.round(votes, 3).tolist()}")
    K = np.diag([factor, factor, factor, 1.0])
    scene["ground"]["size"] *= factor
    scene["input_to_world"] = (K @ np.array(scene["input_to_world"])).tolist()
    for m in scene["camera"]["cam_to_world"]:
        for row in range(3):
            m[row][3] *= factor

    print("openjd_status: Computing mass and collision shapes")
    properties = []
    for desc, mesh, answer in zip(scene["objects"], meshes, answers):
        desc["matrices"] = [(K @ np.array(m)).tolist() for m in desc["matrices"]]
        mesh.apply_scale(factor)
        name = str(answer.get("material", "")).lower().strip()
        material_name = name if name in MATERIALS else "plastic"
        material = MATERIALS[material_name]
        hollow = bool(answer.get("hollow", False))
        mass = mass_properties(mesh, hollow, material)

        # The model's mass ranges are unreliable (it put a ceramic mug at 40-100 g), so they only
        # flag a likely volume or scale error for review rather than override the computed mass.
        lo, hi = number_range(answer.get("typical_mass_kg"), [0, np.inf])
        mass_check = "ok" if lo / 2 <= mass["mass_kg"] <= hi * 2 else "outside_typical_range"

        parts, method = collision_parts(mesh, args.max_hulls, args.hull_threshold)
        obj_dir = out / f"obj_{desc['id']:02d}"
        obj_dir.mkdir(exist_ok=True)
        arrays = {}
        for i, part in enumerate(parts):
            arrays[f"vertices_{i}"], arrays[f"faces_{i}"] = part.vertices, part.faces
        np.savez_compressed(obj_dir / "collision.npz", count=len(parts), **arrays)

        physics = {
            "identified_as": str(answer.get("object", desc["label"])),
            "material": material_name,
            "hollow": hollow,
            **{k: v for k, v in material.items() if k != "wall"},
            **mass,
            "mass_check": mass_check,
            # The scale that takes reconstruct's mesh.ply to meters. Matrices include it too.
            "mesh_scale": float(np.linalg.norm(desc["matrices"][0][0][:3])),
            "dimensions_m": mesh.extents.tolist(),
            "collision": f"obj_{desc['id']:02d}/collision.npz",
            "collision_method": method,
            "collision_parts": len(parts),
        }
        desc["physics"] = physics
        properties.append({"id": desc["id"], "label": desc["label"], **physics, "model_answer": answer})
        print(
            f"Object {desc['id']} {desc['label']} ({physics['identified_as']}): {material_name}{' shell' if hollow else ''}, "
            f"{np.round(mesh.extents * 100, 1).tolist()} cm, {mass['mass_kg']:.3f} kg ({mass_check}), "
            f"{len(parts)} collision parts ({method})"
        )

    stages.write_json(out / "scene.json", scene)
    stages.write_json(out / "properties.json", {"scale_correction": factor, "scale_votes": votes, "objects": properties})
    print(f"openjd_status: Estimated physics for {len(properties)} objects, scale correction {factor:.2f}")


if __name__ == "__main__":
    main()
