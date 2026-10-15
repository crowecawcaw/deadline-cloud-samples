"""Environment stage: reconstruct the static surroundings (floor, walls, tables, everything else) as meshes.

Inputs (analyze):  frames/, depth.npz, cameras.json, masks/obj_NN.npz
Inputs (physics):  scene.json, for the world transform after scale correction and the object list
Outputs (environment):
  environment.json         One entry per surface: kind, plane, mesh, material, and friction
  surfaces/<name>.ply      Meshes in the world frame (Z up, meters), with vertex colors
  labels.png               The first frame colored by surface, for checking at a glance

How it works:
  1. Fit the major planes among the background points, split each into connected patches, and
     classify them as floor, horizontal surface (a table top or shelf), or wall.
  2. Edit each frame's depth: background pixels close to a patch snap onto it, so floors and walls
     come out flat instead of carrying monocular depth noise. Pixels covered by an object take the
     depth of the patch behind it, which fills the floor and table top under the objects.
  3. Fuse the edited depth of all frames into a TSDF volume (Open3D) and extract a colored mesh.
  4. Split the mesh by patch, and ask the vision-language model what material each surface is.

Only surfaces the camera saw are reconstructed. With a single image or a static camera, the
environment is a shell seen from one side.
"""

import argparse
import gc
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull, Delaunay

import stages
from physics import MATERIALS, parse_json_object
from register import load_masks

SURFACE_PROMPT = (
    "The highlighted region of this image (everything else is grayed out) is one surface of the "
    'scene, which looks like a {kind}. Answer with only a JSON object: {{"surface": <string>, '
    '"material": <string>}}. "surface" says what it is, like "wood floor" or "painted wall". '
    '"material" is its main material, one of: {materials}.'
)


class Patch:
    """A connected planar region: plane n.x + d = 0 and its convex outline in plane coordinates."""

    def __init__(self, normal, d, points, kind):
        self.normal, self.d, self.kind = normal, d, kind
        helper = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
        self.u = np.cross(normal, helper)
        self.u /= np.linalg.norm(self.u)
        self.v = np.cross(normal, self.u)
        uv = self.to_plane(points)
        hull = ConvexHull(uv)
        self.outline = uv[hull.vertices]
        self.area = float(hull.volume)
        self.tri = Delaunay(self.outline)
        self.height = float(np.median(points[:, 2]))

    def to_plane(self, points):
        return np.stack([points @ self.u, points @ self.v], axis=-1)

    def contains(self, points):
        return self.tri.find_simplex(self.to_plane(points)) >= 0


def fit_plane(points):
    centroid = points.mean(0)
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][2]
    return normal, -normal @ centroid


def find_patches(points, cam_centers, threshold, max_planes=10, min_fraction=0.02):
    """Fit planes with RANSAC, merge near-duplicates, and split each plane into connected patches."""
    planes = []  # [normal, d, points]
    remaining = points
    min_points = max(int(min_fraction * len(points)), 500)
    for _ in range(max_planes):
        if len(remaining) < min_points:
            break
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining))
        plane, _ = pcd.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=3000)
        normal = np.array(plane[:3]) / np.linalg.norm(plane[:3])
        d = plane[3] / np.linalg.norm(plane[:3])
        inliers = np.abs(remaining @ normal + d) < threshold
        if inliers.sum() < min_points:
            break
        normal, d = fit_plane(remaining[inliers])
        inliers = np.abs(remaining @ normal + d) < threshold
        found, remaining = remaining[inliers], remaining[~inliers]
        # Monocular depth bends large surfaces like floors, so RANSAC can return the same surface
        # as several slightly offset planes. Merge a plane into an earlier one it mostly agrees with.
        for earlier in planes:
            if abs(earlier[0] @ normal) > np.cos(np.radians(8)) and \
                    np.median(np.abs(found @ earlier[0] + earlier[1])) < 4 * threshold:
                earlier[2] = np.concatenate([earlier[2], found])
                earlier[0], earlier[1] = fit_plane(earlier[2])
                break
        else:
            planes.append([normal, d, found])

    patches = []
    rng = np.random.default_rng(0)
    for normal, d, plane_points in planes:
        if abs(normal[2]) > 0.95:
            kind = "horizontal"
            if normal[2] < 0:
                normal, d = -normal, -d
        elif abs(normal[2]) < 0.15:
            kind = "wall"
            if np.mean(cam_centers @ normal + d) < 0:
                normal, d = -normal, -d
        else:
            continue  # slanted planes are left to the free-form mesh
        sample = plane_points[rng.choice(len(plane_points), min(len(plane_points), 20000), replace=False)]
        labels = np.asarray(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample)).cluster_dbscan(eps=8 * threshold, min_points=10)
        )
        for label in range(labels.max() + 1):
            cluster = sample[labels == label]
            if len(cluster) >= 0.1 * len(sample) and len(cluster) >= 50:
                patches.append(Patch(normal, d, cluster, kind))
    # The lowest large horizontal patch is the floor; other horizontal patches are surfaces.
    horizontal = [p for p in patches if p.kind == "horizontal"]
    if horizontal:
        largest = max(p.area for p in horizontal)
        floor = min((p for p in horizontal if p.area > 0.3 * largest), key=lambda p: p.height)
        for p in horizontal:
            p.kind = "floor" if p is floor else "surface"
    return patches


def edit_depth(depth, object_mask, K, cam_to_world, A, patches, snap):
    """Snap background depth onto nearby patches and fill object pixels with the patch behind them.

    Returns the edited depth (0 where there's nothing) and per-pixel patch labels (-1 for none).
    Depth stays in the analyze stage's units, so it can be fused with the analyze cameras.
    """
    H, W = depth.shape
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays_cam = np.stack([(xs - K[0][2]) / K[0][0], (ys - K[1][2]) / K[1][1], np.ones_like(xs)], -1)
    M = A @ cam_to_world
    origin = M[:3, 3]
    rays = rays_cam @ M[:3, :3].T  # world offset per unit of analyze depth
    points = origin + rays * depth[..., None]

    out = np.where(object_mask, 0.0, depth).astype(np.float32)
    labels = np.full((H, W), -1, np.int32)
    best_residual = np.full((H, W), np.inf)
    fill_t = np.full((H, W), np.inf)
    for i, patch in enumerate(patches):
        denom = rays @ patch.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -(origin @ patch.normal + patch.d) / denom
        valid = np.isfinite(t) & (t > 0)
        hit = origin + rays * np.where(valid, t, 0)[..., None]
        valid &= patch.contains(hit.reshape(-1, 3)).reshape(H, W)
        residual = np.abs(points @ patch.normal + patch.d)
        snap_here = valid & ~object_mask & (depth > 0) & (residual < snap) & (residual < best_residual)
        out[snap_here], labels[snap_here], best_residual[snap_here] = t[snap_here], i, residual[snap_here]
        fill_here = valid & object_mask & (t > 0.98 * depth) & (t < fill_t)
        fill_t[fill_here] = t[fill_here]
        labels[fill_here] = i
    filled = np.isfinite(fill_t)
    out[filled] = fill_t[filled]
    return out, labels


def ask_materials(model_id, requests):
    """requests: list of (kind, RGB image with the surface highlighted). Returns parsed answers."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_id)
    answers = []
    for kind, image in requests:
        prompt = SURFACE_PROMPT.format(kind=kind, materials=", ".join(MATERIALS))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        print(f"Model answer for a {kind}: {text.strip()}")
        answers.append({"raw": text.strip(), **parse_json_object(text)})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return answers


def highlight(frame, mask):
    """The frame cropped around the mask, with everything outside the mask grayed out."""
    ys, xs = np.nonzero(mask)
    pad = 20
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    gray = (0.3 * frame + 0.7 * 128).astype(np.uint8)
    image = np.where(mask[..., None], frame, gray)[y0:y1, x0:x1]
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-frames", type=int, default=16, help="Frames to fuse.")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="TSDF voxel in meters. 0 picks one from the scene size.")
    parser.add_argument("--max-faces", type=int, default=300000)
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    analyze = stages.input_dir("analyze")
    scene = stages.read_json(stages.input_dir("physics") / "scene.json")
    out = stages.output_dir("environment")
    (out / "surfaces").mkdir(exist_ok=True)
    cameras = stages.read_json(analyze / "cameras.json")
    depth_all = np.load(analyze / "depth.npz")["depth"].astype(np.float32)
    frame_paths = sorted((analyze / "frames").glob("*.jpg"))
    num_frames = len(cameras["frames"])
    used = np.unique(np.linspace(0, num_frames - 1, min(args.max_frames, num_frames)).round().astype(int))
    A = np.array(scene["input_to_world"])
    world_scale = float(np.cbrt(np.linalg.det(A[:3, :3])))

    # Only the objects that made it into the scene are cut out of the environment. Objects that
    # failed to reconstruct stay part of it, which is better than leaving a hole.
    object_mask = np.zeros(depth_all.shape, bool)
    for desc in scene["objects"]:
        object_mask |= load_masks(analyze / "masks" / f"obj_{desc['id']:02d}.npz")
    object_mask = np.stack([binary_dilation(m, iterations=3) for m in object_mask])

    Ks = [np.array(f["K"]) for f in cameras["frames"]]
    c2w = [np.array(f["cam_to_world"]) for f in cameras["frames"]]
    cam_centers = np.array([(A @ m)[:3, 3] for m in c2w])

    # Background points in the world frame, for plane fitting.
    from register import unproject

    background = []
    for i in used:
        valid = ~object_mask[i] & (depth_all[i] < np.quantile(depth_all[i], 0.99))
        pts = unproject(depth_all[i], valid, Ks[i], c2w[i], 400000 // len(used), rng)
        background.append(pts @ A[:3, :3].T + A[:3, 3])
    background = np.concatenate(background)
    extent = float(np.linalg.norm(np.quantile(background, 0.95, 0) - np.quantile(background, 0.05, 0)))
    voxel = args.voxel_size or float(np.clip(extent / 400, 0.005, 0.03))
    snap = 2.5 * voxel
    print(f"openjd_status: Fitting planes ({len(background)} points, extent {extent:.2f} m, voxel {voxel * 100:.1f} cm)")
    patches = find_patches(background, cam_centers, threshold=snap)
    names, counts = [], {}
    for p in patches:
        counts[p.kind] = counts.get(p.kind, 0) + 1
        names.append(p.kind if p.kind == "floor" else f"{p.kind}_{counts[p.kind]}")
    for name, p in zip(names, patches):
        print(f"  {name}: normal {np.round(p.normal, 3).tolist()}, height {p.height:.2f} m, area {p.area:.2f} m^2")

    print("openjd_status: Fusing depth")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel / world_scale, sdf_trunc=4 * voxel / world_scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    label_images = {}
    for i in used:
        depth, labels = edit_depth(depth_all[i], object_mask[i], Ks[i], c2w[i], A, patches, snap)
        label_images[int(i)] = labels
        H, W = depth.shape
        color = np.asarray(Image.open(frame_paths[i]).convert("RGB").resize((W, H)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=float(np.quantile(depth_all[i], 0.99)) * 1.05, convert_rgb_to_intensity=False,
        )
        K = Ks[i]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0][0], K[1][1], K[0][2], K[1][2])
        volume.integrate(rgbd, intrinsic, np.linalg.inv(c2w[i]))
    mesh = volume.extract_triangle_mesh()
    mesh.transform(A)
    clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
    clusters, cluster_sizes = np.asarray(clusters), np.asarray(cluster_sizes)
    mesh.remove_triangles_by_mask(cluster_sizes[clusters] < 0.002 * len(clusters))
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > args.max_faces:
        mesh = mesh.simplify_quadric_decimation(args.max_faces)
    print(f"Environment mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # Assign each face to the patch it lies on, or to "other".
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    centers = vertices[faces].mean(1)
    face_label = np.full(len(faces), -1)
    best = np.full(len(faces), np.inf)
    for i, p in enumerate(patches):
        residual = np.abs(centers @ p.normal + p.d)
        on = (residual < snap) & (residual < best) & p.contains(centers)
        face_label[on], best[on] = i, residual[on]

    # Highlight each surface in the frame where it covers the most pixels, and ask for its material.
    frame_rgb = {i: np.asarray(Image.open(frame_paths[i]).convert("RGB")) for i in label_images}
    surfaces, requests = [], []
    for i, name in [(i, n) for i, n in enumerate(names)] + [(-1, "other")]:
        selected = face_label == i
        if selected.sum() < 50:
            continue
        sub = o3d.geometry.TriangleMesh(mesh)
        sub.remove_triangles_by_mask(~selected)
        sub.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(str(out / "surfaces" / f"{name}.ply"), sub)
        kind = patches[i].kind if i >= 0 else "other"
        surfaces.append({
            "name": name, "kind": kind, "mesh": f"surfaces/{name}.ply", "faces": int(selected.sum()),
            "plane": (patches[i].normal.tolist() + [float(patches[i].d)]) if i >= 0 else None,
        })
        frame = max(label_images, key=lambda f: (label_images[f] == i).sum())
        pixels = label_images[frame] == i
        description = {"floor": "floor", "surface": "table top or shelf", "wall": "wall"}.get(kind, "object or fixture")
        if pixels.sum() >= 100:
            rgb = frame_rgb[frame]
            mask = cv2.resize(pixels.astype(np.uint8), rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            requests.append((len(surfaces) - 1, description, highlight(rgb, mask)))

    print("openjd_status: Estimating surface materials")
    answers = ask_materials(args.vlm_model, [(d, img) for _, d, img in requests]) if requests else []
    for (index, _, image), answer in zip(requests, answers):
        surfaces[index]["model_answer"] = answer
        image.save(out / f"{surfaces[index]['name']}_query.jpg")
    for surface in surfaces:
        answer = surface.get("model_answer", {})
        name = str(answer.get("material", "")).lower().strip()
        default = {"floor": "wood", "wall": "plaster"}.get(surface["kind"], "wood")
        surface["material"] = name if name in MATERIALS else default
        surface["identified_as"] = str(answer.get("surface", surface["kind"]))
        surface.update({k: v for k, v in MATERIALS[surface["material"]].items() if k != "wall"})
        print(f"  {surface['name']}: {surface['identified_as']} ({surface['material']}), {surface['faces']} faces")

    # Debug image: the first used frame colored by surface.
    palette = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212]])
    first = int(used[0])
    labels = label_images[first]
    rgb = np.asarray(Image.open(frame_paths[first]).convert("RGB").resize(labels.shape[::-1])).astype(float)
    overlay = np.where((labels >= 0)[..., None], palette[labels % len(palette)], rgb)
    Image.fromarray((0.5 * rgb + 0.5 * overlay).astype(np.uint8)).save(out / "labels.png")

    stages.write_json(out / "environment.json", {"voxel_size": voxel, "frames": used.tolist(), "surfaces": surfaces})
    kinds = ", ".join(f"{s['name']} ({s['material']})" for s in surfaces)
    print(f"openjd_status: Environment with {len(surfaces)} surfaces: {kinds}")


if __name__ == "__main__":
    main()
