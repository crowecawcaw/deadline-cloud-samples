"""Register stage: place each generated mesh in the scene and track the objects that move.

Inputs (analyze stage):  cameras.json, depth.npz, objects.json, masks/obj_NN.npz
Inputs (reconstruct):    obj_NN/mesh.ply, Z up, centered near the origin
Outputs (register):      scene.json, everything in Blender's world frame (Z up, meters, ground at z=0)
                         debug/obj_NN_points.ply, the observed points used for each fit

The fit is deliberately simple: objects are assumed upright, so each mesh gets a yaw rotation,
a uniform scale, and a translation, found with ICP from the observed depth points to the mesh
surface. Static objects are fitted to points pooled over many frames. Moving objects are fitted
on their best frame, then tracked frame to frame starting from the previous pose.
"""

import argparse
import os

import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

import stages

CV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def load_masks(path):
    data = np.load(path)
    return np.unpackbits(data["packed"], axis=-1, count=int(data["width"])).astype(bool)


def unproject(depth, mask, K, cam_to_world, max_points=4000, rng=None):
    """World-space points for the masked pixels with valid depth."""
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if len(xs) == 0:
        return np.zeros((0, 3))
    if rng is not None and len(xs) > max_points:
        keep = rng.choice(len(xs), max_points, replace=False)
        xs, ys = xs[keep], ys[keep]
    z = depth[ys, xs]
    x = (xs + 0.5 - K[0][2]) / K[0][0] * z
    y = (ys + 0.5 - K[1][2]) / K[1][1] * z
    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (cam_to_world @ cam.T).T[:, :3]


def fit_ground(points, cam_centers, object_points):
    """Return a 4x4 transform taking world points to a frame with the ground at z=0, Z up.

    The ground is the dominant plane among background points near the objects, which is the
    surface they rest on, rather than a distant floor or wall.
    """
    if len(object_points):
        center = np.median(object_points, axis=0)
        radius = 3.0 * np.quantile(np.linalg.norm(object_points - center, axis=1), 0.9)
        near = np.linalg.norm(points - center, axis=1) < radius
        if near.sum() > 500:
            points = points[near]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # Robust extent, so outliers don't loosen the threshold.
    extent = np.linalg.norm(np.quantile(points, 0.95, axis=0) - np.quantile(points, 0.05, axis=0))
    threshold = 0.005 * extent
    plane, inliers = pcd.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=5000)
    normal, d = np.array(plane[:3]), plane[3]
    scale = np.linalg.norm(normal)
    normal, d = normal / scale, d / scale
    # Least-squares refit on the inliers.
    inliers = np.abs(points @ normal + d) < threshold
    centroid = points[inliers].mean(0)
    normal = np.linalg.svd(points[inliers] - centroid, full_matrices=False)[2][2]
    d = -normal @ centroid
    inliers = np.nonzero(np.abs(points @ normal + d) < threshold)[0]
    # Point the normal toward the cameras so they end up above the ground.
    if np.mean(cam_centers @ normal + d) < 0:
        normal, d = -normal, -d

    z = normal
    x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows are the new axes
    G = np.eye(4)
    G[:3, :3] = R
    G[:3, 3] = R @ (-d * normal)  # the plane point closest to the origin maps to z=0
    ground_points = points[inliers] @ R.T + G[:3, 3]
    return G, ground_points


def yaw_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def solve_upright(src, dst, scale=None, z=None):
    """Least squares dst ~ s * Rz(theta) @ src + t, rotating only about Z.

    Pass scale to hold the scale fixed, and z to hold the vertical translation fixed.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    num = np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    den = np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])
    theta = np.arctan2(num, den)
    R = yaw_matrix(theta)
    s = scale if scale is not None else np.sum((a @ R.T) * b) / max(np.sum(a * a), 1e-12)
    t = mu_d - s * R @ mu_s
    if z is not None:
        t[2] = z
    return s, theta, t


def to_matrix(s, theta, t):
    M = np.eye(4)
    M[:3, :3] = s * yaw_matrix(theta)
    M[:3, 3] = t
    return M


def residuals(mesh_tree, observed, s, theta, t):
    M_inv = np.linalg.inv(to_matrix(s, theta, t))
    return mesh_tree.query(observed @ M_inv[:3, :3].T + M_inv[:3, 3])


def icp_upright(mesh_samples, mesh_tree, observed, s, theta, t, iterations=30, fix_scale=False, fix_z=False):
    """Fit observed points to the mesh surface. Returns (s, theta, t, rms) for mesh->world."""
    trim = 0.8
    for _ in range(iterations):
        dist, idx = residuals(mesh_tree, observed, s, theta, t)
        keep = dist <= np.quantile(dist, trim)  # trim outliers from mask edges and depth noise
        s_new, theta_new, t_new = solve_upright(
            mesh_samples[idx[keep]], observed[keep], s if fix_scale else None, t[2] if fix_z else None
        )
        converged = abs(s_new - s) < 1e-4 * s and abs(theta_new - theta) < 1e-4 and np.allclose(t_new, t, atol=1e-5)
        s, theta, t = s_new, theta_new, t_new
        if converged:
            break
    dist, _ = residuals(mesh_tree, observed, s, theta, t)
    dist = dist * s
    return s, theta, t, float(np.sqrt(np.mean(np.minimum(dist, np.quantile(dist, trim)) ** 2)))


class SilhouetteCheck:
    """Fraction of mesh points that project outside the object's mask where they'd be visible.

    Points on the visible surface constrain a pose only up to symmetries like a mug rotating
    about its axis. Checking the mesh against the masks in other frames breaks those ties.
    """

    def __init__(self, masks, depth, Ks, c2w, frames, dilate=3):
        from scipy.ndimage import binary_dilation

        self.views = []
        for i in frames:
            self.views.append(
                (binary_dilation(masks[i], iterations=dilate), depth[i], Ks[i], np.linalg.inv(c2w[i]))
            )

    def __call__(self, world_points):
        outside, total = 0, 0
        for mask, depth, K, w2c in self.views:
            cam = world_points @ w2c[:3, :3].T + w2c[:3, 3]
            z = cam[:, 2]
            front = z > 1e-3
            u = np.floor(cam[front, 0] / z[front] * K[0][0] + K[0][2]).astype(int)
            v = np.floor(cam[front, 1] / z[front] * K[1][1] + K[1][2]).astype(int)
            h, w = mask.shape
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u, v, zf = u[inside], v[inside], z[front][inside]
            observed = depth[v, u]
            # Only count points that would be visible: not behind something else in the depth map.
            visible = ~(observed > 0) | (zf <= observed * 1.03)
            outside += np.sum(visible & ~mask[v, u])
            total += np.sum(visible)
        return outside / max(total, 1)


def initial_fit(mesh_samples, mesh_tree, observed, view_dir, silhouette=None):
    """Search over yaw angles from an initial guess based on the observed height and position.

    Objects whose observed points reach down to the ground are treated as resting on it: their
    scale comes from their height and their bottom stays at z=0, leaving yaw and horizontal
    position to fit. Partial views constrain those far better than a free scale.
    """
    mesh_min, mesh_max = mesh_samples.min(0), mesh_samples.max(0)
    mesh_height = max(mesh_max[2] - mesh_min[2], 1e-6)
    obs_top = np.quantile(observed[:, 2], 0.98)
    obs_bottom = np.quantile(observed[:, 2], 0.02)
    # From above, the bottom of an object resting on the ground is hidden, so allow a gap.
    grounded = obs_top > 0 and obs_bottom < 0.5 * obs_top
    s0 = (obs_top if grounded else obs_top - obs_bottom) / mesh_height
    z0 = -s0 * mesh_min[2] if grounded else obs_bottom - s0 * mesh_min[2]
    # The camera sees the front surface, so the center is somewhere behind the observed points.
    radius = s0 * 0.25 * np.linalg.norm((mesh_max - mesh_min)[:2])
    side = np.cross([0.0, 0.0, 1.0], view_dir)
    observed_center = np.median(observed, axis=0)
    centers = [observed_center + radius * (a * view_dir + b * side) for a in (0.0, 1.0, 2.0) for b in (-1.0, 0.0, 1.0)]
    mesh_center = np.array([*((mesh_min + mesh_max) / 2)[:2], 0.0])
    check_points = mesh_samples[:: max(1, len(mesh_samples) // 2000)]
    best, best_cost = None, np.inf
    for theta, center in ((th, c) for th in np.linspace(0, 2 * np.pi, 24, endpoint=False) for c in centers):
        t = np.array([center[0], center[1], z0]) - s0 * yaw_matrix(theta) @ mesh_center
        t[2] = z0
        result = icp_upright(mesh_samples, mesh_tree, observed, s0, theta, t, fix_scale=grounded, fix_z=grounded)
        s, th, tt, rms = result
        cost = rms / (s * mesh_height)
        if silhouette is not None:
            M = to_matrix(s, th, tt)
            cost += 0.2 * silhouette(check_points @ M[:3, :3].T + M[:3, 3])
        if os.environ.get("REGISTER_DEBUG"):
            print(f"  yaw0={np.degrees(theta):4.0f} -> yaw={np.degrees(th):6.1f} rms={rms:.4f} cost={cost:.4f}")
        if cost < best_cost:
            best, best_cost = result, cost
    return (*best, grounded)


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    rng = np.random.default_rng(0)

    analyze = stages.input_dir("analyze")
    reconstruct = stages.input_dir("reconstruct")
    out = stages.output_dir("register")
    cameras = stages.read_json(analyze / "cameras.json")
    objects = stages.read_json(analyze / "objects.json")["objects"]
    depth = np.load(analyze / "depth.npz")["depth"].astype(np.float32)
    frames = cameras["frames"]
    num_frames = len(frames)
    c2w = [np.array(f["cam_to_world"]) for f in frames]
    Ks = [np.array(f["K"]) for f in frames]

    masks = {o["id"]: load_masks(analyze / "masks" / f"obj_{o['id']:02d}.npz") for o in objects}
    any_object = np.zeros_like(next(iter(masks.values()))) if masks else np.zeros(depth.shape, bool)
    for m in masks.values():
        any_object |= m

    # Ground plane from background pixels of a handful of frames
    sample_frames = np.linspace(0, num_frames - 1, min(8, num_frames)).astype(int)
    background = np.concatenate(
        [unproject(depth[i], ~any_object[i], Ks[i], c2w[i], 20000, rng) for i in sample_frames]
    )
    object_points = np.concatenate(
        [unproject(depth[i], any_object[i], Ks[i], c2w[i], 5000, rng) for i in sample_frames]
    )
    cam_centers = np.array([m[:3, 3] for m in c2w])
    G, ground_points = fit_ground(background, cam_centers, object_points)
    c2w = [G @ m for m in c2w]
    ground_size = float(2 * np.max(np.abs(ground_points[:, :2])) + 0.5)

    scene_objects, report = [], []
    for obj in objects:
        oid = obj["id"]
        mesh_path = reconstruct / f"obj_{oid:02d}" / "mesh.ply"
        if not mesh_path.exists():
            print(f"WARNING: No mesh for object {oid} ({obj['label']}), skipping")
            continue
        mesh = trimesh.load(mesh_path, process=False)
        mesh_samples = np.asarray(mesh.sample(20000))
        mesh_tree = cKDTree(mesh_samples)

        eroded = np.stack([binary_erosion(m, iterations=2) for m in masks[oid]])
        per_frame = [unproject(depth[i], eroded[i], Ks[i], c2w[i], 3000, rng) for i in range(num_frames)]
        key = int(obj.get("crop_frame", int(np.argmax([len(p) for p in per_frame]))))
        if obj["motion"] == "static":
            observed = np.concatenate([per_frame[i] for i in sample_frames if len(per_frame[i])])
        else:
            observed = per_frame[key]
        if len(observed) > 6000:
            observed = observed[rng.choice(len(observed), 6000, replace=False)]
        if len(observed) < 50:
            print(f"WARNING: Too few depth points for object {oid} ({obj['label']}), skipping")
            continue
        (out / "debug").mkdir(exist_ok=True)
        trimesh.PointCloud(observed).export(out / "debug" / f"obj_{oid:02d}_points.ply")

        view_dir = observed.mean(0) - c2w[key][:3, 3]
        view_dir[2] = 0
        view_dir /= max(np.linalg.norm(view_dir), 1e-9)
        check_frames = sample_frames if obj["motion"] == "static" else [key]
        silhouette = SilhouetteCheck(masks[oid], depth, Ks, c2w, check_frames)
        s, theta, t, rms, grounded = initial_fit(mesh_samples, mesh_tree, observed, view_dir, silhouette)
        print(f"Object {oid} ({obj['label']}): scale={s:.3f} yaw={np.degrees(theta):.0f} rms={rms:.4f} grounded={grounded}")

        if obj["motion"] == "static":
            matrices = [to_matrix(s, theta, t)]
        else:
            # Track outward from the key frame in both directions.
            poses = {key: (theta, t)}
            for order in (range(key + 1, num_frames), range(key - 1, -1, -1)):
                prev_theta, prev_t = theta, t
                for i in order:
                    if len(per_frame[i]) >= 50:
                        _, prev_theta, prev_t, _ = icp_upright(
                            mesh_samples, mesh_tree, per_frame[i], s, prev_theta, prev_t, 15,
                            fix_scale=True, fix_z=grounded,
                        )
                    poses[i] = (prev_theta, prev_t)
            matrices = [to_matrix(s, *poses[i]) for i in range(num_frames)]

        scene_objects.append(
            {
                "id": oid,
                "label": obj["label"],
                "motion": obj["motion"],
                "mesh": f"obj_{oid:02d}/mesh.ply",
                "matrices": [m.tolist() for m in matrices],
            }
        )
        report.append(
            {"id": oid, "label": obj["label"], "scale": s, "yaw_deg": float(np.degrees(theta)), "rms": rms,
             "grounded": bool(grounded), "points": len(observed)}
        )

    stages.write_json(
        out / "scene.json",
        {
            "fps": cameras.get("fps", 24),
            "width": cameras["width"],
            "height": cameras["height"],
            "ground": {"size": ground_size},
            # Maps the analyze stage's camera frame to this world frame, for debugging.
            "input_to_world": G.tolist(),
            "camera": {
                "fx": Ks[0][0][0],
                "cx": Ks[0][0][2],
                "cy": Ks[0][1][2],
                "cam_to_world": [(m @ CV_TO_BLENDER_CAMERA).tolist() for m in c2w],
            },
            "objects": scene_objects,
        },
    )
    stages.write_json(out / "report.json", report)
    print(f"openjd_status: Registered {len(scene_objects)} of {len(objects)} objects")


if __name__ == "__main__":
    main()
