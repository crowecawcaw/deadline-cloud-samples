"""Reconstruct stage: generate a mesh with vertex colors for one object using TRELLIS (MIT).

Inputs (analyze):   objects.json, crops/obj_NN.png
Outputs:            reconstruct/obj_NN/mesh.ply   Z up, centered, vertex colors (used by later stages)
                    reconstruct/obj_NN/mesh.glb   Y up, for previewing in any glTF viewer

This uses only the TRELLIS mesh decoder and takes the colors from its vertex attributes. That
avoids the texture-baking path, which needs nvdiffrast and a Gaussian rasterizer that have
non-commercial licenses.
"""

import argparse
import os
import sys
import types

import numpy as np

import stages

# Set before importing TRELLIS. Sparse attention supports only xformers or flash-attn.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")


def stub_unused_imports():
    """TRELLIS imports kaolin (only for tensor shape checks) and rembg (only to remove backgrounds,
    which we skip by passing RGBA crops) at module import. Stub them to avoid installing them."""
    kaolin, utils, testing = (types.ModuleType(n) for n in ("kaolin", "kaolin.utils", "kaolin.utils.testing"))
    testing.check_tensor = lambda *args, **kwargs: True
    kaolin.utils, utils.testing = utils, testing
    sys.modules.setdefault("kaolin", kaolin)
    sys.modules.setdefault("kaolin.utils", utils)
    sys.modules.setdefault("kaolin.utils.testing", testing)
    sys.modules.setdefault("rembg", types.ModuleType("rembg"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-number", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-faces", type=int, default=60000)
    parser.add_argument("--attempts", type=int, default=4, help="Seeds to try when the shape looks wrong.")
    args = parser.parse_args()

    analyze = stages.input_dir("analyze")
    objects = stages.read_json(analyze / "objects.json")["objects"]
    if args.object_number > len(objects):
        print(f"openjd_status: Only {len(objects)} objects, nothing to do for object {args.object_number}")
        return
    obj = objects[args.object_number - 1]
    if "crop" not in obj:
        print(f"openjd_status: Object {obj['id']} ({obj['label']}) has no crop, skipping")
        return

    stub_unused_imports()
    import open3d as o3d
    import trimesh
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline

    print(f"openjd_status: Generating mesh for object {obj['id']}: {obj['label']}")
    image = Image.open(analyze / obj["crop"]).convert("RGBA")
    alpha = np.asarray(image)[:, :, 3] > 127
    ys, xs = np.nonzero(alpha)
    crop_aspect = (np.ptp(ys) + 1) / (np.ptp(xs) + 1)
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    out = stages.output_dir("reconstruct") / f"obj_{obj['id']:02d}"
    out.mkdir(parents=True, exist_ok=True)

    # TRELLIS sometimes returns a flat sheet or a stretched shape, especially for small crops. Try a
    # few seeds and keep the mesh whose height-to-width ratio best matches the crop's silhouette.
    best = None
    for attempt in range(args.attempts):
        seed = args.seed + attempt
        mesh = pipeline.run(image, seed=seed, formats=["mesh"])["mesh"][0]
        vertices = mesh.vertices.detach().cpu().numpy().astype(np.float64)
        faces = mesh.faces.detach().cpu().numpy()
        colors = mesh.vertex_attrs[:, :3].clamp(0, 1).detach().cpu().numpy()
        if attempt == 0:
            trimesh.Trimesh(vertices, faces, vertex_colors=(colors * 255).astype(np.uint8), process=False).export(
                out / "raw_mesh.ply"
            )
        candidate = clean_mesh(o3d, trimesh, vertices, faces, colors, args.max_faces)
        extent = np.ptp(candidate.vertices, axis=0)
        mesh_aspect = extent[2] / max(extent[0], extent[1], 1e-6)
        mismatch = abs(np.log(max(mesh_aspect, 1e-3) / crop_aspect))
        print(f"Seed {seed}: extent {np.round(extent, 3)}, height/width {mesh_aspect:.2f} vs crop {crop_aspect:.2f}")
        if best is None or mismatch < best[0]:
            best = (mismatch, seed, candidate)
        if mismatch < np.log(1.3):
            break

    mismatch, seed, out_mesh = best
    if mismatch > np.log(2.0):
        print(f"openjd_status: No plausible mesh for {obj['label']} after {args.attempts} seeds, skipping it")
        image.save(out / "input.png")
        return
    # TRELLIS meshes are Z up, which matches Blender and the register stage.
    out_mesh.export(out / "mesh.ply")
    z_up_to_y_up = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    out_mesh.copy().apply_transform(z_up_to_y_up).export(out / "mesh.glb")
    image.save(out / "input.png")
    print(f"openjd_status: Wrote mesh for {obj['label']} (seed {seed}) with {len(out_mesh.faces)} faces")


def clean_mesh(o3d, trimesh, vertices, faces, colors, max_faces):
    """Decimate, keep the main connected piece, and center the mesh."""
    o3d_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_degenerate_triangles()
    if len(o3d_mesh.triangles) > max_faces:
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(max_faces)
    # Keep the largest connected piece to drop floating fragments, ignoring flat sheets that span
    # the whole generation volume.
    clusters, counts, _ = o3d_mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    tri_vertices = np.asarray(o3d_mesh.vertices)[np.asarray(o3d_mesh.triangles)]
    keep = int(np.argmax(counts))
    for c in np.argsort(-counts)[:10]:
        extent = np.ptp(tri_vertices[clusters == c].reshape(-1, 3), axis=0)
        if not (np.sum(extent > 0.95) >= 2 and extent.min() < 0.05):
            keep = int(c)
            break
    o3d_mesh.remove_triangles_by_mask(clusters != keep)
    o3d_mesh.remove_unreferenced_vertices()
    vertices = np.asarray(o3d_mesh.vertices)
    vertices = vertices - (vertices.min(0) + vertices.max(0)) / 2
    colors_u8 = (np.asarray(o3d_mesh.vertex_colors) * 255).round().astype(np.uint8)
    return trimesh.Trimesh(vertices, np.asarray(o3d_mesh.triangles), vertex_colors=colors_u8, process=False)


if __name__ == "__main__":
    main()
