"""ExportUSD stage: write a simulation-ready USD asset per object and a scene that references them.

Inputs (reconstruct): obj_NN/mesh.ply
Inputs (physics):     scene.json, obj_NN/collision.npz
Inputs (environment): environment.json, surfaces/<name>.ply (optional)
Outputs (usd):        objects/obj_NN_<label>.usd   One rigid body per file, meters, Z up
                      scene.usda                   Ground, physics scene, camera, lights, and every object

Uses the UsdPhysics schemas that Isaac Sim and other Omniverse apps load directly:
  - RigidBodyAPI and MassAPI (mass, center of mass, principal inertia) on the object's root prim
  - A visual mesh with vertex colors, and convex collision parts (CollisionAPI with
    MeshCollisionAPI approximation convexHull) in a "collision" scope that isn't rendered
  - A physics material with friction and restitution, bound with the "physics" purpose

Environment surfaces are static colliders that use their triangle mesh directly (approximation
"none"), each with its own physics material. The ground box stays under them as an invisible
collider, in case the reconstructed floor has holes.

In the scene, objects that stayed put are dynamic bodies at their registered pose. Objects that
moved in the video are kinematic bodies that replay the tracked motion as time samples.
"""

import re

import numpy as np
import trimesh
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

import stages

# Lift objects off the ground slightly so their collision shapes don't start interpenetrating.
CLEARANCE = 0.002


def setup_stage(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)


def physics_material(stage, path, props):
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(props["static_friction"])
    api.CreateDynamicFrictionAttr(props["dynamic_friction"])
    api.CreateRestitutionAttr(props["restitution"])
    api.CreateDensityAttr(props["density"])
    return material


def bind_physics_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def vertex_color_material(stage, path):
    """UsdPreviewSurface that reads the displayColor primvar, so vertex colors render in RTX."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    reader = UsdShade.Shader.Define(stage, f"{path}/DisplayColor")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def define_mesh(stage, path, vertices, faces, colors=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = np.min(vertices, 0), np.max(vertices, 0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    if colors is not None:
        primvar = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec3fArray.FromNumpy(np.asarray(colors, np.float32)))
    return mesh


def write_object_asset(path, name, mesh, parts, props):
    """One object as a standalone rigid body asset, in its local frame at metric scale."""
    stage = Usd.Stage.CreateNew(str(path))
    setup_stage(stage)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(props["mass_kg"])
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*props["center_of_mass"]))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*props["diagonal_inertia"]))
    axes = Gf.Matrix3d(*np.asarray(props["principal_axes"]).T.ravel())  # Gf rows = principal axes
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(axes.ExtractRotation().GetQuat()))
    for key in ("identified_as", "material", "hollow", "mass_check", "collision_method"):
        root.GetPrim().SetCustomDataByKey(f"oneVideoToScene:{key}", props[key])

    looks = f"/{name}/Looks"
    visual = define_mesh(stage, f"/{name}/visual", mesh.vertices, mesh.faces,
                         mesh.visual.vertex_colors[:, :3] / 255.0)
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(vertex_color_material(stage, f"{looks}/visual"))

    collision = UsdGeom.Scope.Define(stage, f"/{name}/collision")
    collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
    for i, part in enumerate(parts):
        part_mesh = define_mesh(stage, f"/{name}/collision/part_{i:02d}", part.vertices, part.faces)
        UsdPhysics.CollisionAPI.Apply(part_mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(part_mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    bind_physics_material(root.GetPrim(), physics_material(stage, f"{looks}/physics", props))
    stage.GetRootLayer().Save()


def rigid_pose(matrix):
    """Translation and rotation of a 4x4 matrix with uniform scale, as (Vec3d, Quatf)."""
    M = np.array(matrix)
    R = M[:3, :3] / np.linalg.norm(M[:3, 0])
    rotation = Gf.Matrix3d(*R.T.ravel()).ExtractRotation().GetQuat()  # Gf uses row vectors
    return Gf.Vec3d(*M[:3, 3]), Gf.Quatf(rotation)


def main():
    scene_desc = stages.read_json(stages.input_dir("physics") / "scene.json")
    physics_dir = stages.input_dir("physics")
    reconstruct = stages.input_dir("reconstruct")
    out = stages.output_dir("usd")
    (out / "objects").mkdir(exist_ok=True)

    num_frames = len(scene_desc["camera"]["cam_to_world"])
    stage = Usd.Stage.CreateNew(str(out / "scene.usda"))
    setup_stage(stage)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(num_frames - 1)
    stage.SetTimeCodesPerSecond(scene_desc.get("fps", 24))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    # Ground: a thin box with its top face at z=0.
    size = scene_desc["ground"]["size"]
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01))
    ground.AddScaleOp().Set(Gf.Vec3f(size, size, 0.02))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    ground_props = {"static_friction": 0.6, "dynamic_friction": 0.5, "restitution": 0.2, "density": 1000}
    bind_physics_material(ground.GetPrim(), physics_material(stage, "/World/Looks/ground_physics", ground_props))

    try:
        env_dir = stages.input_dir("environment")
        surfaces = stages.read_json(env_dir / "environment.json")["surfaces"]
    except FileNotFoundError:
        surfaces = []
    if surfaces:
        UsdGeom.Scope.Define(stage, "/World/Environment")
        ground.MakeInvisible()
    for surface in surfaces:
        env_mesh = trimesh.load(env_dir / surface["mesh"], process=False)
        name = re.sub(r"[^A-Za-z0-9_]", "_", surface["name"])
        colors = env_mesh.visual.vertex_colors[:, :3] / 255.0 if env_mesh.visual.kind == "vertex" else None
        prim = define_mesh(stage, f"/World/Environment/{name}", env_mesh.vertices, env_mesh.faces, colors)
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(
            vertex_color_material(stage, f"/World/Environment/Looks/{name}_visual")
        )
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
        bind_physics_material(prim.GetPrim(), physics_material(stage, f"/World/Environment/Looks/{name}_physics", surface))
        for key in ("kind", "identified_as", "material"):
            prim.GetPrim().SetCustomDataByKey(f"oneVideoToScene:{key}", surface[key])
        print(f"Environment {name}: {surface['identified_as']} ({surface['material']}), {len(env_mesh.faces)} faces")

    UsdGeom.Scope.Define(stage, "/World/Objects")
    for desc in scene_desc["objects"]:
        props = desc["physics"]
        name = f"obj_{desc['id']:02d}_{re.sub(r'[^A-Za-z0-9_]', '_', desc['label'])}"
        mesh = trimesh.load(reconstruct / desc["mesh"], process=False)
        mesh.apply_scale(props["mesh_scale"])
        data = np.load(physics_dir / props["collision"])
        parts = [trimesh.Trimesh(data[f"vertices_{i}"], data[f"faces_{i}"]) for i in range(int(data["count"]))]
        asset = f"objects/{name}.usd"
        write_object_asset(out / asset, name, mesh, parts, props)

        prim = UsdGeom.Xform.Define(stage, f"/World/Objects/{name}")
        prim.GetPrim().GetReferences().AddReference(f"./{asset}")
        translate, orient = prim.AddTranslateOp(), prim.AddOrientOp()
        if desc["motion"] == "static":
            t, q = rigid_pose(desc["matrices"][0])
            translate.Set(t + Gf.Vec3d(0, 0, CLEARANCE))
            orient.Set(q)
        else:
            for frame, matrix in enumerate(desc["matrices"]):
                t, q = rigid_pose(matrix)
                translate.Set(t + Gf.Vec3d(0, 0, CLEARANCE), frame)
                orient.Set(q, frame)
            UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim()).CreateKinematicEnabledAttr(True)
        print(f"{name}: {props['mass_kg']:.3f} kg {props['material']}, {props['collision_parts']} collision parts, "
              f"{desc['motion']}")

    # Camera following the recovered path. cam_to_world is already in the USD/Blender camera
    # convention (looking down -Z, Y up). Apertures are in millimeters, matching a 36 mm sensor.
    cam_desc = scene_desc["camera"]
    width, height = scene_desc["width"], scene_desc["height"]
    camera = UsdGeom.Camera.Define(stage, "/World/VideoCamera")
    aperture = 36.0
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture * height / width)
    camera.CreateFocalLengthAttr(cam_desc["fx"] * aperture / width)
    camera.CreateHorizontalApertureOffsetAttr((width / 2 - cam_desc["cx"]) / width * aperture)
    camera.CreateVerticalApertureOffsetAttr((cam_desc["cy"] - height / 2) / width * aperture)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    transform = camera.AddTransformOp()
    for frame, matrix in enumerate(cam_desc["cam_to_world"]):
        transform.Set(Gf.Matrix4d(*np.array(matrix).T.ravel()), frame)

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(3000)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(40, 10, 30))
    UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(500)

    stage.GetRootLayer().Save()
    print(f"openjd_status: Wrote USD for {len(scene_desc['objects'])} objects")


if __name__ == "__main__":
    main()
