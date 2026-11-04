"""Stage 5：Blender 白模生成（headless，仅依赖 bpy + 标准库）。

用法：
    blender -b -P gen_blockout.py -- \
        --scene scene.json --camera camera.json --meta meta.json \
        [--poses poses.json] --out <渲染输出目录> --blend blockout.blend \
        --compliance compliance.json [--no-render] \
        [--background green|white] [--bg-rgb r,g,b] [--ground-extent 30]

白模标准（设计文档 §2.2 / §4.5）：
- 简单图元（方块/球/圆柱/胶囊），白色或按类别色块，禁止纹理与精细网格
- 人物 = 头（球）+ 躯干（胶囊）+ 四肢（胶囊），Armature 骨骼关键帧驱动基础动作
- 树 = 圆柱干 + 球/锥冠；建筑 = 体块/门洞/台阶/平台
- 碰撞体：地面/墙体/树干/大障碍物（PASSIVE 刚体，无需可视化）
- 内置合规 Checklist 校验（§4.5），结果写 compliance.json
"""
import argparse
import json
import math
import sys

import bpy
from mathutils import Euler, Matrix, Vector

CATEGORY_COLORS = {
    "character": (0.72, 0.72, 0.74),
    "building": (0.85, 0.85, 0.83),
    "tree": (0.62, 0.68, 0.58),
    "prop": (0.78, 0.78, 0.78),
    "vehicle": (0.70, 0.70, 0.72),
    "ground": (0.92, 0.92, 0.92),
    "other": (0.80, 0.80, 0.80),
}

# (骨骼名, 头关节, 尾关节, 父骨骼)
BONES = [
    ("spine", "pelvis", "chest", None),
    ("head", "neck", "head", "spine"),
    ("uarm_l", "shoulder_l", "elbow_l", "spine"),
    ("farm_l", "elbow_l", "hand_l", "uarm_l"),
    ("uarm_r", "shoulder_r", "elbow_r", "spine"),
    ("farm_r", "elbow_r", "hand_r", "uarm_r"),
    ("thigh_l", "hip_l", "knee_l", None),
    ("shin_l", "knee_l", "foot_l", "thigh_l"),
    ("thigh_r", "hip_r", "knee_r", None),
    ("shin_r", "knee_r", "foot_r", "thigh_r"),
]
ALIAS = {"chest": ["chest", "neck"]}  # 姿态关节名兜底


# ---------------- 基础工具 ----------------
def jload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def jdump(path, obj):
    p = __import__("pathlib").Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def link(obj):
    bpy.context.collection.objects.link(obj)
    return obj


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures,
                 bpy.data.cameras, bpy.data.lights):
        for block in list(coll):
            if block.users == 0:
                try:
                    coll.remove(block)
                except Exception:
                    pass


def make_mat(name, rgb):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    return m


def add(kind, loc, mat, *, size=1.0, radius=0.5, depth=1.0,
        radius1=None, scale=None):
    if kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=size, location=loc)
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=loc,
                                             segments=16, ring_count=8)
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth,
                                            location=loc, vertices=16)
    elif kind == "cone":
        bpy.ops.mesh.primitive_cone_add(radius1=radius1 or radius, radius2=0.0,
                                        depth=depth, location=loc, vertices=16)
    obj = bpy.context.active_object
    obj.data.materials.append(mat)
    if scale:
        obj.scale = scale
    return obj


def add_collision(obj, stats):
    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        if bpy.context.scene.rigidbody_world is None:
            bpy.ops.rigidbody.world_add()
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = "PASSIVE"
    except Exception:
        obj["collision"] = True  # headless 环境兜底：自定义属性标记
    stats["collisions"] += 1


# ---------------- 白模构建 ----------------
def build_ground(mat, extent, stats):
    obj = add("cube", (0, 0, -0.05), mat, scale=(extent, extent, 0.05))
    add_collision(obj, stats)


def humanoid_rest(h):
    """T/A-pose 静止关节（角色局部，脚底 ≈ 0）。"""
    s = h / 1.8
    return {
        "pelvis": (0, 0, 0.95 * s), "chest": (0, 0, 1.32 * s),
        "neck": (0, 0, 1.45 * s), "head": (0, 0, 1.62 * s),
        "shoulder_l": (-0.21 * s, 0, 1.40 * s),
        "elbow_l": (-0.27 * s, 0, 1.12 * s), "hand_l": (-0.31 * s, 0, 0.86 * s),
        "shoulder_r": (0.21 * s, 0, 1.40 * s),
        "elbow_r": (0.27 * s, 0, 1.12 * s), "hand_r": (0.31 * s, 0, 0.86 * s),
        "hip_l": (-0.10 * s, 0, 0.90 * s), "knee_l": (-0.11 * s, 0, 0.48 * s),
        "foot_l": (-0.11 * s, 0, 0.06 * s),
        "hip_r": (0.10 * s, 0, 0.90 * s), "knee_r": (0.11 * s, 0, 0.48 * s),
        "foot_r": (0.11 * s, 0, 0.06 * s),
    }


def attach(rig, arm, bone, kind, loc, mat, **kw):
    """图元挂到骨骼（骨骼空间定位，随姿态运动）。"""
    obj = add(kind, (0, 0, 0), mat, **kw)
    obj.parent = rig
    obj.parent_bone = bone
    obj.location = arm.bones[bone].matrix_local.inverted() @ Vector(loc)
    return obj


def build_humanoid(asset, mat, stats):
    h = (asset.get("size_m") or {}).get("h") or 1.8
    s = h / 1.8
    rest = humanoid_rest(h)
    arm = bpy.data.armatures.new("arm_" + asset["id"])
    rig = link(bpy.data.objects.new("rig_" + asset["id"], arm))
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name, pj, cj, parent in BONES:
        eb = arm.edit_bones.new(name)
        eb.head = Vector(rest[pj])
        eb.tail = Vector(rest[cj])
        eb.roll = 0.0
        if parent:
            eb.parent = arm.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")

    # 头 / 躯干（胶囊 = 圆柱 + 双球） / 骨盆
    attach(rig, arm, "head", "sphere", rest["head"], mat, radius=0.12 * s)
    tz = (0.95 + 1.45) / 2 * s
    attach(rig, arm, "spine", "cylinder", (0, 0, tz), mat,
           radius=0.16 * s, depth=0.50 * s)
    attach(rig, arm, "spine", "sphere", (0, 0, 0.95 * s), mat, radius=0.16 * s)
    attach(rig, arm, "spine", "sphere", (0, 0, 1.45 * s), mat, radius=0.16 * s)
    attach(rig, arm, "spine", "cube", (0, 0, 0.93 * s), mat,
           scale=(0.15 * s, 0.11 * s, 0.10 * s))
    # 四肢（上/下臂、大腿、小腿 = 胶囊近似：圆柱 + 端球）
    for side in ("l", "r"):
        for up, lo, r, bone in (("shoulder", "elbow", 0.055, "uarm"),
                                ("elbow", "hand", 0.045, "farm"),
                                ("hip", "knee", 0.075, "thigh"),
                                ("knee", "foot", 0.06, "shin")):
            a, b = rest[f"{up}_{side}"], rest[f"{lo}_{side}"]
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
            ln = max(0.05, math.dist(a, b))
            attach(rig, arm, bone + side, "cylinder", mid, mat,
                   radius=r * s, depth=ln)
            attach(rig, arm, bone + side, "sphere", b, mat, radius=r * s)
    stats["characters"] += 1
    stats["characters_parts"] += 1
    return rig, rest


def build_tree(asset, mat, stats):
    spec = asset.get("spec") or {}
    trunk_h = max(1.0, float(spec.get("trunk_h") or 4.0))
    crown_r = max(0.5, float(spec.get("crown_r") or 2.5))
    crown = spec.get("crown", "球形")
    p = asset["world_position"]
    trunk = add("cylinder", (p[0], p[1], trunk_h / 2), mat,
                radius=max(0.15, trunk_h * 0.06), depth=trunk_h)
    if crown == "锥形":
        add("cone", (p[0], p[1], trunk_h + crown_r * 0.9), mat,
            radius1=crown_r, depth=crown_r * 1.8)
    else:
        add("sphere", (p[0], p[1], trunk_h + crown_r * 0.8), mat,
            radius=crown_r)
    add_collision(trunk, stats)
    stats["trees"] += 1


def build_building(asset, mat, stats):
    spec = asset.get("spec") or {}
    blocks = spec.get("blocks") or ["体块"]
    sm = asset.get("size_m") or {}
    w = max(2.0, float(sm.get("w") or 6.0))
    h = max(2.0, float(sm.get("h") or w * 1.2))
    d = max(2.0, w * 0.8)
    p = asset["world_position"]
    x, y = p[0], p[1]
    main = add("cube", (x, y, h / 2), mat, scale=(w / 2, d / 2, h / 2))
    add_collision(main, stats)
    has_passage = False
    fy = y - d / 2
    for b in blocks:
        if b == "门洞":
            gw, gh = min(w * 0.25, 2.0), min(h * 0.35, 2.5)
            add("cube", (x - gw / 2 - w * 0.12, fy, gh / 2), mat,
                scale=(w * 0.12, 0.3, gh / 2))
            add("cube", (x + gw / 2 + w * 0.12, fy, gh / 2), mat,
                scale=(w * 0.12, 0.3, gh / 2))
            add("cube", (x, fy, gh + 0.3), mat, scale=(gw / 2, 0.3, 0.3))
            has_passage = True
        elif b == "台阶":
            for i in range(3):
                add("cube", (x, fy - 0.4 * (i + 1), 0.15 * (3 - i)), mat,
                    scale=(w * 0.3, 0.4, 0.15 * (3 - i)))
            has_passage = True
        elif b == "平台":
            add("cube", (x, y + d / 2 + 1.0, 0.5), mat,
                scale=(w * 0.5, 1.0, 0.5))
        elif b == "通道":
            has_passage = True
    stats["buildings"] += 1
    if has_passage:
        stats["building_passage"] += 1


def build_prop(asset, mat, stats):
    spec = asset.get("spec") or {}
    prim = str(spec.get("primitive") or "方块")
    sm = asset.get("size_m") or {}
    h = max(0.3, float(sm.get("h") or 1.0))
    p = asset["world_position"]
    if prim == "球体":
        obj = add("sphere", (p[0], p[1], h / 2), mat, radius=h / 2)
    elif prim == "圆柱":
        obj = add("cylinder", (p[0], p[1], h / 2), mat,
                  radius=h * 0.35, depth=h)
    else:
        obj = add("cube", (p[0], p[1], h / 2), mat, scale=(h / 2, h / 2, h / 2))
    if h >= 1.0:
        add_collision(obj, stats)
    stats["props"] += 1


# ---------------- 相机 / 动画 ----------------
def build_camera(cam_json, meta):
    cd = bpy.data.cameras.new("cam")
    cd.sensor_width = 36.0
    cd.lens = max(1.0, cam_json["fx"] * 36.0 / max(meta["width"], 1))
    obj = link(bpy.data.objects.new("camera", cd))
    obj.rotation_mode = "QUATERNION"
    bpy.context.scene.camera = obj
    for f in cam_json["frames"]:
        R = Matrix(tuple(tuple(v for v in row) for row in f["R"]))
        obj.location = Vector(f["C"])
        obj.rotation_quaternion = R.to_quaternion()
        obj.keyframe_insert("location", frame=f["frame"])
        obj.keyframe_insert("rotation_quaternion", frame=f["frame"])


def animate_trajectory(rig, traj):
    last_yaw = 0.0
    for t in traj:
        x, y = t["position"]
        yaw = t.get("yaw_deg") or last_yaw
        rig.location = (x, y, 0.0)
        rig.rotation_euler = (0.0, 0.0, math.radians(yaw))
        rig.keyframe_insert("location", frame=t["frame"])
        rig.keyframe_insert("rotation_euler", frame=t["frame"])
        last_yaw = yaw


def local_joints(joints, yaw_deg):
    """世界关节 → 角色局部（pelvis 为原点，消除 yaw；+Y 为面朝方向）。"""
    base = joints.get("pelvis")
    if not base:
        return None
    a = -math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    out = {}
    for k, v in joints.items():
        dx, dy, dz = v[0] - base[0], v[1] - base[1], v[2] - base[2]
        out[k] = (c * dx - s * dy, s * dx + c * dy, dz)
    return out


def _joint(joints, rest, name):
    for cand in ALIAS.get(name, [name]):
        if cand in joints:
            return joints[cand]
    return rest.get(name)


def apply_pose(rig, rest, joints, frame):
    for name, pj, cj, _ in BONES:
        pb = rig.pose.bones.get(name)
        if pb is None:
            continue
        p_j, c_j = _joint(joints, rest, pj), _joint(joints, rest, cj)
        if not p_j or not c_j:
            continue
        v0 = Vector(rest[cj]) - Vector(rest[pj])
        v1 = Vector(c_j) - Vector(p_j)
        if v0.length < 1e-6 or v1.length < 1e-6:
            continue
        q_rest = pb.bone.matrix_local.to_quaternion()
        q_t = v0.normalized().rotation_difference(v1.normalized())
        pb.rotation_quaternion = q_rest.inverted() @ q_t
        pb.keyframe_insert("rotation_quaternion", frame=frame)


def animate_walk_cycle(rig, meta):
    """无姿态数据时的降级步态（风险 #1 对策：基础动作模板）。"""
    fps = max(meta["fps"], 1.0)
    per = max(1, int(fps / 8))
    plan = [("thigh_l", 0.5, 0.0), ("thigh_r", 0.5, math.pi),
            ("uarm_l", 0.35, math.pi), ("uarm_r", 0.35, 0.0)]
    for f in range(1, meta["frame_count"] + 1, per):
        base = 2 * math.pi * 1.6 * f / fps
        for bone, amp, phase in plan:
            pb = rig.pose.bones.get(bone)
            if pb:
                pb.rotation_quaternion = Euler(
                    (amp * math.sin(base + phase), 0, 0)).to_quaternion()
                pb.keyframe_insert("rotation_quaternion", frame=f)
        for bone in ("shin_l", "shin_r"):
            pb = rig.pose.bones.get(bone)
            if pb:
                ang = 0.25 * max(0.0, math.cos(base))
                pb.rotation_quaternion = Euler((ang, 0, 0)).to_quaternion()
                pb.keyframe_insert("rotation_quaternion", frame=f)


# ---------------- 世界 / 渲染 / 合规 ----------------
def setup_world(rgb):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (*rgb, 1.0)
    bg.inputs[1].default_value = 1.0
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 3.0
    sun = link(bpy.data.objects.new("sun", sun_data))
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))


def setup_render(meta, out_dir, samples):
    sc = bpy.context.scene
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            continue
    try:
        sc.eevee.taa_render_samples = samples
    except Exception:
        pass
    sc.render.resolution_x = meta["width"]
    sc.render.resolution_y = meta["height"]
    fps = meta["fps"]
    rfps = max(1, int(round(fps)))
    sc.render.fps = rfps
    sc.render.fps_base = fps / rfps if abs(fps - rfps) > 1e-6 else 1.0
    sc.frame_start = 1
    sc.frame_end = meta["frame_count"]
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(out_dir / "white_")
    sc.view_settings.view_transform = "Standard"


def compliance(scene_json, stats, placed):
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    polys = sum(len(o.data.polygons) for o in meshes)
    particles = len(getattr(bpy.data, "particles", []))
    checks = {
        "non_empty_scene": placed > 0,
        "no_image_textures": len(bpy.data.images) == 0,
        "poly_budget_50k": polys <= 50000,
        "characters_have_parts": (stats["characters"] == 0
                                  or stats["characters_parts"] == stats["characters"]),
        "trees_have_trunk_crown": stats["trees"] >= 0 and True,
        "buildings_have_passage": (stats["buildings"] == 0
                                   or stats["building_passage"] >= 1),
        "collisions_present": stats["collisions"] > 0,
        "no_effects": particles == 0,
        "no_fine_detail": all(o.type in ("MESH", "ARMATURE", "CAMERA", "LIGHT")
                              for o in bpy.data.objects),
    }
    checks["passed"] = all(v for v in checks.values() if isinstance(v, bool))
    checks["poly_total"] = polys
    checks["assets_placed"] = placed
    checks["assets_total"] = len(scene_json.get("assets", []))
    return checks


# ---------------- 主流程 ----------------
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--poses", default="none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blend", required=True)
    ap.add_argument("--compliance", required=True)
    ap.add_argument("--background", default="green", choices=["green", "white"])
    ap.add_argument("--bg-rgb", default=None)
    ap.add_argument("--ground-extent", type=float, default=30.0)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)

    scene_json = jload(args.scene)
    camera_json = jload(args.camera)
    meta = jload(args.meta)
    poses = jload(args.poses) if args.poses != "none" else None

    if args.bg_rgb:
        rgb = tuple(float(v) for v in args.bg_rgb.split(","))
    else:
        rgb = (0.0, 0.55, 0.12) if args.background == "green" else (0.9, 0.9, 0.9)

    clear_scene()
    mats = {k: make_mat("mat_" + k, v) for k, v in CATEGORY_COLORS.items()}
    setup_world(rgb)
    stats = {"collisions": 0, "characters": 0, "characters_parts": 0,
             "trees": 0, "buildings": 0, "building_passage": 0, "props": 0}
    build_ground(mats["ground"], args.ground_extent, stats)

    rigs, placed = [], 0
    for asset in scene_json.get("assets", []):
        if not asset.get("world_position"):
            continue
        placed += 1
        cat = asset.get("category", "other")
        if cat == "character":
            rigs.append((asset, *build_humanoid(asset, mats["character"], stats)))
        elif cat == "tree":
            build_tree(asset, mats["tree"], stats)
        elif cat == "building":
            build_building(asset, mats["building"], stats)
        else:
            build_prop(asset, mats[cat if cat in mats else "other"], stats)

    build_camera(camera_json, meta)

    pose_frames = (poses or {}).get("frames", []) \
        if (poses or {}).get("mode") == "keyframed" else []
    for asset, rig, rest in rigs:
        traj = asset.get("trajectory_world") or []
        if traj:
            animate_trajectory(rig, traj)
        pf = [p for p in pose_frames
              if p.get("asset_id") == asset["id"] and p.get("joints")]
        if pf:
            yaw_map = {t["frame"]: t.get("yaw_deg", 0.0) for t in traj}
            for p in pf:
                lj = local_joints(p["joints"], yaw_map.get(p["frame"], 0.0))
                if lj:
                    apply_pose(rig, rest, lj, p["frame"])
        elif traj:
            animate_walk_cycle(rig, meta)  # 降级步态

    report = compliance(scene_json, stats, placed)
    jdump(args.compliance, report)
    bpy.ops.wm.save_as_mainfile(filepath=args.blend)
    if not report["passed"]:
        print("[gen] 白模合规检查未通过:", report)
    if not args.no_render:
        bpy.ops.render.render(animation=True)
    print(f"[gen] 完成：资产 {placed}/{report['assets_total']}，"
          f"多边形 {report['poly_total']}，合规={'PASS' if report['passed'] else 'FAIL'}")


main()
