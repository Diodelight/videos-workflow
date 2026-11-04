"""Stage 4b：尺度恢复 + 地面拟合 + 世界坐标融合（设计文档 §4.4）。

流程：
1. 地面拟合：稀疏点云 PCA（轴假设搜索，SfM 世界 up 任意）
2. 对齐：地面法线 → +Z（Blender Z-up），地面 z=0；相机/点云同步变换
3. 尺度恢复：人物语义尺寸（1.5–2.1m）作为米制锚点（§4.4 尺度锚点）
4. 资产落位：bbox 底边中心射线 ∩ 地面 → 世界坐标；人物轨迹/朝向
5. 姿态提升：2D 关键帧骨架 × 相机 × 人物距离 → 世界 3D 关节（poses.json）
6. 回写 camera.json（convention=blender）与 scene.json（世界字段）

用法：python -m src.sfm.scale_ground --clip <clip_id>
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from src.common.config import load_config
from src.common.paths import clip_paths
from src.common.schema import dump_json, load_json
from src.vision.depth import load_depth

COCO_MAP = {0: "head", 5: "shoulder_l", 6: "shoulder_r",
            7: "elbow_l", 8: "elbow_r", 9: "hand_l", 10: "hand_r",
            11: "hip_l", 12: "hip_r", 13: "knee_l", 14: "knee_r",
            15: "foot_l", 16: "foot_r"}


# ---------------- 几何工具 ----------------
def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotation_from_to(a, b) -> np.ndarray:
    a = np.asarray(a, float); a /= (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, float); b /= (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        v = np.cross(a, axis); v /= (np.linalg.norm(v) + 1e-12)
        K = _skew(v)
        return np.eye(3) + 2 * K @ K
    K = _skew(v / np.linalg.norm(v))
    ang = math.atan2(np.linalg.norm(v), c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def pca_plane(pts: np.ndarray):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, -float(n @ c)


def fit_ground(points: np.ndarray):
    """三个轴假设取最低 1/3 点 PCA 拟合，内点最多者胜出。"""
    if len(points) < 30:
        return None
    thresh = max(1e-3, float(np.ptp(points, axis=0).max()) * 0.02)
    best = None
    for axis in range(3):
        idx = np.argsort(points[:, axis])
        low = points[idx[: max(10, len(points) // 3)]]
        n, d = pca_plane(low)
        if n[axis] < 0:
            n, d = -n, -d
        inl = int(np.sum(np.abs(points @ n + d) < thresh))
        if best is None or inl > best[0]:
            best = (inl, n, d)
    _, n, d = best
    return {"normal": n.tolist(), "d": d, "inliers": best[0],
            "total": len(points)}


def load_points3D(sfm_dir) -> np.ndarray | None:
    sfm_dir = __import__("pathlib").Path(sfm_dir)
    hits = list(sfm_dir.rglob("points3D.txt"))
    if not hits:
        return None
    pts = []
    for line in hits[0].read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            p = line.split()
            pts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(pts) if pts else None


# ---------------- 相机投影（blender 约定：cam -Z 前向） ----------------
def pixel_dir(u, v, R_c2w, K) -> np.ndarray:
    d_cam = np.array([(u - K[0, 2]) / K[0, 0],
                      -(v - K[1, 2]) / K[1, 1], -1.0])
    d_cam /= (np.linalg.norm(d_cam) + 1e-12)
    return R_c2w @ d_cam


def project(P, R_c2w, C, K):
    pc = R_c2w.T @ (np.asarray(P, float) - C)
    z = -pc[2]
    if z <= 1e-6:
        return None
    return (K[0, 0] * pc[0] / z + K[0, 2],
            -K[1, 1] * pc[1] / z + K[1, 2], z)


def ray_ground(C, d, z_ground=0.0, default_dist=None):
    if abs(d[2]) > 1e-6:
        t = (z_ground - C[2]) / d[2]
        if t > 0:
            return C + t * d
    if default_dist:
        return C + default_dist * d
    return None


# ---------------- 主流程 ----------------
def run(clip_id: str, cfg) -> None:
    paths = clip_paths(clip_id, cfg).ensure()
    cam = load_json(paths.camera_json)
    scene = load_json(paths.scene_json)
    meta = load_json(paths.meta)
    if not (cam and scene and meta):
        raise RuntimeError("缺少 camera.json/scene.json/meta.json")

    W, H = meta["width"], meta["height"]
    K = np.array([[cam["fx"], 0, cam["cx"]],
                  [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    frames = {f["frame"]: f for f in cam["frames"]}
    pts = load_points3D(paths.sfm_dir)

    # 1. 地面拟合 + 对齐（colmap 空间 → z-up、地面 z=0）
    ground_method = "default_z0"
    R_al, shift = np.eye(3), np.zeros(3)
    if pts is not None and len(pts) >= 30:
        plane = fit_ground(pts)
        if plane:
            R_al = rotation_from_to(plane["normal"], [0, 0, 1.0])
            shift = np.array([0.0, 0.0, plane["d"]])
            pts = pts @ R_al.T + shift
            ground_method = "sparse_pca"
    elif cam.get("convention") == "colmap":
        ground_method = "sfm_no_points"

    for f in cam["frames"]:
        R = np.asarray(f["R"], float)
        C = np.asarray(f["C"], float)
        if cam.get("convention") == "colmap":
            C = R_al @ C + shift
            R = R_al @ R
            # blender 相机：X 右 / Y 上 / -Z 前向
            R = R @ np.diag([1.0, -1.0, -1.0])
        f["R"], f["C"] = R.tolist(), C.tolist()
    cam["convention"] = "blender"

    # 2. 尺度恢复（人物锚点）
    scale, scale_conf = 1.0, 0.1
    lo, hi = cfg.get("camera.scale_person_height_range", [1.5, 2.1])
    chars = [a for a in scene["assets"] if a["category"] == "character"
             and (a.get("size_m") or {}).get("h") and lo <= a["size_m"]["h"] <= hi]
    if pts is not None and chars:
        heights = []
        for a in chars[:4]:
            hs = _person_height_sfm(pts, a, frames, K, W, H)
            if hs:
                heights.append(hs)
        if heights:
            ratios = [a["size_m"]["h"] / h for a, h in zip(chars[:len(heights)], heights)]
            scale = float(np.clip(np.median(ratios), 0.05, 100.0))
            scale_conf = 0.6
    if abs(scale - 1) > 1e-9:
        pts = pts * scale if pts is not None else None
        for f in cam["frames"]:
            f["C"] = (np.asarray(f["C"]) * scale).tolist()
    print(f"[world] scale={scale:.3f} (conf={scale_conf}), ground={ground_method}")

    # 3. 资产世界落位
    dd = {"character": 8.0, "vehicle": 10.0, "tree": 12.0,
          "building": 20.0, "prop": 10.0}
    for a in scene["assets"]:
        obs = a.get("trajectory") or []
        P = []
        for o in obs:
            f = frames.get(o["frame"])
            if f is None:
                continue
            R = np.asarray(f["R"], float)
            C = np.asarray(f["C"], float)
            bbox = o["bbox"]
            u = (bbox[0] + bbox[2]) / 2 * W
            v = bbox[3] * H
            p = ray_ground(C, pixel_dir(u, v, R, K),
                           default_dist=dd.get(a["category"]))
            if p is not None:
                P.append((o["frame"], p, C, R))
        if not P:
            a["world_position"] = None
            continue
        pos = np.median(np.asarray([p for _, p, _, _ in P]), axis=0)
        pos[2] = 0.0  # 对象立于地面
        a["world_position"] = [round(float(x), 3) for x in pos]

        if a["category"] == "character":
            traj = []
            last = None
            for fr, p, C, R in sorted(P, key=lambda x: x[0]):
                yaw = 0.0
                if last is not None:
                    dx, dy = p[0] - last[0], p[1] - last[1]
                    if math.hypot(dx, dy) > 0.05:
                        yaw = round(math.degrees(math.atan2(dy, dx)) - 90.0, 1)
                traj.append({"frame": fr, "position": [round(p[0], 3), round(p[1], 3)],
                             "yaw_deg": yaw})
                last = p
            a["trajectory_world"] = traj
            a["yaw_deg"] = traj[0]["yaw_deg"] if traj else 0.0

        if a["category"] in ("building", "prop", "vehicle"):
            o = obs[len(obs) // 2]
            f = frames.get(o["frame"])
            if f is not None:
                R = np.asarray(f["R"], float)
                C = np.asarray(f["C"], float)
                w_px = (o["bbox"][2] - o["bbox"][0]) * W
                dist = float(np.linalg.norm(pos - C))
                if not (a.get("size_m") or {}).get("w"):
                    a.setdefault("size_m", {})["w"] = round(w_px * dist / K[0, 0], 2)

    # 4. 姿态提升（世界 3D 关节）
    _lift_poses(paths, scene, frames, K, W, H, cfg)

    # 5. 回写
    scene["ground"] = {"normal": [0, 0, 1.0], "offset": 0.0,
                       "method": ground_method}
    scene["world"] = {"scale": scale, "scale_confidence": scale_conf,
                      "camera_method": cam.get("method")}
    scene["environment"] = (scene["assets"][0]["spec"].get("environment")
                            if scene["assets"] else None) or scene.get("environment")
    dump_json(paths.camera_json, cam)
    dump_json(paths.scene_json, scene)
    print(f"[world] 资产落位 {sum(1 for a in scene['assets'] if a.get('world_position'))}"
          f"/{len(scene['assets'])} → scene.json")


def _person_height_sfm(pts, asset, frames, K, W, H):
    """稀疏点云中该人物 bbox 内点簇的 up 向高度（SfM 单位）。"""
    zs = []
    for o in asset["trajectory"][:6]:
        f = frames.get(o["frame"])
        if f is None:
            continue
        R = np.asarray(f["R"], float)
        C = np.asarray(f["C"], float)
        x0, y0, x1, y1 = [o["bbox"][0] * W, o["bbox"][1] * H,
                          o["bbox"][2] * W, o["bbox"][3] * H]
        for P in pts[:: max(1, len(pts) // 4000)]:
            pr = project(P, R, C, K)
            if pr and x0 <= pr[0] <= x1 and y0 <= pr[1] <= y1:
                zs.append(P[2])
    if len(zs) < 10:
        return None
    zs = np.asarray(zs)
    return float(np.percentile(zs, 95) - zs.min())


def _lift_poses(paths, scene, frames, K, W, H, cfg) -> None:
    p2 = load_json(paths.poses2d_json)
    chars = [a for a in scene["assets"] if a["category"] == "character"]
    if not p2 or not chars:
        dump_json(paths.poses_json, {"mode": "none", "frames": []})
        return
    h_person = cfg.get("pose.person_height_m", 1.75)
    out = []
    for fr in p2["frames"]:
        f = frames.get(fr["frame"])
        if not fr["persons"] or f is None:
            continue
        R = np.asarray(f["R"], float)
        C = np.asarray(f["C"], float)
        person = fr["persons"][0]  # 已按置信度排序
        kps = person["keypoints2d"]

        def ok(i):
            return 0 <= i < len(kps) and kps[i][2] > 0.3

        # 踝部中点 → 地面点 → 人物平面距离
        if ok(15) and ok(16):
            ua = (kps[15][0] + kps[16][0]) / 2
            va = (kps[15][1] + kps[16][1]) / 2
        else:
            b = person["bbox_px"]
            ua, va = (b[0] + b[2]) / 2, b[3]
        pa = ray_ground(C, pixel_dir(ua, va, R, K), default_dist=6.0)
        if pa is None:
            continue
        dper = float(np.linalg.norm(pa - C))
        joints = {}
        for idx, name in COCO_MAP.items():
            if ok(idx):
                d = pixel_dir(kps[idx][0], kps[idx][1], R, K)
                joints[name] = [round(float(x), 3) for x in (C + dper * d)]
        if ok(5) and ok(6):
            joints["neck"] = [(joints["shoulder_l"][i] + joints["shoulder_r"][i]) / 2
                              for i in range(3)]
        if ok(11) and ok(12):
            joints["pelvis"] = [(joints["hip_l"][i] + joints["hip_r"][i]) / 2
                                for i in range(3)]
        if "head" in joints and "neck" in joints:
            dv = np.asarray(joints["head"]) - np.asarray(joints["neck"])
            dv = dv / (np.linalg.norm(dv) + 1e-9)
            joints["head_top"] = [round(float(x), 3)
                                  for x in np.asarray(joints["head"]) + 0.10 * h_person * dv]
        out.append({"frame": fr["frame"], "timestamp": fr["timestamp"],
                    "asset_id": chars[0]["id"], "joints": joints})
    dump_json(paths.poses_json, {"mode": "keyframed",
                                 "person_height_m": h_person, "frames": out})
    print(f"[world] 姿态提升 {len(out)} 帧 → poses.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4b：尺度/地面/世界融合")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config))


if __name__ == "__main__":
    main()
