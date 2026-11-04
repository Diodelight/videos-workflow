"""Stage 4a：相机轨迹解算（设计文档 §4.4）。

- MASt3R（优先）/ GLOMAP（备选）：命令模板由 config 提供，
  要求输出 COLMAP 文本格式（cameras.txt / images.txt / points3D.txt）
- 解析 → 关键帧锚点 → 位姿内插到全部帧（SLERP + LERP，风险 #2 降级路径）
- 解算失败 → 降级恒定机位（低置信度标注，质检加严）

camera.json 约定：
  convention: "colmap"（R=cam→world 原始 SfM 空间）或 "blender"（降级直出）
  frames[i] = {frame, timestamp, R(3x3), C[3], confidence}

用法：python -m src.sfm.camera_solve --clip <clip_id>
"""
from __future__ import annotations

import argparse
import math
import shlex
import subprocess
from pathlib import Path

import numpy as np

from src.common.config import load_config
from src.common.paths import clip_paths
from src.common.schema import dump_json, load_json

EPS = 1e-9


# ---------------- 四元数 / 位姿工具 ----------------
def quat_from_mat(m: np.ndarray) -> np.ndarray:
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + EPS)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / (np.linalg.norm(q0) + EPS)
    q1 = q1 / (np.linalg.norm(q1) + EPS)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / (np.linalg.norm(q) + EPS)
    ang = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(ang)
    return (math.sin((1 - t) * ang) / s) * q0 + (math.sin(t * ang) / s) * q1


# ---------------- COLMAP 文本解析 ----------------
def parse_colmap_sparse(sparse_dir: Path):
    cams, images, points = {}, {}, []
    cfile = sparse_dir / "cameras.txt"
    ifile = sparse_dir / "images.txt"
    pfile = sparse_dir / "points3D.txt"
    if not (cfile.exists() and ifile.exists()):
        return None, None, None
    for line in cfile.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        cams[int(p[0])] = {"model": p[1], "w": int(p[2]), "h": int(p[3]),
                           "params": [float(x) for x in p[4:]]}
    lines = [l for l in ifile.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):  # images.txt 每图两行
        p = lines[i].split()
        q = np.array([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
        t = np.array([float(p[5]), float(p[6]), float(p[7])])
        w, x, y, z = q / (np.linalg.norm(q) + EPS)
        r_wc = np.array([  # world→cam
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        images[int(p[0])] = {"name": p[9],
                             "R_c2w": r_wc.T, "C": (-r_wc.T @ t)}
    if pfile.exists():
        for line in pfile.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            points.append([float(p[1]), float(p[2]), float(p[3])])
    return cams, images, (np.asarray(points) if points else None)


# ---------------- 外部解算 ----------------
def _run_backend(cmd_tpl: str, frames_dir: Path, out_dir: Path) -> bool:
    if not cmd_tpl:
        return False
    cmd = [c.format(frames_dir=str(frames_dir), out_dir=str(out_dir))
           for c in shlex.split(cmd_tpl)]
    print(f"[camera] 运行解算后端: {' '.join(cmd[:3])}…")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(f"[camera] 后端失败: {r.stderr[-400:]}")
            return False
    except Exception as e:
        print(f"[camera] 后端异常: {e}")
        return False
    hits = list(out_dir.rglob("images.txt"))
    if not hits:
        print("[camera] 后端未产出 COLMAP 文本")
        return False
    return True


# ---------------- 主流程 ----------------
def run(clip_id: str, cfg) -> None:
    paths = clip_paths(clip_id, cfg).ensure()
    meta = load_json(paths.meta)
    index = load_json(paths.keyframes_index)
    if not (meta and index):
        raise RuntimeError("缺少 meta/keyframes，请先运行 preprocess")

    frame_count, fps = meta["frame_count"], meta["fps"]
    W, H = meta["width"], meta["height"]
    anchors = [k["frame"] for k in index["key_frames"]]
    if len(anchors) > cfg.get("camera.keyframe_subsample", 24):
        step = len(anchors) / cfg["camera.keyframe_subsample"]
        anchors = [anchors[int(i * step)] for i in
                   range(cfg["camera.keyframe_subsample"])]

    cams, images, points = None, None, None
    method, conf = "degraded_identity", 0.1
    backend = cfg.get("camera.backend", "auto")
    sfm_out = paths.sfm_dir
    sfm_out.mkdir(parents=True, exist_ok=True)
    frames_dir = paths.frames_full  # SfM 需要连续图像序列

    if backend in ("auto", "mast3r") and cfg.get("camera.mast3r_cmd"):
        if _run_backend(cfg["camera.mast3r_cmd"], frames_dir, sfm_out):
            cams, images, points = parse_colmap_sparse(_sparse_dir(sfm_out))
            if images:
                method, conf = "mast3r", 0.8
    if images is None and backend in ("auto", "glomap") and cfg.get("camera.glomap_cmd"):
        if _run_backend(cfg["camera.glomap_cmd"], frames_dir, sfm_out):
            cams, images, points = parse_colmap_sparse(_sparse_dir(sfm_out))
            if images:
                method, conf = "glomap", 0.7

    if images:
        # 图像文件名 f000123.jpg → 帧号
        by_name = {Path(im["name"]).stem: im for im in images.values()}
        anchors = sorted(int(n[1:]) for n in by_name if n.startswith("f"))
        anchor_entries = []
        for f in anchors:
            im = by_name[f"f{f:06d}"]
            anchor_entries.append({"frame": f, "R": im["R_c2w"].tolist(),
                                   "C": im["C"].tolist(), "confidence": conf})
        cam = cams[next(iter(cams))]
        params = cam["params"]
        if cam["model"] == "SIMPLE_PINHOLE":
            fx = fy, cx, cy = params[0], params[1], params[2]
        elif cam["model"] == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        else:  # OPENCV/SIMPLE_RADIAL 等取前 4 项近似
            fx, fy, cx, cy = (params[:4] if len(params) >= 4
                              else (params[0], params[0], W / 2, H / 2))
        frames = interpolate(anchor_entries, frame_count, fps)
    else:
        # 降级：固定机位（低置信度，质检加严，风险 #2 兜底）
        print("[camera] 解算不可用 → 降级恒定机位（低置信度）")
        fx = 0.5 * W / math.tan(math.radians(30))  # 等效 60° FOV
        frames = [{"frame": f, "timestamp": (f) / max(fps, EPS),
                   "R": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                   "C": [0.0, -10.0, 2.5], "confidence": 0.1}
                  for f in range(frame_count)]

    dump_json(paths.camera_json, {
        "clip_id": clip_id, "method": method, "convention": "colmap",
        "width": W, "height": H,
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "frames": frames,
        "note": "R 为 cam→world；尺度和地面在 scale_ground 阶段对齐",
    })
    print(f"[camera] {clip_id}: method={method}, 锚点 {len(anchor_entries) if images else 0}, "
          f"输出 {len(frames)} 帧 → camera.json")


def _sparse_dir(root: Path) -> Path:
    hits = list(root.rglob("cameras.txt"))
    return hits[0].parent if hits else root


def interpolate(anchor_entries: list[dict], frame_count: int, fps: float) -> list[dict]:
    """关键帧锚点 → 全帧位姿（旋转 SLERP + 位置 LERP）。"""
    out = []
    fr = [a["frame"] for a in anchor_entries]
    qs = [quat_from_mat(np.asarray(a["R"])) for a in anchor_entries]
    cs = [np.asarray(a["C"], dtype=float) for a in anchor_entries]
    for f in range(frame_count):
        if f <= fr[0]:
            i, t = 0, 0.0
        elif f >= fr[-1]:
            i, t = len(fr) - 2, 1.0
        else:
            i = max(j for j in range(len(fr) - 1) if fr[j] <= f)
            t = (f - fr[i]) / max(fr[i + 1] - fr[i], 1)
        q = quat_slerp(qs[i], qs[i + 1], t)
        w, x, y, z = q / (np.linalg.norm(q) + EPS)
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        C = cs[i] * (1 - t) + cs[i + 1] * t
        out.append({"frame": f, "timestamp": f / max(fps, EPS),
                    "R": R.tolist(), "C": C.tolist(), "confidence": 0.8})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4a：相机轨迹解算")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config))


if __name__ == "__main__":
    main()
