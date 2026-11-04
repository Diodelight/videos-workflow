"""角色姿态估计（设计文档 §3.2）。

后端选择（config pose.backend）：
- auto / ultralytics：YOLO11-pose（pip 可装，接口简单；风格化游戏域外性能有限）
- rtmpose：RTMPose/WHAM（mmpose，需自行配置 config/checkpoint，骨架留 TODO）
- off：跳过

输出 poses2d.json（关键帧 COCO17 2D 骨架）；3D 世界提升在 Stage 4 scale_ground 中完成。
姿态失败时降级路径：白模人物使用内置 walk_cycle 步态（风险 #1 对策）。

用法：python -m src.vision.pose --clip <clip_id>
"""
from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.paths import clip_paths
from src.common.schema import dump_json

COCO17 = ["nose", "leye", "reye", "lear", "rear",
          "lshoulder", "rshoulder", "lelbow", "relbow",
          "lwrist", "rwrist", "lhip", "rhip", "lknee", "rknee",
          "lankle", "rankle"]


def _predict_ultralytics(model_name: str, files):
    from ultralytics import YOLO
    model = YOLO(model_name)
    out = []
    for f in files:
        res = model.predict(source=str(f), verbose=False)[0]
        persons = []
        if res.keypoints is not None and res.keypoints.xy is not None:
            kxy = res.keypoints.xy.cpu().numpy()
            kcf = (res.keypoints.conf.cpu().numpy()
                   if res.keypoints.conf is not None
                   else [[1.0] * kxy.shape[1]] * kxy.shape[0])
            boxes = (res.boxes.xyxy.cpu().numpy()
                     if res.boxes is not None
                     else [[0, 0, 1, 1]] * len(kxy))
            for i in range(len(kxy)):
                persons.append({
                    "bbox_px": [float(v) for v in boxes[i]],
                    "keypoints2d": [[float(x), float(y), float(c)]
                                    for (x, y), c in zip(kxy[i], kcf[i])],
                })
        out.append({"frame": int(f.stem[1:]), "persons": persons})
    return out


def run(clip_id: str, cfg) -> bool:
    paths = clip_paths(clip_id, cfg)
    skip = paths.analysis / "pose.skipped"
    if paths.poses2d_json.exists():
        print("[pose] 已完成，跳过")
        return True
    backend = cfg.get("pose.backend", "auto")
    if backend == "off" or not cfg.get("pose.enabled", True):
        skip.write_text("disabled", encoding="utf-8")
        return False
    files = sorted(paths.frames_key.glob("f*.jpg"))
    try:
        if backend in ("auto", "ultralytics"):
            frames = _predict_ultralytics(
                cfg.get("pose.yolo_pose_model", "yolo11n-pose.pt"), files)
        else:
            raise ImportError(f"后端 {backend} 未配置（RTMPose/WHAM 需 mmpose 环境，"
                              "预留 TODO）")
    except ImportError as e:
        skip.write_text(str(e), encoding="utf-8")
        print(f"[pose] 跳过（{e}）→ 白模人物将使用 walk_cycle 降级步态")
        return False

    # 单一主体：取关键点置信度总和最大者
    for fr in frames:
        fr["persons"] = sorted(
            fr["persons"],
            key=lambda p: -sum(k[2] for k in p["keypoints2d"]))[:3]
    dump_json(paths.poses2d_json, {"clip_id": clip_id,
                                   "keypoints_order": COCO17,
                                   "frames": frames})
    n = sum(len(f["persons"]) for f in frames)
    print(f"[pose] {len(frames)} 关键帧 → {n} 个观测 → {paths.poses2d_json}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="角色 2D 姿态估计")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config))


if __name__ == "__main__":
    main()
