"""SAM2 实例分割封装（设计文档 §3.2 / §5.2）。

对关键帧做自动 mask 生成 + IoU 贪心跟踪（track id 跨帧稳定）。
未安装 sam2 时优雅跳过（写 segment.skipped 标记）。

用法：python -m src.vision.segment --clip <clip_id>
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.common.config import load_config
from src.common.imio import imread, imwrite
from src.common.paths import clip_paths
from src.common.schema import dump_json


def _mask_generator(cfg):
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as e:
        raise ImportError("缺少 sam2：pip install sam2（官方仓库）") from e
    model = build_sam2(cfg.get("segment.model_config",
                               "configs/sam2.1/sam2.1_hiera_t.yaml"),
                       cfg.get("segment.checkpoint",
                               "checkpoints/sam2_hiera_t.pt"),
                       device="cuda")
    return SAM2AutomaticMaskGenerator(model)


def _iou_xyxy(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run(clip_id: str, cfg) -> bool:
    paths = clip_paths(clip_id, cfg)
    skip = paths.analysis / "segment.skipped"
    if (paths.masks_dir / "tracks.json").exists():
        print("[segment] 已完成，跳过")
        return True
    if not cfg.get("segment.enabled", True):
        skip.write_text("disabled", encoding="utf-8")
        return False
    try:
        gen = _mask_generator(cfg)
    except Exception as e:
        skip.write_text(str(e), encoding="utf-8")
        print(f"[segment] 跳过（{e}）")
        return False

    paths.masks_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(paths.frames_key.glob("f*.jpg"))
    tracks: list[dict] = []   # {track_id, frames: {frame: bbox}}
    next_id = 0
    for f in files:
        frame_no = int(f.stem[1:])
        img = cv2.cvtColor(imread(f), cv2.COLOR_BGR2RGB)
        masks = sorted(gen(img), key=lambda m: -m["area"])[:20]
        for m in masks:
            x, y, w, h = m["bbox"]  # xywh
            bbox = [float(x), float(y), float(x + w), float(y + h)]
            best, best_iou = None, 0.35
            for t in tracks:
                last = t["frames"][max(t["frames"])]
                iou = _iou_xyxy(bbox, last["bbox"])
                if iou > best_iou:
                    best, best_iou = t, iou
            if best is None:
                best = {"track_id": next_id, "frames": {}}
                tracks.append(best)
                next_id += 1
            best["frames"][frame_no] = {"bbox": bbox,
                                        "score": float(m.get("predicted_iou", 0)),
                                        "area": float(m["area"])}
            # 保存 mask 预览
            seg = m["segmentation"].astype(np.uint8) * 255
            imwrite(paths.masks_dir / f"t{best['track_id']:03d}_f{frame_no:06d}.png",
                    seg)

    dump_json(paths.masks_dir / "tracks.json",
              {"clip_id": clip_id,
               "tracks": [{"track_id": t["track_id"],
                           "frames": [k for k in sorted(t["frames"])]}
                          for t in tracks]})
    print(f"[segment] {len(files)} 关键帧 → {len(tracks)} 条 track → {paths.masks_dir}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM2 实例分割与跟踪")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config))


if __name__ == "__main__":
    main()
