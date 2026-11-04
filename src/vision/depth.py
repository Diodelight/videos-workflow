"""Depth Anything V2 本地推理封装（设计文档 §3.2 / §5.2）。

未安装 torch/transformers 时优雅跳过（写 depth.skipped 标记），
流水线继续运行（深度仅用于遮挡排序与降级路径）。

用法：python -m src.vision.depth --clip <clip_id> [--target key|full]
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from src.common.config import load_config
from src.common.imio import imread, imwrite
from src.common.paths import clip_paths
from src.common.schema import dump_json


def _device(cfg) -> str:
    dev = cfg.get("depth.device", "auto")
    if dev != "auto":
        return dev
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _pipeline(cfg):
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError("缺少 transformers/torch：pip install torch transformers") from e
    return pipeline("depth-estimation",
                    model=cfg.get("depth.model_id",
                                  "depth-anything/Depth-Anything-V2-Small-hf"),
                    device=_device(cfg))


def run(clip_id: str, cfg, target: str = "key") -> bool:
    paths = clip_paths(clip_id, cfg)
    skip = paths.analysis / "depth.skipped"
    if (paths.depth_dir / "done.json").exists():
        print("[depth] 已完成，跳过")
        return True
    if not cfg.get("depth.enabled", True):
        skip.write_text("disabled", encoding="utf-8")
        return False
    try:
        pipe = _pipeline(cfg)
    except ImportError as e:
        skip.write_text(str(e), encoding="utf-8")
        print(f"[depth] 跳过（{e}）")
        return False

    frames_dir = paths.frames_key if target == "key" else paths.frames_full
    files = sorted(frames_dir.glob("f*.jpg"))
    paths.depth_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        out = pipe(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        depth = np.asarray(out["predicted_depth"], dtype=np.float32)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        np.save(paths.depth_dir / f"f{f.stem[1:]}.npy", depth.astype(np.float16))
        norm = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
        imwrite(paths.depth_dir / f"f{f.stem[1:]}.png",
                (norm * 255).astype(np.uint8))
    dump_json(paths.depth_dir / "done.json",
              {"model": cfg.get("depth.model_id"), "count": len(files),
               "target": target})
    print(f"[depth] 完成 {len(files)} 帧 → {paths.depth_dir}")
    return True


def load_depth(paths, frame: int) -> np.ndarray | None:
    """读取某帧深度图（float32，相对深度）。"""
    p = paths.depth_dir / f"f{frame:06d}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Depth Anything V2 深度估计")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--target", default="key", choices=["key", "full"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config), args.target)


if __name__ == "__main__":
    main()
