"""纯净度预检（Stage 1 门禁）：字幕条 / 台标 / 水印 / 播放器 UI 启发式检测。

设计文档 §4.1：含 UI 即打回，不进入后续环节。
- 启发式 A：顶/底边缘低方差色块（台标、进度条、字幕底条）
- 启发式 B（可选）：PaddleOCR 文本区域检测（未安装则记录 skipped）

用法：python -m src.preprocess.purity --clip <clip_id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.common.config import load_config
from src.common.imio import imread
from src.common.paths import clip_paths
from src.common.schema import dump_json


def _solid_bar_score(img: np.ndarray, band: np.ndarray) -> float:
    """条带内逐行标准差很低 → 疑似字幕底条/进度条。返回 0-1 疑似度。"""
    std = float(band.reshape(band.shape[0], -1).std(axis=1).min())
    return 1.0 if std < 6.0 else max(0.0, 1.0 - (std - 6.0) / 20.0)


def check_frames(frames_dir: Path, n_samples: int) -> dict:
    files = sorted(frames_dir.glob("f*.jpg"))
    if not files:
        return {"passed": False, "reasons": ["无可用帧，请先运行 preprocess"]}
    picks = [files[int(i)] for i in np.linspace(0, len(files) - 1, n_samples)]
    reasons: list[str] = []
    worst_bar = 0.0

    for f in picks:
        img = imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        top = img[: int(h * 0.10)]
        bottom = img[int(h * 0.90):]
        score = max(_solid_bar_score(img, top), _solid_bar_score(img, bottom))
        worst_bar = max(worst_bar, score)

    if worst_bar > 0.9:
        reasons.append(f"顶/底边缘存在低方差色块（疑似字幕条/进度条/UI），score={worst_bar:.2f}")

    ocr = None
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
    except Exception:
        pass

    if ocr is not None:
        for f in picks:
            try:
                result = ocr.ocr(str(f), cls=False)
            except TypeError:
                result = ocr.ocr(str(f))
            for line in (result[0] or []):
                box, (text, conf) = line[0], line[1]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                ratio = area / float(w * h)
                edge = min(ys) < h * 0.15 or max(ys) > h * 0.85
                if ratio > 0.01 or edge:
                    reasons.append(f"OCR 检出疑似字幕/水印: “{text}” conf={conf:.2f}")
        method = "paddleocr+heuristic"
    else:
        method = "heuristic_only"

    return {"passed": not reasons, "reasons": reasons, "method": method,
            "bar_score": round(worst_bar, 3)}


def run(clip_id: str, cfg) -> None:
    paths = clip_paths(clip_id, cfg)
    report = check_frames(paths.frames_full,
                          cfg.get("preprocess.purity_frame_samples", 8))
    dump_json(paths.purity_json, {"clip_id": clip_id, **report})
    if not report["passed"]:
        raise RuntimeError("纯净度预检未通过（脏片打回，不进入流水线）: "
                           + "; ".join(report["reasons"]))
    print(f"[purity] {clip_id}: PASS ({report['method']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="纯净度预检（进流水线门禁）")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config))


if __name__ == "__main__":
    main()
