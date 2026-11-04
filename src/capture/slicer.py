"""Stage 1：切镜检测与 15–30s 一镜到底切片（PySceneDetect + ffmpeg）。

设计文档 §4.1：
- 检出切镜的段直接回退；>30s 一镜到底按 30s 截断（manifest 标记 trimmed）
- 纯净度预检在 Stage 2 中执行，含 UI/水印即打回

用法：
    python -m src.capture.slicer --input data/raw/game.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from src.common.config import load_config
from src.common.media import run_ffmpeg, video_info
from src.common.schema import dump_json


def detect_cut_times(video: Path, threshold: float) -> list[float]:
    """返回切镜时刻列表（秒）。"""
    try:
        from scenedetect import ContentDetector, detect
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少 scenedetect：pip install scenedetect") from e
    scenes = detect(str(video), ContentDetector(threshold=threshold))
    # scenes = [(start, end), ...]；切点 = 每段的结束时刻（除最后一段）
    return [s[1].get_seconds() for s in scenes[:-1]]


def slice_video(video: Path, out_dir: Path, cfg) -> list[dict]:
    """按切镜边界切段并输出 15-30s 片段，返回 manifest 条目列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    duration = info["duration"]
    cuts = detect_cut_times(video, cfg.get("clip.scene_detect_threshold", 27.0))

    # 无切镜的完整区间；超过 max_duration 的一镜到底按 max 截断
    boundaries = [0.0] + cuts + [duration]
    segments: list[dict] = []
    min_d = cfg.get("clip.min_duration_s", 15.0)
    max_d = cfg.get("clip.max_duration_s", 30.0)

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        dur = end - start
        chunks = []
        if dur < min_d:
            continue  # 过短直接丢弃
        n = math.ceil(dur / max_d)
        for k in range(n):
            cs = start + k * max_d
            cd = min(max_d, end - cs)
            if cd >= min_d * 0.6:  # 允许略短收尾，否则丢弃
                chunks.append((cs, cd))
        for cs, cd in chunks:
            segments.append({"start": round(cs, 3), "duration": round(cd, 3),
                             "trimmed": (end - cs) > max_d + 1e-6})

    manifest = []
    stem = video.stem
    for idx, seg in enumerate(segments):
        clip_id = f"{stem}_seg{idx:03d}"
        out = out_dir / f"{clip_id}.mp4"
        if not out.exists():
            run_ffmpeg([
                "-ss", f"{seg['start']:.3f}", "-i", str(video),
                "-t", f"{seg['duration']:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k", str(out),
            ])
        manifest.append({"clip_id": clip_id, "source": str(video), **seg})

    dump_json(out_dir / "manifest.json", {"source": str(video),
                                          "segments": manifest})
    print(f"[slicer] 切镜 {len(cuts)} 处 → 产出 {len(manifest)} 个片段 → {out_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1：切镜检测与切片")
    ap.add_argument("--input", required=True, help="源视频路径")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    video = Path(args.input)
    if not video.exists():
        raise SystemExit(f"输入不存在: {video}")
    slice_video(video, cfg.path("paths.clips"), cfg)


if __name__ == "__main__":
    main()
