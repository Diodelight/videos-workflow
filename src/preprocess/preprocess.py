"""Stage 2：抽帧（全帧 + 运动自适应关键帧）、音频提取、元数据入库。

设计文档 §4.2：
- 全帧序列：供深度/分割/渲染对齐用（cv2 逐帧读取保证帧号精确对齐）
- 关键帧序列：默认 0.5fps 起步，按帧差能量自适应加密
- 音频：提取源视频 audio 轨道暂存待合成（需求：白模视频保留 audio）
- 元数据：帧数/fps/分辨率/时长 → meta.json，作为对齐校验基准

用法：python -m src.preprocess.preprocess --clip <clip_id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.common.config import load_config
from src.common.imio import imread, imwrite
from src.common.media import run_ffmpeg
from src.common.paths import clip_paths
from src.common.schema import dump_json


def probe_with_cv2(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    cap.release()
    return {"frame_count": frames, "fps": fps, "width": width, "height": height,
            "duration": frames / fps if fps else 0.0}


def extract_full_frames(video: Path, out_dir: Path, quality: int) -> int:
    """逐帧导出 f{frame:06d}.jpg，返回帧数。"""
    if any(out_dir.glob("f*.jpg")):
        return len(list(out_dir.glob("f*.jpg")))
    cap = cv2.VideoCapture(str(video))
    idx = failed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # cv2.imwrite 在 Windows 中文路径下静默失败，必须用 Unicode 安全写入
        if not imwrite(out_dir / f"f{idx:06d}.jpg", frame,
                       [cv2.IMWRITE_JPEG_QUALITY, quality]):
            failed += 1
        idx += 1
    cap.release()
    if failed:
        raise RuntimeError(f"{failed} 帧写入失败（路径: {out_dir}）")
    return idx


def motion_energy(frames_dir: Path) -> np.ndarray:
    """逐帧差分能量（下采样灰度），用于关键帧自适应加密。"""
    files = sorted(frames_dir.glob("f*.jpg"))
    energy = np.zeros(len(files), dtype=np.float64)
    prev = None
    for i, f in enumerate(files):
        img = imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        small = cv2.resize(img, (64, 36))
        if prev is not None:
            energy[i] = float(np.mean(cv2.absdiff(small, prev)))
        prev = small
    return energy


def select_keyframes(frame_count: int, fps: float, energy: np.ndarray, cfg) -> list[int]:
    base = cfg.get("preprocess.keyframe_base_fps", 0.5)
    min_iv = cfg.get("preprocess.keyframe_min_interval_s", 1.0)
    max_iv = cfg.get("preprocess.keyframe_max_interval_s", 4.0)
    step = max(1, int(round(fps / base)))
    min_gap = max(1, int(min_iv * fps))
    max_gap = max(min_gap + 1, int(max_iv * fps))
    thr = 2.0 * float(energy.mean() or 1.0)

    selected = [0]
    while selected[-1] + step < frame_count - 1:
        nxt = selected[-1] + step
        seg = energy[selected[-1]:nxt]
        # 运动加密：区间能量显著偏高则插入中点（不低于最小间隔）
        if seg.size and seg.mean() > thr and (nxt - selected[-1]) >= 2 * min_gap:
            nxt = selected[-1] + (nxt - selected[-1]) // 2
        selected.append(nxt)
    if frame_count - 1 not in selected:
        selected.append(frame_count - 1)

    # 最大间隔兜底：中点插入
    out: list[int] = []
    for f in selected:
        while out and f - out[-1] > max_gap:
            out.append(out[-1] + (f - out[-1]) // 2)
        if not out or f != out[-1]:
            out.append(f)
    return out


def extract_audio(video: Path, out: Path) -> bool:
    """提取音频轨（aac 转码 128k）；无音轨返回 False。"""
    from src.common.media import audio_present
    if not audio_present(video):
        return False
    run_ffmpeg(["-i", str(video), "-vn", "-c:a", "aac", "-b:a", "128k", str(out)])
    return True


def run(clip_id: str, cfg) -> None:
    paths = clip_paths(clip_id, cfg).ensure()
    video = paths.clip_video
    if not video.exists():
        raise RuntimeError(f"切片不存在: {video}")

    info = probe_with_cv2(video)
    n_full = extract_full_frames(video, paths.frames_full,
                                 cfg.get("preprocess.jpeg_quality", 92))
    info["frame_count"] = n_full  # 以实际解码帧数为准（对齐基准）
    info["duration"] = n_full / info["fps"] if info["fps"] else 0.0

    energy = motion_energy(paths.frames_full)
    keyframes = select_keyframes(n_full, info["fps"], energy, cfg)

    # 关键帧文件：从全帧序列复制到 frames_key（供语义/姿态/深度使用）
    import shutil
    for f in keyframes:
        src = paths.frames_full / f"f{f:06d}.jpg"
        dst = paths.frames_key / f"f{f:06d}.jpg"
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)

    dump_json(paths.keyframes_index, {
        "clip_id": clip_id,
        "key_frames": [{"frame": f, "file": f"f{f:06d}.jpg",
                        "timestamp": round(f / info["fps"], 3) if info["fps"] else 0.0}
                       for f in keyframes],
    })

    has_audio = extract_audio(video, paths.audio)
    meta = {**info, "clip_id": clip_id, "source_video": str(video),
            "n_keyframes": len(keyframes), "has_audio": has_audio}
    dump_json(paths.meta, meta)
    print(f"[preprocess] {clip_id}: {n_full} 帧 @ {info['fps']:.3f}fps, "
          f"{info['width']}x{info['height']}, 关键帧 {len(keyframes)}, "
          f"audio={'有' if has_audio else '无'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2：抽帧/音频/元数据")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config))


if __name__ == "__main__":
    main()
