"""ffmpeg / ffprobe 子进程封装（Stage 1/2/6/7 共用）。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_ffmpeg(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=check, capture_output=True, text=True,
        encoding="utf-8", errors="replace")  # Windows 中文环境默认 GBK，需显式 UTF-8


def run_ffprobe(args: list[str]) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", *args],
        check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return json.loads(out.stdout or "{}")


def probe_streams(video: str | Path) -> dict:
    return run_ffprobe(
        ["-show_streams", "-show_format", "-of", "json", str(video)])


def video_info(video: str | Path) -> dict:
    """宽/高/fps/时长/流数。"""
    data = probe_streams(video)
    vs = next((s for s in data.get("streams", [])
               if s.get("codec_type") == "video"), None)
    if vs is None:
        raise RuntimeError(f"无视频流: {video}")
    fps = 0.0
    rate = vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "0/1"
    if "/" in rate:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    return {
        "width": int(vs["width"]),
        "height": int(vs["height"]),
        "fps": fps,
        "duration": float(vs.get("duration") or data.get("format", {}).get("duration", 0) or 0),
        "nb_frames": int(vs["nb_frames"]) if str(vs.get("nb_frames", "")).isdigit() else None,
    }


def audio_present(video: str | Path) -> bool:
    data = probe_streams(video)
    return any(s.get("codec_type") == "audio" for s in data.get("streams", []))


def count_frames(video: str | Path) -> int:
    """精确解码计数（对齐校验用）。"""
    data = run_ffprobe(["-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=nb_read_frames",
                        "-of", "json", str(video)])
    s = (data.get("streams") or [{}])[0]
    return int(s.get("nb_read_frames", 0))
