"""Stage 6：白模序列 + 原音频 → 白模视频；side-by-side 质检片；成对交付。

设计文档 §4.6：
- 合成保留原 audio（需求：白模视频保留 audio）
- 帧数 = 源视频帧数（逐帧对齐的物理保证），分辨率/帧率一致
- 交付目录：白模视频.mp4 / 目标视频.mp4 / 对齐映射说明.json / side_by_side.mp4

用法：python -m src.render.compose --clip <clip_id>
"""
from __future__ import annotations

import argparse
import shutil

from src.common.config import load_config
from src.common.media import run_ffmpeg, video_info
from src.common.paths import clip_paths
from src.common.schema import dump_json, load_json


def mux_white(paths, meta) -> None:
    """渲染序列 + 音频 → 白模视频.mp4（保留原 audio）。"""
    pattern = str(paths.render_dir / "white_%04d.png")
    out = str(paths.white_video)
    fps = f"{meta['fps']:.6f}"
    args = ["-framerate", fps, "-start_number", "1", "-i", pattern]
    if paths.audio.exists():
        args += ["-i", str(paths.audio),
                 "-map", "0:v:0", "-map", "1:a:0",
                 "-c:a", "aac", "-b:a", "128k"]
    else:
        args += ["-map", "0:v:0"]
    args += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-r", fps,
             "-frames:v", str(meta["frame_count"]), out]
    run_ffmpeg(args)


def side_by_side(white, target, out) -> None:
    """白模 | 目标并排质检片（供人工抽检）。"""
    run_ffmpeg([
        "-i", str(white), "-i", str(target),
        "-filter_complex",
        "[0:v]scale=-2:360[a];[1:v]scale=-2:360[b];[a][b]hstack=inputs=2[v]",
        "-map", "[v]", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        str(out)])


def run(clip_id: str, cfg) -> None:
    paths = clip_paths(clip_id, cfg)
    meta = load_json(paths.meta)
    if not meta:
        raise RuntimeError("缺少 meta.json，请先运行 preprocess")
    if not list(paths.render_dir.glob("white_*.png")):
        raise RuntimeError("渲染序列缺失，请先运行 generate 阶段")

    mux_white(paths, meta)

    delivery = paths.delivery_dir
    delivery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.clip_video, delivery / "目标视频.mp4")
    shutil.copy2(paths.white_video, delivery / "白模视频.mp4")

    wi = video_info(paths.white_video)
    ti = video_info(paths.clip_video)
    dump_json(delivery / "对齐映射说明.json", {
        "white_video": {"width": wi["width"], "height": wi["height"],
                        "fps": wi["fps"], "frames": meta["frame_count"]},
        "target_video": {"width": ti["width"], "height": ti["height"],
                         "fps": ti["fps"]},
        "scale": {"x": round(ti["width"] / wi["width"], 6),
                  "y": round(ti["height"] / wi["height"], 6)},
        "note": "两路逐帧对齐（同帧率同帧数）；分辨率不一致时按 scale 缩放映射",
    })
    try:
        side_by_side(paths.white_video, delivery / "目标视频.mp4",
                     delivery / "side_by_side.mp4")
    except Exception as e:  # 质检片失败不影响交付
        print(f"[compose] side-by-side 生成失败（不影响交付）: {e}")
    print(f"[compose] 成对交付 → {delivery}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6：配对合成与交付")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config))


if __name__ == "__main__":
    main()
