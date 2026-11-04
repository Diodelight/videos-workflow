"""端到端流水线编排（设计文档 §3.1 架构 → Stage 1–7 顺序执行）。

- 断点续跑：data/analysis/<clip_id>/_done_<stage>.marker
- 单阶段失败即中止该片段（生产档语义）；本地模型缺失为优雅跳过
- 开发调试期建议配合 blender-mcp 人工介入；批量生产直接跑本入口

用法：
    python run_pipeline.py --input data/raw/game.mp4            # 切片 + 全流程
    python run_pipeline.py --clip xxx_seg000                    # 续跑单片段
    python run_pipeline.py --clip xxx_seg000 --stages semantic,world,generate
    python run_pipeline.py --clip xxx_seg000 --force            # 忽略断点重跑
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.blender_gen import generate
from src.common.config import load_config
from src.common.paths import clip_paths
from src.preprocess import preprocess, purity
from src.qa import qa_check
from src.render import compose
from src.sfm import camera_solve, scale_ground
from src.vision import depth, pose as pose_stage, segment, semantic_parse

STAGES = [
    ("preprocess", preprocess.run),      # Stage 2 抽帧/音频/元数据
    ("purity", purity.run),              # Stage 1 门禁：纯净度预检
    ("semantic", semantic_parse.run),    # Stage 3 语义解析 + 资产库
    ("depth", depth.run),                # Stage 3 深度（可跳过）
    ("segment", segment.run),            # Stage 3 分割（可跳过）
    ("pose", pose_stage.run),            # Stage 3 姿态（可跳过）
    ("camera", camera_solve.run),        # Stage 4a 相机解算
    ("world", scale_ground.run),         # Stage 4b 尺度/地面/世界融合
    ("generate", generate.run),          # Stage 5 Blender 白模 + 渲染
    ("compose", compose.run),            # Stage 6 配对合成与交付
    ("qa", qa_check.run),                # Stage 7 自动质检
]


class StageError(RuntimeError):
    pass


def run_clip(clip_id: str, cfg, only: set | None = None,
             force: bool = False) -> None:
    paths = clip_paths(clip_id, cfg).ensure()
    print(f"===== clip: {clip_id} =====")
    for name, fn in STAGES:
        if only and name not in only:
            continue
        if paths.marker(name).exists() and not force:
            print(f"[skip] {name}（已完成）")
            continue
        print(f"[stage] {name}")
        try:
            fn(clip_id, cfg)
        except Exception as e:
            raise StageError(f"[{name}] {e}") from e
        paths.marker(name).write_text("ok", encoding="utf-8")
    print(f"[done] {clip_id}: 请求的阶段全部完成")


def main() -> None:
    ap = argparse.ArgumentParser(description="白模实时建模流水线")
    ap.add_argument("--clip", help="切片 clip_id（data/clips/<id>.mp4）")
    ap.add_argument("--input", help="原始游戏视频：自动切片后逐段生产")
    ap.add_argument("--stages", help="阶段子集，逗号分隔（默认全部）")
    ap.add_argument("--force", action="store_true", help="忽略断点重跑")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    only = set(args.stages.split(",")) if args.stages else None

    if args.input:
        from src.capture.slicer import slice_video
        segs = slice_video(Path(args.input), cfg.path("paths.clips"), cfg)
        for seg in segs:
            run_clip(seg["clip_id"], cfg, only, args.force)
    elif args.clip:
        run_clip(args.clip, cfg, only, args.force)
    else:
        ap.error("必须提供 --clip 或 --input")


if __name__ == "__main__":
    main()
