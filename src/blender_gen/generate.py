"""Stage 5 驱动：调用 Blender headless 执行 gen_blockout.py。

调试期建议改用 blender-mcp 交互式修正（设计文档 §4.5 / §5.1）。

由流水线自动调用；也可手动：
    python -m src.blender_gen.generate --clip <clip_id>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from src.common.config import load_config
from src.common.paths import clip_paths
from src.common.schema import load_json

GEN = Path(__file__).parent / "gen_blockout.py"


def _blender_exe(cfg) -> str:
    exe = str(cfg.get("blender.executable", "blender"))
    return shutil.which(exe) or (exe if Path(exe).exists() else "")


def run(clip_id: str, cfg) -> None:
    paths = clip_paths(clip_id, cfg).ensure()
    flags = load_json(paths.flags_json, {})
    if flags.get("multi_subject"):
        raise RuntimeError("整对打回：" + flags.get("reject_reason",
                                                  "同帧多个可辨识人物主体"))
    if not paths.scene_json.exists():
        raise RuntimeError("缺少 scene.json，请先运行 semantic + world 阶段")
    blender = _blender_exe(cfg)
    if not blender:
        raise RuntimeError("未找到 Blender，请检查 config.yaml blender.executable")

    bg = cfg.get("blender.background", "green")
    rgb = cfg.get("blender.green_rgb" if bg == "green" else "blender.white_rgb",
                  [0.0, 0.55, 0.12])
    cmd = [blender, "-b", "-P", str(GEN), "--",
           "--scene", str(paths.scene_json),
           "--camera", str(paths.camera_json),
           "--meta", str(paths.meta),
           "--poses", str(paths.poses_json) if paths.poses_json.exists() else "none",
           "--out", str(paths.render_dir),
           "--blend", str(paths.blend_file),
           "--compliance", str(paths.compliance_json),
           "--background", bg,
           "--bg-rgb", ",".join(str(v) for v in rgb),
           "--ground-extent", str(cfg.get("blender.ground_extent_m", 30.0)),
           "--samples", str(cfg.get("blender.eevee_samples", 64))]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"Blender 执行失败:\n{r.stderr[-1000:]}")
    comp = load_json(paths.compliance_json, {})
    if not comp.get("passed", False):
        raise RuntimeError(f"白模合规检查未通过: {comp}")
    print(f"[generate] 渲染完成 → {paths.render_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5：Blender 白模生成")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config))


if __name__ == "__main__":
    main()
