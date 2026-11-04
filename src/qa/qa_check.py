"""Stage 7：成对自动质检（设计文档 §4.7）。

| 检查项 | 方法 |
|--------|------|
| 对齐   | 帧数(解码计数)/fps/分辨率一致性，0 错帧 |
| 运动同步 | 差分能量曲线归一化后 Pearson 相关（光流/差分能量） |
| 纯净度 | 源视频进线门禁 purity.json；白模天然无 UI |
| 白模合规 | gen 阶段 compliance.json（§2.2 规则） |
| 音频   | 音轨存在 + 响度正常（非静音） |

人工抽检（30%）：依据 delivery 目录 side_by_side.mp4。
用法：python -m src.qa.qa_check --clip <clip_id>
"""
from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np

from src.common.config import load_config
from src.common.media import audio_present, count_frames, video_info
from src.common.paths import clip_paths
from src.common.schema import dump_json, load_json


def _frame_energy(video, n_samples: int):
    """帧间差分能量序列（下采样灰度，归一化）。"""
    cap = cv2.VideoCapture(str(video))
    small = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small.append(cv2.resize(gray, (64, 36)))
    cap.release()
    if len(small) < 3:
        return None
    arr = np.asarray(small, dtype=np.float32)
    energy = np.abs(np.diff(arr, axis=0)).reshape(len(arr) - 1, -1).mean(axis=1)
    idx = np.linspace(0, len(energy) - 1, min(n_samples, len(energy))).astype(int)
    x = energy[idx]
    return (x - x.mean()) / (x.std() + 1e-9)


def _audio_mean_volume(video) -> float | None:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video), "-map", "0:a",
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in r.stderr.splitlines():
        if "mean_volume" in line:
            try:
                return float(line.split(":")[1].replace("dB", "").strip())
            except ValueError:
                return None
    return None


def run(clip_id: str, cfg) -> None:
    paths = clip_paths(clip_id, cfg)
    meta = load_json(paths.meta)
    white = paths.white_video
    if not white.exists():
        raise RuntimeError("白模视频不存在，请先运行 compose")
    comp = load_json(paths.compliance_json, {})
    purity = load_json(paths.purity_json, {})
    n_qa = cfg.get("qa.motion_sample_frames", 120)

    checks: dict = {}

    # 1. 对齐
    try:
        wi = video_info(white)
        wf = count_frames(white)
        ok = (wf == meta["frame_count"]
              and abs(wi["fps"] - meta["fps"]) < 0.05
              and wi["width"] == meta["width"]
              and wi["height"] == meta["height"])
        checks["alignment"] = {
            "passed": ok,
            "white": {"frames": wf, "fps": wi["fps"],
                      "res": [wi["width"], wi["height"]]},
            "source": {"frames": meta["frame_count"], "fps": meta["fps"],
                       "res": [meta["width"], meta["height"]]},
            "standard": "0 错帧：帧数/时长/fps 完全一致",
        }
    except Exception as e:
        checks["alignment"] = {"passed": False, "error": str(e)}

    # 2. 运动同步（起势/动作/收势能量曲线相关）
    try:
        a = _frame_energy(white, n_qa)
        b = _frame_energy(paths.clip_video, n_qa)
        corr = (float(np.corrcoef(a, b)[0, 1])
                if a is not None and b is not None else -1.0)
        checks["motion_sync"] = {
            "passed": corr >= cfg.get("qa.motion_corr_min", 0.60),
            "corr": round(corr, 3),
            "standard": f"相关系数 ≥ {cfg.get('qa.motion_corr_min', 0.60)}",
        }
    except Exception as e:
        checks["motion_sync"] = {"passed": False, "error": str(e)}

    # 3. 纯净度
    checks["purity"] = {"passed": bool(purity.get("passed", False)),
                        "method": purity.get("method"),
                        "standard": "无字幕/台标/水印/UI（进线门禁）"}

    # 4. 白模合规
    checks["blockout_compliance"] = {
        "passed": bool(comp.get("passed", False)),
        "detail": {k: v for k, v in comp.items() if isinstance(v, bool)},
        "standard": "主体轮廓清晰、非空场景、简单图元、无特效",
    }

    # 5. 音频
    if meta.get("has_audio"):
        has = audio_present(white)
        vol = _audio_mean_volume(white) if has else None
        checks["audio"] = {
            "passed": has and vol is not None and vol > -60,
            "mean_volume_db": vol,
            "standard": "保留原 audio，无静音",
        }
    else:
        checks["audio"] = {"passed": True, "note": "源视频无音轨"}

    passed = all(c.get("passed", False) for c in checks.values())
    report = {"clip_id": clip_id, "passed": passed, "checks": checks,
              "sample_note": f"人工抽检比例 {cfg.get('qa.delivery_sample_rate', 0.30)}"
                             "（依据 side_by_side.mp4）"}
    dump_json(paths.qa_report, report)
    print(f"[qa] {clip_id}: {'PASS' if passed else 'FAIL'} → {paths.qa_report}")
    if not passed:
        failed = [k for k, c in checks.items() if not c.get("passed", False)]
        raise RuntimeError(f"质检未通过: {', '.join(failed)}（整批返工流程见 pair-qa skill）")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 7：成对自动质检")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config))


if __name__ == "__main__":
    main()
