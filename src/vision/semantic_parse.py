"""Stage 3：关键帧语义解析（多模型 LLM）→ 结构化 JSON → 场景资产库。

设计文档 §4.3 / §7.3：
- 系统 prompt 注入白模标准 + schema + 跨帧一致性规则（稳定前缀利于缓存）
- JSON schema 校验，失败自动重试 1 次，仍失败标记人工复核
- multi_subject（多人同框）→ flags.json 标记，整对打回（风险 #3）
- provider 可切换：anthropic / dashscope(Qwen-VL 推荐) / zhipu / doubao / moonshot
- 429/5xx 退避重试；4xx 降级 fallback 模型

用法：python -m src.vision.semantic_parse --clip <clip_id> [--frames 0,12,24]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.common.config import load_config
from src.common.paths import clip_paths
from src.common.schema import dump_json, load_json, validate_semantic_frame
from src.vision import llm_client
from src.vision.asset_store import SceneAssetStore

SYSTEM_PROMPT = """你是游戏视频白模（blockout）流水线的场景解析器。输出将被程序消费，必须只输出一个 JSON 对象。

## 白模建模标准（必须遵守）
- 几何仅允许：方块(cube)、球体(sphere)、圆柱(cylinder)、胶囊(capsule)；禁止精细网格
- 人物：头=球体，躯干=胶囊，四肢=胶囊x4；parts 固定为 {"head":"球体","torso":"胶囊","limbs":"胶囊x4"}
- 树木：圆柱树干 + 球形/锥形树冠；必须给 trunk_h 与 crown_r（米制）
- 建筑：保留主要体块、墙体、门洞、通道、台阶和平台；blocks 取值：["体块","墙体","门洞","通道","台阶","平台"]
- prop：primitive 取值：球体 / 方块 / 圆柱
- 所有对象给米制尺寸估计（size_est_m.h / size_est_m.w）
- 不解析：五官、服装、树叶、材质贴图、特效、UI 元素

## 输出 JSON schema
{
 "frame_id": "<原样返回输入的 frame_id>",
 "timestamp": <秒>,
 "scene_type": "fps|tps|rpg|arpg|moba|open_world|other",
 "environment": {"type": "outdoor_natural|outdoor_urban|indoor|other",
                 "lighting_dir": [x,y,z], "time": "daylight|dusk|night"},
 "objects": [{
   "id": "obj_01",
   "category": "character|building|tree|prop|vehicle|other",
   "name": "主角",
   "pose": "idle|walking|running|jumping|attacking|other",
   "screen_bbox": [x0,y0,x1,y1],
   "size_est_m": {"h": 1.8, "w": 0.6},
   "parts": {}, "blocks": [], "primitive": "", "trunk_h": 4.0, "crown_r": 2.5, "crown": "球形|锥形"
 }],
 "ground": {"type": "plane", "extent": <地面估计半径，米>},
 "occlusion": [{"a": "obj_02", "b": "obj_03", "rel": "a 在 b 前"}],
 "multi_subject": false
}

## 跨帧一致性规则
- 与前一关键帧摘要中相同实体必须沿用其 id；新实体用 obj_NN 递增新 id
- 类别不得跳变；尺寸估计与前一帧保持连续（漂移>30% 视为可疑）
- 若画面中出现多于 1 个可辨识人物主体，必须设置 "multi_subject": true（该片段将被打回）
"""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("响应中未找到 JSON")
    return json.loads(m.group(0))


def _user_text(fid: str, ts: float, prev_summary: dict, extra: str = "") -> str:
    return (f"frame_id: {fid}\ntimestamp: {ts:.2f}s\n"
            f"前一关键帧摘要: {json.dumps(prev_summary, ensure_ascii=False)}\n"
            "请解析本关键帧，只输出 JSON。" + extra)


def call_llm(image_path: Path, fid: str, ts: float,
             prev_summary: dict, cfg) -> dict:
    """单帧解析：JSON 解析失败带错误提示重试 1 次（§4.3）。"""
    provider = cfg.get("semantic.provider", "anthropic")
    content = llm_client.vision_content(
        image_path, _user_text(fid, ts, prev_summary), provider)
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            text = llm_client.chat(cfg, SYSTEM_PROMPT, content)
            return _extract_json(text)
        except json.JSONDecodeError as e:
            last_err = e
            content = llm_client.vision_content(
                image_path,
                _user_text(fid, ts, prev_summary,
                           "\n\n上面的输出不是合法 JSON。请重新输出，"
                           "只包含一个 JSON 对象，不要任何其它文本。"),
                provider)
    raise RuntimeError(f"语义解析失败: {last_err}")


def _prev_summary(frame_json: dict | None) -> dict:
    if not frame_json:
        return {}
    return {"timestamp": frame_json.get("timestamp"),
            "scene_type": frame_json.get("scene_type"),
            "objects": [{"id": o["id"], "category": o["category"],
                         "name": o.get("name", ""),
                         "size_est_m": o.get("size_est_m")}
                        for o in frame_json.get("objects", [])]}


def run(clip_id: str, cfg, frames: list[int] | None = None) -> None:
    paths = clip_paths(clip_id, cfg).ensure()
    index = load_json(paths.keyframes_index)
    if not index:
        raise RuntimeError("缺少 keyframes.json，请先运行 preprocess")
    store = SceneAssetStore(
        paths.asset_db,
        match_iou=cfg.get("semantic.match_iou", 0.30),
        drift_threshold=cfg.get("semantic.size_drift_threshold", 0.30))

    kfs = index["key_frames"]
    if frames:
        kfs = [k for k in kfs if k["frame"] in set(frames)]
    prev, parsed, failures = None, 0, []
    for kf in kfs:
        fid = f"{clip_id}_f{kf['frame']:06d}"
        img = paths.frames_key / kf["file"]
        try:
            fj = call_llm(img, fid, kf["timestamp"],
                          _prev_summary(prev), cfg)
        except Exception as e:
            failures.append({"frame": kf["frame"], "error": str(e)})
            continue  # 单帧失败不阻塞整对（§7.3）
        errors = validate_semantic_frame(fj)
        if errors:
            failures.append({"frame": kf["frame"], "schema_errors": errors})
            continue
        dump_json(paths.semantic_dir / f"f{kf['frame']:06d}.json", fj)
        store.ingest_frame(fj)
        prev = fj
        parsed += 1
        print(f"[semantic] {fid} 解析完成（{len(fj.get('objects', []))} 对象）")
    store.close()

    # flags：多人同框整对打回（风险 #3）+ 需人工复核清单
    scene = store_export(paths)
    flags = load_json(paths.flags_json, default={})
    chars = [a for a in scene["assets"] if a["category"] == "character"]
    if len(chars) > 1:
        flags["multi_subject"] = True
        flags["reject_reason"] = "同帧出现多个可辨识人物主体"
    if scene["needs_reparse"]:
        flags["needs_reparse"] = scene["needs_reparse"]
    if failures:
        flags["semantic_failures"] = failures
    dump_json(paths.flags_json, flags)
    dump_json(paths.scene_json, {**scene, "clip_id": clip_id})
    if parsed == 0:
        raise RuntimeError("语义解析全部关键帧失败，请检查 API/网络")
    print(f"[semantic] 完成 {parsed}/{len(kfs)} 帧 → scene.json "
          f"({len(scene['assets'])} 资产, needs_reparse={scene['needs_reparse']})")


def store_export(paths) -> dict:
    store = SceneAssetStore(paths.asset_db)
    try:
        return store.export_scene()
    finally:
        store.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3：关键帧语义解析")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", default=None, help="指定帧号，逗号分隔")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    frames = ([int(x) for x in args.frames.split(",")]
              if args.frames else None)
    run(args.clip, load_config(args.config), frames)


if __name__ == "__main__":
    main()
