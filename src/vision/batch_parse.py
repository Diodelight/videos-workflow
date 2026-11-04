"""Batch API 批量语义解析（设计文档 §7.3：离线批量成本减半）。

与交互式 semantic_parse 的差异：批量帧之间无上下文链（各自独立解析），
适合例行帧；关键决策帧仍建议走 semantic_parse 单帧链路。

用法：
    python -m src.vision.batch_parse --clip <id> submit
    python -m src.vision.batch_parse --clip <id> status
    python -m src.vision.batch_parse --clip <id> fetch
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from src.common.config import load_config
from src.common.paths import clip_paths
from src.common.schema import dump_json, load_json, validate_semantic_frame
from src.vision.semantic_parse import SYSTEM_PROMPT, _extract_json
from src.vision.asset_store import SceneAssetStore


def _requests(clip_id: str, paths) -> list[dict]:
    index = load_json(paths.keyframes_index)
    reqs = []
    for kf in index["key_frames"]:
        img = paths.frames_key / kf["file"]
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        fid = f"{clip_id}_f{kf['frame']:06d}"
        reqs.append({
            "custom_id": fid,
            "params": {
                "model": None,  # 占位，submit 时填充
                "max_tokens": 8000,
                "system": SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/jpeg",
                                                     "data": b64}},
                        {"type": "text",
                         "text": f"frame_id: {fid}\ntimestamp: {kf['timestamp']:.2f}s\n"
                                 "批量模式（无前帧上下文），只输出 JSON。"},
                    ],
                }],
            },
        })
    return reqs


def _create(cfg):
    import os

    from anthropic import Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("未设置 ANTHROPIC_API_KEY 环境变量")
    return Anthropic()


def run(clip_id: str, cfg, action: str) -> None:
    paths = clip_paths(clip_id, cfg).ensure()
    if cfg.get("semantic.provider", "anthropic") != "anthropic":
        raise SystemExit("Batch API 目前仅支持 provider=anthropic；国内模型"
                         "请直接运行 python -m src.vision.semantic_parse "
                         "--clip <id>（逐帧解析，结果一致）")
    client = _create(cfg)
    model = cfg.get("semantic.model", "claude-opus-5-5")
    state_file = paths.analysis / "batch_state.json"

    if action == "submit":
        reqs = _requests(clip_id, paths)
        for r in reqs:
            r["params"]["model"] = model
        batch = client.messages.batches.create(requests=[
            {"custom_id": r["custom_id"], "params": r["params"]} for r in reqs])
        dump_json(state_file, {"batch_id": batch.id})
        print(f"[batch] 已提交 {len(reqs)} 帧，batch_id={batch.id}")

    elif action == "status":
        bid = load_json(state_file)["batch_id"]
        b = client.messages.batches.retrieve(bid)
        print(f"[batch] {bid}: {b.processing_status} "
              f"(succeeded={b.request_counts.succeeded}, "
              f"errored={b.request_counts.errored})")

    elif action == "fetch":
        bid = load_json(state_file)["batch_id"]
        b = client.messages.batches.retrieve(bid)
        if b.processing_status != "ended":
            print("[batch] 未结束，稍后再 fetch")
            sys.exit(1)
        store = SceneAssetStore(paths.asset_db)
        ok, fail = 0, 0
        for result in client.messages.batches.results(bid):
            if result.result.type != "succeeded":
                fail += 1
                continue
            text = "".join(
                blk.text for blk in result.result.message.content
                if getattr(blk, "type", "") == "text")
            try:
                fj = _extract_json(text)
                errors = validate_semantic_frame(fj)
                if errors:
                    raise ValueError(errors)
                frame_no = int(result.custom_id.rsplit("_f", 1)[-1])
                dump_json(paths.semantic_dir / f"f{frame_no:06d}.json", fj)
                store.ingest_frame(fj)
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[batch] {result.custom_id} 解析失败: {e}")
        store.close()
        scene = store_export(paths)
        dump_json(paths.scene_json, {**scene, "clip_id": clip_id})
        print(f"[batch] 完成：成功 {ok}，失败 {fail}")
    else:
        raise SystemExit(f"未知 action: {action}")


def store_export(paths):
    store = SceneAssetStore(paths.asset_db)
    try:
        return store.export_scene()
    finally:
        store.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch API 批量语义解析")
    ap.add_argument("--clip", required=True)
    ap.add_argument("action", choices=["submit", "status", "fetch"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.clip, load_config(args.config), args.action)


if __name__ == "__main__":
    main()
