"""结构化 JSON：语义解析输出 schema（设计文档 §4.3）与 JSON 读写工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

CATEGORIES = ["character", "building", "tree", "prop", "vehicle", "other"]
SCENE_TYPES = ["fps", "tps", "rpg", "arpg", "moba", "open_world", "other"]

_BBOX = {"type": "array", "items": {"type": "number"},
         "minItems": 4, "maxItems": 4}

SEMANTIC_FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["frame_id", "timestamp", "scene_type", "objects", "ground"],
    "properties": {
        "frame_id": {"type": "string"},
        "timestamp": {"type": "number"},
        "scene_type": {"enum": SCENE_TYPES},
        "multi_subject": {"type": "boolean"},
        "environment": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "lighting_dir": {"type": "array", "items": {"type": "number"},
                                 "minItems": 3, "maxItems": 3},
                "time": {"type": "string"},
            },
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "category", "screen_bbox"],
                "properties": {
                    "id": {"type": "string", "pattern": r"^obj_[A-Za-z0-9_]+$"},
                    "category": {"enum": CATEGORIES},
                    "name": {"type": "string"},
                    "pose": {"type": "string"},
                    "screen_bbox": _BBOX,
                    "size_est_m": {"type": "object",
                                   "properties": {"h": {"type": "number"},
                                                  "w": {"type": "number"}}},
                    "parts": {"type": "object"},
                    "blocks": {"type": "array", "items": {"type": "string"}},
                    "trunk_h": {"type": "number"},
                    "crown_r": {"type": "number"},
                    "crown": {"type": "string"},
                    "primitive": {"type": "string"},
                },
            },
        },
        "ground": {"type": "object", "required": ["type"],
                   "properties": {"type": {"type": "string"},
                                  "extent": {"type": "number"}}},
        "occlusion": {"type": "array",
                      "items": {"type": "object", "required": ["a", "b", "rel"],
                                "properties": {"a": {"type": "string"},
                                               "b": {"type": "string"},
                                               "rel": {"type": "string"}}}},
        "notes": {"type": "string"},
    },
}

_VALIDATOR = Draft202012Validator(SEMANTIC_FRAME_SCHEMA)


def validate_semantic_frame(obj: dict) -> list[str]:
    """返回错误列表；空列表 = 通过。"""
    return [f"{list(e.path)}: {e.message}" for e in _VALIDATOR.iter_errors(obj)]


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
