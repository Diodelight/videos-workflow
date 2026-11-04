"""全局配置加载：config.yaml → 支持点路径访问的 Config 对象。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录


class Config:
    def __init__(self, data: dict[str, Any]):
        self._data = data or {}

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """配置项解析为绝对路径；相对路径基于项目根。"""
        raw = self.get(dotted)
        if raw is None:
            raise KeyError(f"config 缺少配置项: {dotted}")
        p = Path(str(raw))
        return p if p.is_absolute() else ROOT / p

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)
