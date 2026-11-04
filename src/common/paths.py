"""统一约定每个片段的分析目录布局（设计文档 §11 data/analysis）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.config import Config


@dataclass
class ClipPaths:
    clip_id: str
    cfg: Config

    # 输入
    @property
    def clip_video(self) -> Path:
        return self.cfg.path("paths.clips") / f"{self.clip_id}.mp4"

    # 分析目录
    @property
    def analysis(self) -> Path:
        return self.cfg.path("paths.analysis") / self.clip_id

    @property
    def meta(self) -> Path:
        return self.analysis / "meta.json"

    @property
    def frames_full(self) -> Path:
        return self.analysis / "frames_full"

    @property
    def frames_key(self) -> Path:
        return self.analysis / "frames_key"

    @property
    def keyframes_index(self) -> Path:
        return self.analysis / "keyframes.json"

    @property
    def audio(self) -> Path:
        return self.analysis / "audio.m4a"

    @property
    def purity_json(self) -> Path:
        return self.analysis / "purity.json"

    @property
    def semantic_dir(self) -> Path:
        return self.analysis / "semantic"

    @property
    def asset_db(self) -> Path:
        return self.analysis / "asset_store.sqlite"

    @property
    def scene_json(self) -> Path:
        return self.analysis / "scene.json"

    @property
    def flags_json(self) -> Path:
        return self.analysis / "flags.json"

    @property
    def depth_dir(self) -> Path:
        return self.analysis / "depth"

    @property
    def masks_dir(self) -> Path:
        return self.analysis / "masks"

    @property
    def poses2d_json(self) -> Path:
        return self.analysis / "poses2d.json"

    @property
    def poses_json(self) -> Path:
        return self.analysis / "poses.json"

    @property
    def sfm_dir(self) -> Path:
        return self.analysis / "sfm"

    @property
    def camera_json(self) -> Path:
        return self.analysis / "camera.json"

    @property
    def render_dir(self) -> Path:
        return self.analysis / "render"

    @property
    def blend_file(self) -> Path:
        return self.analysis / "blockout.blend"

    @property
    def compliance_json(self) -> Path:
        return self.analysis / "compliance.json"

    @property
    def white_video(self) -> Path:
        return self.analysis / "白模视频.mp4"

    @property
    def qa_report(self) -> Path:
        return self.analysis / "qa_report.json"

    # 交付目录 data/output/<clip_id>/
    @property
    def delivery_dir(self) -> Path:
        return self.cfg.path("paths.output") / self.clip_id

    def ensure(self) -> "ClipPaths":
        for p in (self.analysis, self.frames_full, self.frames_key,
                  self.semantic_dir, self.render_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def marker(self, stage: str) -> Path:
        return self.analysis / f"_done_{stage}.marker"


def clip_paths(clip_id: str, cfg: Config) -> ClipPaths:
    """工厂函数：按 clip_id 与全局配置构造路径集。"""
    return ClipPaths(clip_id=clip_id, cfg=cfg)
