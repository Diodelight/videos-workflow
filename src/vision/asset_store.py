"""场景资产库（跨帧一致性核心，设计文档 §3.2 / §4.3）。

SQLite 存储：assets（稳定对象）+ observations（逐帧观测）。
- 跨帧匹配：类别一致 + 屏幕 bbox IoU → 命中更新并融合尺寸；未命中新建资产
- 尺寸漂移 >30% → status=needs_reparse（触发重解析/人工复核）
- export_scene()：融合后场景 JSON（后续世界坐标融合的输入）
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets(
  id TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  name TEXT,
  size_h REAL,
  size_w REAL,
  size_n INTEGER DEFAULT 0,
  first_frame INTEGER,
  last_frame INTEGER,
  status TEXT DEFAULT 'ok',
  spec_json TEXT
);
CREATE TABLE IF NOT EXISTS observations(
  asset_id TEXT NOT NULL,
  frame_id INTEGER NOT NULL,
  timestamp REAL,
  bbox_json TEXT,
  cx REAL,
  cy REAL,
  depth REAL,
  obs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs ON observations(asset_id, frame_id);
"""

_COLS = ("id, category, name, size_h, size_w, size_n, "
         "first_frame, last_frame, status, spec_json")


def iou(a: list, b: list) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def frame_index_of(frame_id: str) -> int:
    m = re.search(r"(\d+)$", str(frame_id))
    return int(m.group(1)) if m else 0


class SceneAssetStore:
    def __init__(self, db_path: str | Path, match_iou: float = 0.30,
                 drift_threshold: float = 0.30):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self.match_iou = match_iou
        self.drift = drift_threshold

    def close(self) -> None:
        self.conn.close()

    # ---------- 内部 ----------
    def _exists(self, aid: str) -> bool:
        return self.conn.execute("SELECT 1 FROM assets WHERE id=?",
                                 (aid,)).fetchone() is not None

    def _open_assets(self, category: str) -> list[dict]:
        cur = self.conn.execute(
            f"SELECT {_COLS} FROM assets WHERE category=? AND status!='merged'",
            (category,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _last_obs(self, asset_id: str) -> dict | None:
        cur = self.conn.execute(
            "SELECT bbox_json FROM observations WHERE asset_id=? "
            "ORDER BY frame_id DESC LIMIT 1", (asset_id,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def _add_obs(self, aid: str, frame: dict, obj: dict, frame_id: int) -> None:
        bbox = obj["screen_bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.conn.execute(
            "INSERT INTO observations VALUES(?,?,?,?,?,?,?,?)",
            (aid, frame_id, frame["timestamp"], json.dumps(bbox),
             cx, cy, None, json.dumps(obj, ensure_ascii=False)))

    # ---------- 写入 ----------
    def ingest_frame(self, frame: dict) -> None:
        """frame: 已通过 schema 校验的语义 JSON。"""
        fid = frame_index_of(frame["frame_id"])
        for obj in frame.get("objects", []):
            bbox, best, best_score = obj["screen_bbox"], None, 0.0
            for cand in self._open_assets(obj["category"]):
                last = self._last_obs(cand["id"])
                if last is None:
                    continue
                score = iou(bbox, last)
                if score > best_score:
                    best, best_score = cand, score
            if best is not None and best_score >= self.match_iou:
                self._update(best, obj, frame, fid)
            else:
                self._create(obj, frame, fid)
        self.conn.commit()

    def _create(self, obj: dict, frame: dict, fid: int) -> None:
        aid, base, k = obj["id"], obj["id"], 1
        while self._exists(aid):
            aid = f"{base}_v{k}"
            k += 1
        size = obj.get("size_est_m") or {}
        self.conn.execute(
            f"INSERT INTO assets({_COLS}) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (aid, obj["category"], obj.get("name", ""),
             size.get("h"), size.get("w"), 1 if size.get("h") else 0,
             fid, fid, "ok", json.dumps(obj, ensure_ascii=False)))
        self._add_obs(aid, frame, obj, fid)

    def _update(self, asset: dict, obj: dict, frame: dict, fid: int) -> None:
        aid = asset["id"]
        size = obj.get("size_est_m") or {}
        h, w, n = asset["size_h"], asset["size_w"], asset["size_n"] or 0
        status = asset["status"]
        if size.get("h"):
            if h:
                if abs(size["h"] - h) / max(h, 1e-6) > self.drift and status == "ok":
                    status = "needs_reparse"  # 尺寸漂移超阈值 → 重解析
                w_cur = w if w else size["h"]
                w_new = size.get("w", size["h"])
                h = (h * n + size["h"]) / (n + 1)
                w = (w_cur * n + w_new) / (n + 1)
                n += 1
            else:
                h, w, n = size["h"], size.get("w", size["h"]), 1
        self.conn.execute(
            "UPDATE assets SET size_h=?, size_w=?, size_n=?, last_frame=?, "
            "status=?, spec_json=? WHERE id=?",
            (h, w, n, fid, status, json.dumps(obj, ensure_ascii=False), aid))
        self._add_obs(aid, frame, obj, fid)

    def mark_merged(self, keep_id: str, drop_id: str) -> None:
        """合并去重：drop_id 的观测并入 keep_id，drop 标记 merged。"""
        self.conn.execute("UPDATE observations SET asset_id=? WHERE asset_id=?",
                          (keep_id, drop_id))
        self.conn.execute("UPDATE assets SET status='merged' WHERE id=?",
                          (drop_id,))
        self.conn.commit()

    # ---------- 导出 ----------
    def export_scene(self) -> dict:
        cur = self.conn.execute(f"SELECT {_COLS} FROM assets")
        cols = [c[0] for c in cur.description]
        assets = []
        for row in cur.fetchall():
            a = dict(zip(cols, row))
            obs = self.conn.execute(
                "SELECT frame_id, timestamp, bbox_json, cx, cy FROM observations "
                "WHERE asset_id=? ORDER BY frame_id", (a["id"],)).fetchall()
            traj = [{"frame": o[0], "timestamp": o[1],
                     "bbox": json.loads(o[2]),
                     "screen_center": [o[3], o[4]]} for o in obs]
            assets.append({
                "id": a["id"], "category": a["category"], "name": a["name"],
                "size_m": {"h": a["size_h"], "w": a["size_w"]},
                "status": a["status"],
                "spec": json.loads(a["spec_json"]) if a["spec_json"] else {},
                "trajectory": traj,
            })
        return {"assets": assets,
                "needs_reparse": [a["id"] for a in assets
                                  if a["status"] == "needs_reparse"]}
