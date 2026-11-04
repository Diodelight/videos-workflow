"""OpenCV 图像读写（Windows 中文/Unicode 路径安全）。

cv2.imwrite/imread 在 Windows 非 ASCII 路径下会静默失败/返回 None，
改走 numpy 的 tofile/fromfile（Python 层 Unicode 支持）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, img: np.ndarray,
            params: list | None = None) -> bool:
    ext = "." + str(path).rsplit(".", 1)[-1]
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True
