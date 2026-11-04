"""自研 vision-mcp server（设计文档 §5.2）。

把本地视觉推理封装为 MCP 工具，供调试期交互调用；
生产批量期由 Python 流水线直接调用同一套推理代码。

用法：python -m tools.vision_mcp.server    （stdio 传输，需 pip install mcp）
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.common.config import load_config
from src.vision import depth, pose, segment

mcp = FastMCP("vision")


@mcp.tool()
def estimate_depth(clip_id: str, target: str = "key") -> str:
    """对片段帧做单目深度估计（Depth Anything V2）。target: key|full"""
    ok = depth.run(clip_id, load_config(), target)
    return "完成" if ok else "跳过（本地模型未安装或已禁用，详见 depth.skipped）"


@mcp.tool()
def segment(clip_id: str) -> str:
    """对片段关键帧做 SAM2 实例分割与跨帧跟踪。"""
    ok = segment.run(clip_id, load_config())
    return "完成" if ok else "跳过（本地模型未安装或已禁用，详见 segment.skipped）"


@mcp.tool()
def estimate_pose(clip_id: str) -> str:
    """对片段关键帧做人物 2D 姿态估计（驱动白模人形）。"""
    ok = pose.run(clip_id, load_config())
    return "完成" if ok else "跳过（本地模型未安装或已禁用，详见 pose.skipped）"


if __name__ == "__main__":
    mcp.run()  # stdio
