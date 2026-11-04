"""统一 LLM 客户端（设计文档 §7）：Claude / 国内 OpenAI 兼容多模态模型。

国内 provider（均为 OpenAI 兼容接口，需 pip install openai）：
- dashscope 阿里通义 Qwen-VL（推荐：grounding/bbox 输出强、JSON 稳定、Batch 半价）
- zhipu     智谱 GLM-4V / GLM-4.5V
- doubao    火山引擎豆包 vision
- moonshot  Kimi vision
- openai_compatible  任意自定 base_url（配 semantic.base_url / api_key_env）
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

PROVIDERS = {
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                  "api_key_env": "DASHSCOPE_API_KEY"},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4",
              "api_key_env": "ZHIPUAI_API_KEY"},
    "doubao": {"base_url": "https://ark.cn-beijing.volces.com/api/v3",
               "api_key_env": "ARK_API_KEY"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1",
                 "api_key_env": "MOONSHOT_API_KEY"},
}


class LLMError(RuntimeError):
    pass


def vision_content(image_path: str | Path, user_text: str,
                   provider: str) -> list:
    """构造带图的 user 消息内容（多模态，按 provider 区分格式）。"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    if provider == "anthropic":
        return [{"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg",
                            "data": b64}},
                {"type": "text", "text": user_text}]
    return [{"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": user_text}]


def chat(cfg, system: str, content: list) -> str:
    """网络层重试（429/5xx 退避；4xx 降级 fallback 模型），返回文本。"""
    provider = cfg.get("semantic.provider", "anthropic")
    if provider == "anthropic":
        return _chat_anthropic(cfg, system, content)
    return _chat_openai(cfg, system, content)


def _retryable(name: str) -> bool:
    return any(k in name for k in ("RateLimit", "InternalServer", "Connection",
                                   "Overloaded", "Timeout"))


def _chat_anthropic(cfg, system: str, content: list) -> str:
    from anthropic import Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMError("未设置 ANTHROPIC_API_KEY 环境变量")
    client = Anthropic()
    model = cfg.get("semantic.model", "claude-opus-5-5")
    fallback = cfg.get("semantic.fallback_model")
    max_tokens = cfg.get("semantic.max_tokens", 8000)
    messages = [{"role": "user", "content": content}]
    last: Exception | None = None
    for attempt in range(cfg.get("semantic.retries", 3)):
        try:
            if cfg.get("semantic.use_streaming", False):
                with client.messages.stream(
                        model=model, max_tokens=max_tokens,
                        system=system, messages=messages) as stream:
                    return stream.get_final_text()
            resp = client.messages.create(
                model=model, max_tokens=max_tokens,
                system=system, messages=messages)
            return resp.content[0].text
        except Exception as e:
            last = e
            name = type(e).__name__
            if _retryable(name):
                time.sleep(2 ** attempt)
            elif fallback and model != fallback:
                model = fallback
            else:
                raise
    raise LLMError(f"LLM 调用失败: {last}")


def _chat_openai(cfg, system: str, content: list) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMError("缺少 openai 包：pip install openai") from e
    provider = cfg.get("semantic.provider", "dashscope")
    preset = PROVIDERS.get(provider, {})
    base_url = cfg.get("semantic.base_url") or preset.get("base_url")
    env = cfg.get("semantic.api_key_env") or preset.get("api_key_env",
                                                        "OPENAI_API_KEY")
    key = os.environ.get(env, "")
    if not key:
        raise LLMError(f"未设置 {env} 环境变量（provider={provider}）")
    if not base_url:
        raise LLMError(f"provider={provider} 缺少 base_url 配置")
    client = OpenAI(api_key=key, base_url=base_url)
    model = cfg.get("semantic.model", "qwen-vl-max")
    fallback = cfg.get("semantic.fallback_model")
    max_tokens = cfg.get("semantic.max_tokens", 8000)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": content}]
    last: Exception | None = None
    for attempt in range(cfg.get("semantic.retries", 3)):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=0)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last = e
            name = type(e).__name__
            status = str(getattr(e, "status_code", "") or getattr(e, "code", ""))
            if _retryable(name) or status.startswith(("429", "5")):
                time.sleep(2 ** attempt)
            elif fallback and model != fallback:
                model = fallback  # 降级路由（§7.1）
            else:
                raise
    raise LLMError(f"LLM 调用失败: {last}")
