"""
OpenClaw API 客户端
- _get_openclaw_client：懒加载 OpenAI 兼容客户端
- ask_openclaw：单次同步调用（via asyncio.to_thread）
- ask_openclaw_stream：流式调用
- _classify_openclaw_result：结果类型判断（ok / question / error / partial）
"""
from __future__ import annotations

import asyncio
import logging
import os

from openai import OpenAI

from config.settings import OPENCLAW_BASE_URL, OPENCLAW_TOKEN

logger = logging.getLogger(__name__)

_openclaw_client: OpenAI | None = None


def _get_openclaw_client() -> OpenAI:
    """懒加载 OpenClaw 客户端（复用连接池）。"""
    global _openclaw_client
    if _openclaw_client is None:
        _openclaw_client = OpenAI(
            api_key=OPENCLAW_TOKEN,
            base_url=f"{OPENCLAW_BASE_URL}/v1",
            max_retries=0,
        )
        logger.info(f"[openclaw] 客户端已初始化，Gateway: {OPENCLAW_BASE_URL}")
    return _openclaw_client


async def ask_openclaw(
    task: str,
    timeout: float = 120.0,
    image_path: str | None = None,
) -> str:
    """
    向 OpenClaw 发送任务指令，返回执行结果文本。

    参数：
        image_path: 可选本地图片路径；支持 vision 时以 base64 附加，否则降级为路径文本。
    """
    try:
        client = _get_openclaw_client()
        system_msg = (
            "You are an execution assistant. Execute the given task directly and concisely. "
            "Do NOT ask for clarification or confirmation — just attempt the task and report what you did."
        )

        if image_path and os.path.exists(image_path):
            try:
                import base64 as _b64
                ext = os.path.splitext(image_path)[1].lower().lstrip(".")
                mime = {
                    "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp",
                    "gif": "image/gif",
                }.get(ext, "image/png")
                with open(image_path, "rb") as f:
                    b64_data = _b64.b64encode(f.read()).decode()
                user_content = [
                    {"type": "text", "text": task},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_data}"}},
                ]
                logger.info(f"[openclaw] 附加图片: {image_path} ({mime})")
            except Exception as img_e:
                logger.warning(f"[openclaw] 图片编码失败({img_e})，降级为路径传递")
                user_content = f"{task}\n[参考图片路径: {image_path}]"
        else:
            user_content = task

        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="openclaw",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content},
                ],
                timeout=timeout,
            )
        )
        result = response.choices[0].message.content or ""
        logger.info(f"[openclaw] 任务完成: {task[:40]}... -> {result[:60]}...")
        return result
    except Exception as e:
        logger.warning(f"[openclaw] 调用失败: {e}")
        return f"[OpenClaw 暂时不可用: {e}]"


async def ask_openclaw_stream(
    task: str,
    chunk_callback=None,
    timeout: float = 60.0,
) -> str:
    """流式调用 OpenClaw；chunk_callback(chunk: str) 在每个文本块到达时被调用。"""
    full_text = ""
    try:
        client = _get_openclaw_client()

        def _stream():
            nonlocal full_text
            with client.chat.completions.create(
                model="openclaw",
                messages=[{"role": "user", "content": task}],
                stream=True,
                timeout=timeout,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        full_text += delta
                        if chunk_callback:
                            chunk_callback(delta)

        await asyncio.to_thread(_stream)
        logger.info(f"[openclaw] 流式任务完成: {task[:40]}...")
    except Exception as e:
        logger.warning(f"[openclaw] 流式调用失败: {e}")
        full_text = f"[OpenClaw 暂时不可用: {e}]"
    return full_text


def _classify_openclaw_result(result: str) -> str:
    """
    判断 OpenClaw 返回结果的类型：
    - "question"：OpenClaw 在反问，需要 Kurisu 转达给用户
    - "error"：执行出错，需要 Kurisu 说明情况并请求用户协助
    - "partial"：工具能力受限，部分完成
    - "ok"：正常完成
    """
    lowered = result.lower()
    question_signals = [
        "?", "？",
        "which", "what", "would you", "do you want", "please specify", "please clarify",
        "どうし", "どれ", "どちら", "何を", "教えてください", "確認", "選んで",
    ]
    error_signals = [
        "error", "failed", "permission denied", "not found", "exception", "traceback",
        "エラー", "失敗", "見つかりません", "権限", "アクセス拒否",
    ]
    capability_limit_signals = [
        "設定されていない", "api key", "apiキー", "利用できません", "アクセスできません",
        "not configured", "not available", "not enabled", "not supported",
        "できません", "できなかった", "サポートされていません",
    ]
    if any(s in lowered for s in error_signals):
        return "error"
    if any(s in lowered for s in capability_limit_signals):
        return "partial"
    if any(s in result for s in question_signals):
        return "question"
    return "ok"
