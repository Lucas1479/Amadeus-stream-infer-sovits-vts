"""
本地 llama-server 生命周期管理
- start_llama_server：检测端口 → 拉起子进程 → 健康检查等待
- warmup_local_llm_cache：静默预热 Prompt Cache
- stop_llama_server：安全终止子进程
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess

import aiohttp

from config.settings import (
    LOCAL_LLM_URL,
    LOCAL_LLM_CLI_PATH,
    LOCAL_LLM_CLI_ARGS,
    LOCAL_LLM_MODEL,
    LOCAL_LLM_TYPE,
    USE_LOCAL_LLM,
)

logger = logging.getLogger(__name__)

_llm_server_proc: subprocess.Popen | None = None


async def start_llama_server() -> None:
    """启动本地 llama-server 服务。"""
    global _llm_server_proc

    health_url = LOCAL_LLM_URL.removesuffix("/v1") + "/health"

    # 1. 优先检查端口：已有服务则直接复用
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=1)) as resp:
                if resp.status == 200:
                    logger.info("✅ 检测到 llama-server 已在运行，将直接连接使用")
                    return
    except Exception:
        pass  # 端口未通，继续启动

    if _llm_server_proc is not None and _llm_server_proc.poll() is None:
        logger.info("✅ llama-server 已在运行中")
        return

    if not LOCAL_LLM_CLI_PATH:
        logger.error("❌ LOCAL_LLM_CLI_PATH 未设置，无法启动 llama-server")
        return

    if not os.path.exists(LOCAL_LLM_CLI_PATH):
        logger.error(f"❌ 找不到 llama-server 可执行文件: {LOCAL_LLM_CLI_PATH}")
        return

    cmd = [LOCAL_LLM_CLI_PATH] + LOCAL_LLM_CLI_ARGS
    server_dir = os.path.dirname(LOCAL_LLM_CLI_PATH)

    env = os.environ.copy()
    env["PATH"] = server_dir + os.pathsep + env.get("PATH", "")

    logger.info(f"🐛 DEBUG: CLI_ARGS = {LOCAL_LLM_CLI_ARGS}")
    logger.info(f"🚀 启动 llama-server: {' '.join(cmd)}")
    logger.info(f"📂 工作目录: {server_dir}")

    try:
        log_file = open("llama_server.log", "w", encoding="utf-8")
        _llm_server_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=server_dir,
            env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )
        logger.info(f"✅ llama-server 进程已启动 (PID: {_llm_server_proc.pid})")

        # 2. 轮询健康检查，最多等待 15 秒
        logger.info("⏳ 等待服务就绪...")
        for _ in range(30):
            if _llm_server_proc.poll() is not None:
                logger.error(
                    f"❌ llama-server 启动后立即退出，返回码: {_llm_server_proc.returncode}"
                )
                logger.error("请查看 llama_server.log 获取详细错误信息")
                _llm_server_proc = None
                return
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        health_url, timeout=aiohttp.ClientTimeout(total=1)
                    ) as resp:
                        if resp.status == 200:
                            logger.info("✅ llama-server 健康检查通过，服务已就绪")
                            return
            except Exception:
                pass
            await asyncio.sleep(0.5)

        logger.warning("⚠️ llama-server 启动超时，可能尚未完全就绪")

    except Exception as e:
        logger.error(f"❌ 启动 llama-server 失败: {e}")
        _llm_server_proc = None


async def warmup_local_llm_cache() -> None:
    """
    本地 llama-server Prompt Cache 静默预热。
    仅在 USE_LOCAL_LLM + llama_server 模式时触发；不写入会话历史。
    """
    if not (USE_LOCAL_LLM and LOCAL_LLM_TYPE == "llama_server"):
        return

    try:
        base_url = LOCAL_LLM_URL.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        api_url = f"{base_url}/chat/completions"

        system_prompt = (
            "あなたは牧瀬紅莉栖で,優秀で理知的な性格です.少しツンデレで,でも根は優しい."
            "必ず日本語のみで短く自然に答えてください。"
        )
        warmup_user = "これは事前ウォームアップ用のテストです。一言だけ返事してください。"

        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": warmup_user},
            ],
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 32,
            "cache_prompt": True,
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(api_url, json=payload) as resp:
                if resp.status == 200:
                    await resp.json()
                    logger.info("✅ 本地LLM Prompt Cache 预热完成")
                else:
                    logger.warning(f"⚠️ 本地LLM预热请求返回非200状态码: {resp.status}")
    except Exception as e:
        logger.warning(f"⚠️ 本地LLM Prompt Cache 预热失败（忽略，不影响主流程）: {e}")


def stop_llama_server() -> None:
    """安全停止由本程序启动的 llama-server 子进程。"""
    global _llm_server_proc
    if not _llm_server_proc:
        return
    logger.info("🛑 正在停止 llama-server...")
    try:
        _llm_server_proc.terminate()
        try:
            _llm_server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _llm_server_proc.kill()
        logger.info("✅ llama-server 已停止")
    except Exception as e:
        logger.error(f"⚠️ 停止 llama-server 时出错: {e}")
    finally:
        _llm_server_proc = None
