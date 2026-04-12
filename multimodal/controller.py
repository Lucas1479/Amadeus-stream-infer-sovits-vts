"""multimodal/controller.py — 多模态输入监控与控制层

负责：
  - 管理 MultimodalInputManager（截图/摄像头输入监控）
  - 将视觉上下文转发给 LIVE 直连会话或 sidecar
  - 协调 LIVE API 与多模态监控的启停

运行时配置变量（可由 GUI 或脚本直接修改）：
  MULTIMODAL_ENABLED, MULTIMODAL_INPUT_SOURCE,
  MULTIMODAL_API_URL, MULTIMODAL_API_KEY, VOICE_MODE_ENABLED
"""

import asyncio
import base64
import json
import logging
import os
import time

import live.sidecar as _live_mod
from config.settings import VAD_HANGOVER_MS, VAD_ENERGY_THRESHOLD

logger = logging.getLogger(__name__)

# ===== 可选多模态依赖 =====
try:
    from tools.multimodal_input_manager import MultimodalInputManager
    _MM_AVAILABLE = True
except Exception as _mm_e:
    _MM_AVAILABLE = False
    MultimodalInputManager = None  # type: ignore
    logger.warning(f"多模态管理器不可用: {_mm_e}")

# ===== 运行时配置 =====
MULTIMODAL_ENABLED = False
MULTIMODAL_INPUT_SOURCE = "screen"          # "webcam" | "screen"
MULTIMODAL_API_URL = "http://localhost:9000/mm_api"
MULTIMODAL_API_KEY = ""
VOICE_MODE_ENABLED = False

# ===== 模块级状态 =====
_mm_manager = None
_mm_consumer_task = None
_mm_running_lock = asyncio.Lock()


# =============================================================================
# 启停
# =============================================================================

async def _start_multimodal_if_enabled():
    global _mm_manager, _mm_consumer_task
    if not MULTIMODAL_ENABLED:
        return
    if not _MM_AVAILABLE:
        logger.warning("已启用多模态开关但依赖不可用，跳过启动")
        return

    if _live_mod.LIVE_API_ENABLED:
        if _live_mod.USE_LIVE_SIDECAR:
            asyncio.create_task(_live_mod.start_live_api_sidecar())
        else:
            await _live_mod._start_live_session_if_enabled()

    async with _mm_running_lock:
        if _mm_manager is not None:
            return

        mic_idx = _live_mod._mic_device_index_getter() if _live_mod._mic_device_index_getter else None
        if _live_mod.USE_LIVE_SIDECAR:
            visual_config = {
                "input_source": MULTIMODAL_INPUT_SOURCE,
                "trigger_cooldown_sec": 30.0,
                "min_trigger_interval": 30.0,
                "motion_threshold": 200.0,
                "scene_hist_threshold": 0.95,
                "attach_video_clip": False,
                "input_device_index": mic_idx,
                "vad_energy_threshold": VAD_ENERGY_THRESHOLD,
                "vad_hangover_ms": VAD_HANGOVER_MS,
            }
        else:
            visual_config = {"input_source": MULTIMODAL_INPUT_SOURCE}

        _mm_manager = MultimodalInputManager(visual_config)
        try:
            await _mm_manager.start_monitoring()
            logger.info(f"多模态监控已启动，源: {MULTIMODAL_INPUT_SOURCE}")
        except Exception as e:
            logger.error(f"启动多模态监控失败: {e}")
            _mm_manager = None
            return

        async def _mm_consumer_loop():
            while MULTIMODAL_ENABLED and _mm_manager is not None:
                try:
                    packaged = await _mm_manager.output_queue.get()
                except asyncio.CancelledError:
                    break
                if packaged is None:
                    continue
                ts = packaged.get("timestamp", time.time())
                logger.info(f"[多模态] 收到视觉数据包，时间戳: {ts}, 来源: {MULTIMODAL_INPUT_SOURCE}")

                if (
                    _live_mod.LIVE_API_ENABLED
                    and _live_mod._live_session is not None
                    and _live_mod._LIVE_LIB_AVAILABLE
                ):
                    try:
                        await _send_visual_context_to_live(
                            _live_mod._live_session, packaged, attach_video=False
                        )
                        logger.info("[多模态] 已发送到LIVE session")
                    except Exception as e:
                        logger.error(f"发送LIVE视觉上下文失败: {e}")

                elif (
                    _live_mod.LIVE_API_ENABLED
                    and _live_mod.USE_LIVE_SIDECAR
                    and _live_mod._live_sidecar_proc is not None
                ):
                    try:
                        keyframe = packaged.get("keyframe")
                        video_path = packaged.get("video_path")
                        payload: dict = {"type": "visual", "data": {}}

                        if keyframe is not None:
                            try:
                                import cv2
                                h, w = keyframe.shape[:2]
                                max_side = 640
                                if max(h, w) > max_side:
                                    scale = max_side / max(h, w)
                                    keyframe = cv2.resize(
                                        keyframe,
                                        (int(w * scale), int(h * scale)),
                                        interpolation=cv2.INTER_AREA,
                                    )
                                ok, buf = cv2.imencode(
                                    '.jpg', keyframe,
                                    [cv2.IMWRITE_JPEG_QUALITY, 35],
                                )
                                if ok:
                                    payload["data"]["keyframe_b64"] = (
                                        base64.b64encode(buf.tobytes()).decode('ascii')
                                    )
                                    payload["data"]["keyframe_mime"] = "image/jpeg"
                                    logger.info(
                                        f"[多模态] 关键帧编码成功，大小: "
                                        f"{len(payload['data']['keyframe_b64'])} bytes"
                                    )
                            except Exception as _e:
                                logger.error(f"关键帧编码失败: {_e}")

                        if video_path and os.path.exists(video_path):
                            try:
                                with open(video_path, 'rb') as vf:
                                    payload["data"]["video_b64"] = (
                                        base64.b64encode(vf.read()).decode('ascii')
                                    )
                                    payload["data"]["video_mime"] = "video/mp4"
                                    logger.info("[多模态] 视频编码成功")
                            except Exception as _e:
                                logger.error(f"读取视频失败，忽略视频: {_e}")

                        if payload["data"]:
                            await _live_mod.send_command_to_sidecar(
                                json.dumps(payload, ensure_ascii=False)
                            )
                            logger.info("[多模态] 已发送视觉数据到 sidecar")
                        else:
                            logger.warning("[多模态] payload 为空，未发送")
                    except Exception as e:
                        logger.error(f"发送视觉上下文到 sidecar 失败: {e}")

        _mm_consumer_task = asyncio.create_task(_mm_consumer_loop())


async def _stop_multimodal():
    global _mm_manager, _mm_consumer_task
    async with _mm_running_lock:
        if _mm_consumer_task is not None:
            _mm_consumer_task.cancel()
            try:
                await _mm_consumer_task
            except Exception:
                pass
            _mm_consumer_task = None
        if _mm_manager is not None:
            try:
                await _mm_manager.stop_monitoring()
            except Exception:
                pass
            _mm_manager = None
        if _live_mod.LIVE_API_ENABLED:
            await _live_mod._stop_live_session()


# =============================================================================
# 辅助
# =============================================================================

async def _call_remote_multimodal_api(packaged):
    """占位：调用远程多模态 LLM 服务（当前仅返回占位文本）。"""
    # TODO: 使用 aiohttp 向 MULTIMODAL_API_URL 发送编码后的关键帧/视频
    return "占位返回：检测到显著事件，正在进行分析。"


async def _send_visual_context_to_live(session, packaged, attach_video: bool = False):
    """将触发的关键帧（+可选视频块）发送至 Google LIVE 会话。"""
    if not _live_mod._LIVE_LIB_AVAILABLE:
        raise RuntimeError("LIVE 库不可用")
    keyframe = packaged.get("keyframe")
    if keyframe is None:
        return
    try:
        import cv2
        ok, buf = cv2.imencode('.jpg', keyframe)
        if not ok:
            raise RuntimeError("关键帧编码失败")
        jpg_bytes = buf.tobytes()
    except Exception as e:
        logger.error(f"关键帧编码失败: {e}")
        return

    parts = []
    try:
        from google.genai import types as _t
        img_part = _t.Part.from_data(jpg_bytes, mime_type='image/jpeg')
        parts.append(img_part)
    except Exception as e:
        logger.error(f"构建图像部件失败: {e}")
        return

    if attach_video:
        video_path = packaged.get("video_path")
        if video_path and os.path.exists(video_path):
            try:
                file_obj = await _live_mod._live_client.aio.files.upload(file=video_path)
                from google.genai import types as _t  # noqa: F811
                video_part = _t.Part.from_uri(file_obj.uri, mime_type='video/mp4')
                parts.append(video_part)
            except Exception as e:
                logger.error(f"上传视频失败，忽略视频，仅发送关键帧: {e}")

    try:
        await session.send(input=parts, end_of_turn=False)
    except Exception as e:
        logger.error(f"LIVE 发送失败: {e}")
