"""live/sidecar.py — Google LIVE API 直连会话 + Sidecar 子进程管理

负责：
  - 直连 LIVE 会话（_start_live_session_if_enabled / _stop_live_session）
  - 本地 sidecar 子进程（start_live_api_sidecar / stop_live_api_sidecar）
  - 向 sidecar 发送命令（send_command_to_sidecar）

依赖注入（通过 configure() 在 main() 初始化后调用）：
  - text_handler_fn          : stream_llm_postprocess_text，供 sidecar 文本回调使用
  - mm_manager_getter        : lambda → multimodal._mm_manager（VAD 状态监听用）
  - multimodal_input_source_getter : lambda → multimodal.MULTIMODAL_INPUT_SOURCE
  - mic_device_index_getter  : lambda → ASRManager.MICROPHONE_DEVICE_INDEX
"""

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

# ===== 可选 Google LIVE API 库 =====
try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _LIVE_LIB_AVAILABLE = True
except Exception as _live_e:
    _genai = None
    _genai_types = None
    _LIVE_LIB_AVAILABLE = False
    logger.warning(f"LIVE API 库不可用: {_live_e}")

# ===== 运行时配置（可由 GUI 切换）=====
LIVE_API_ENABLED = False
LIVE_MODEL_NAME = "gemini-2.5-flash-native-audio-preview-12-2025"
LIVE_API_KEY = os.environ.get("GEMINI_API_KEY", "")
try:
    if not LIVE_API_KEY:
        from config.settings import GEMINI_API_KEY as _cfg_key
        LIVE_API_KEY = _cfg_key
except Exception:
    pass

USE_LIVE_SIDECAR = True  # 默认使用 sidecar，避免依赖冲突

# ===== 模块级状态 =====
_live_client = None
_live_session = None
_live_receive_task = None
_live_lock = asyncio.Lock()
_live_sidecar_proc = None
_live_sidecar_reader_task = None
_live_text_buf = ""
_live_text_buf_task = None
_live_text_coalesce_ms = 400
_mm_state_watch_task = None
_live_warmup_skip = 0

# ===== 依赖注入占位 =====
_text_handler_fn = None
_mm_manager_getter = None
_multimodal_input_source_getter = None
_mic_device_index_getter = None


def configure(
    text_handler_fn=None,
    mm_manager_getter=None,
    multimodal_input_source_getter=None,
    mic_device_index_getter=None,
):
    """在 async main() 完成基础初始化后调用，注入运行时回调。"""
    global _text_handler_fn, _mm_manager_getter
    global _multimodal_input_source_getter, _mic_device_index_getter
    if text_handler_fn is not None:
        _text_handler_fn = text_handler_fn
    if mm_manager_getter is not None:
        _mm_manager_getter = mm_manager_getter
    if multimodal_input_source_getter is not None:
        _multimodal_input_source_getter = multimodal_input_source_getter
    if mic_device_index_getter is not None:
        _mic_device_index_getter = mic_device_index_getter


# =============================================================================
# 内部工具
# =============================================================================

def _build_persona_prompt_for_live() -> str:
    """LIVE 模型专用人设 prompt。"""
    return (
        "You are Makise Kurisu. You MUST answer in Japanese strictly. No Chinese allowed.\n"
        "あなたは牧瀬紅莉栖です。日本の科学者であり、母語は日本語です。\n"
        "【絶対遵守】\n"
        "1) 必ず日本語でのみ回答すること。ユーザーの言語が何であっても、日本語以外は一切使用しない。\n"
        "2) 中国語の文字・語句を絶対に使用しない。\n"
        "3) 自然で口語的な文体を保ち、牧瀬紅莉栖として一貫した口調・性格で話す。"
        "ユーザーに頼まれたタスクには必ず応じること。\n"
        "4) 推論過程や思考の連鎖は開示しない（結論のみ提示）。\n"
    )


def _split_at_last_sentence_boundary(buf: str):
    """从缓冲区取出到最后一个句子结束符为止的完整部分，返回 (complete, remaining)。
    方括号内（表情标签）的标点不算句子结束。"""
    _ENDINGS = frozenset("。！？!?\n")
    last_boundary = -1
    depth = 0
    for i, ch in enumerate(buf):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth = max(0, depth - 1)
        elif depth == 0 and ch in _ENDINGS:
            last_boundary = i
    if last_boundary < 0:
        return "", buf
    return buf[:last_boundary + 1], buf[last_boundary + 1:]


# =============================================================================
# LIVE 直连会话
# =============================================================================

async def _start_live_session_if_enabled():
    global _live_client, _live_session, _live_receive_task
    if not LIVE_API_ENABLED:
        return
    if not _LIVE_LIB_AVAILABLE:
        logger.warning("已启用LIVE开关但依赖不可用，跳过启动")
        return
    if not LIVE_API_KEY:
        logger.warning("LIVE API KEY 未配置，跳过启动")
        return
    async with _live_lock:
        if _live_session is not None:
            return
        try:
            _live_client = _genai.Client(
                http_options={"api_version": "v1beta"},
                api_key=LIVE_API_KEY,
            )
            live_config = _genai_types.LiveConnectConfig(
                response_modalities=["TEXT"],
                media_resolution="MEDIA_RESOLUTION_MEDIUM",
            )
            _live_session = await _live_client.aio.live.connect(
                model=LIVE_MODEL_NAME, config=live_config
            ).__aenter__()
            logger.info(f"LIVE 会话已连接: {LIVE_MODEL_NAME}")

            async def _receive_text_loop():
                try:
                    while LIVE_API_ENABLED and _live_session is not None:
                        turn = _live_session.receive()
                        async for resp in turn:
                            if getattr(resp, 'text', None):
                                logger.info(f"[LIVE TEXT] {resp.text}")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"LIVE 文本接收异常: {e}")

            _live_receive_task = asyncio.create_task(_receive_text_loop())
        except Exception as e:
            logger.error(f"启动LIVE会话失败: {e}")
            _live_session = None


async def _stop_live_session():
    global _live_client, _live_session, _live_receive_task
    async with _live_lock:
        if _live_receive_task is not None:
            _live_receive_task.cancel()
            try:
                await _live_receive_task
            except Exception:
                pass
            _live_receive_task = None
        if _live_session is not None:
            try:
                await _live_session.__aexit__(None, None, None)
            except Exception:
                pass
            _live_session = None
        _live_client = None


# =============================================================================
# Sidecar 子进程管理
# =============================================================================

async def start_live_api_sidecar():
    global _live_sidecar_proc, _live_sidecar_reader_task, _mm_state_watch_task
    if _live_sidecar_proc is not None:
        return
    try:
        venv_python = os.path.join(os.getcwd(), ".venv_live", "Scripts", "python.exe")
        if not os.path.exists(venv_python):
            logger.warning(".venv_live 环境未找到，无法启动sidecar")
            return
        script_path = os.path.join(os.getcwd(), "live_api_sidecar.py")
        env = os.environ.copy()
        if LIVE_API_KEY:
            env["LIVE_API_KEY"] = LIVE_API_KEY
        if os.environ.get("GEMINI_API_KEY"):
            env["GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY"]
        mic_idx = _mic_device_index_getter() if _mic_device_index_getter else None
        if mic_idx is not None:
            env["LIVE_MIC_DEVICE"] = str(mic_idx)
        env["LIVE_SYSTEM_INSTRUCTION"] = _build_persona_prompt_for_live()
        mm_source = _multimodal_input_source_getter() if _multimodal_input_source_getter else "screen"
        sidecar_mode = (
            "camera" if mm_source == "webcam"
            else ("screen" if mm_source == "screen" else "none")
        )
        _live_sidecar_proc = await asyncio.create_subprocess_exec(
            venv_python, script_path,
            "--mode", sidecar_mode,
            "--model", LIVE_MODEL_NAME,
            "--api_key", LIVE_API_KEY or "",
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        logger.info("LIVE sidecar 已启动")

        # ── 文本处理回调 ──────────────────────────────────────────────────────
        async def _process_live_result(text: str):
            try:
                if not text:
                    return
                logger.info(f"[LIVE→DEBUG] RAW len={len(text)} preview={text[:40]}")
                if _text_handler_fn is not None:
                    await _text_handler_fn(text, source="LIVE_RAW")
            except Exception as e:
                logger.error(f"处理LIVE结果失败: {e}")

        # ── 文本缓冲 flush ─────────────────────────────────────────────────────
        async def _flush_live_text_buf():
            global _live_text_buf, _live_text_buf_task
            buf = _live_text_buf.strip()
            _live_text_buf_task = None
            if not buf:
                _live_text_buf = ""
                return
            _SENT_ENDINGS = frozenset("。！？!?\n")
            ends_with_terminator = bool(buf) and buf[-1] in _SENT_ENDINGS
            open_count = buf.count('[') - buf.count(']')
            if open_count > 0:
                _live_text_buf = buf
                _live_text_buf_task = asyncio.create_task(asyncio.sleep(1.5))

                def _done_extended(task):
                    if not task.cancelled():
                        asyncio.create_task(_flush_live_text_buf())
                _live_text_buf_task.add_done_callback(_done_extended)
                return
            stripped_no_tag = ''.join(ch for ch in buf if ch not in ('[]'))
            visible_len = len(stripped_no_tag.replace(' ', ''))
            if visible_len < 6 and not ends_with_terminator:
                _live_text_buf = buf
                _live_text_buf_task = asyncio.create_task(asyncio.sleep(0.25))

                def _done_short(task):
                    if not task.cancelled():
                        asyncio.create_task(_flush_live_text_buf())
                _live_text_buf_task.add_done_callback(_done_short)
                return
            _live_text_buf = ""
            await _process_live_result(buf)

        # ── sidecar stdout 读取协程 ────────────────────────────────────────────
        async def read_from_sidecar():
            try:
                assert _live_sidecar_proc.stdout is not None
                while True:
                    line = await _live_sidecar_proc.stdout.readline()
                    if not line:
                        break
                    try:
                        msg = json.loads(line.decode('utf-8').strip())
                    except Exception:
                        continue
                    mtype = msg.get("type")
                    data = msg.get("data")
                    if mtype != "turn_complete":
                        logger.info(f"[LIVE SIDECAR RX] {mtype}: {str(data)[:80]}")

                    if mtype == "text" and data:
                        global _live_text_buf, _live_text_buf_task
                        _live_text_buf += str(data)
                        complete, _live_text_buf = _split_at_last_sentence_boundary(_live_text_buf)
                        if complete.strip():
                            asyncio.create_task(_process_live_result(complete.strip()))
                        if _live_text_buf_task is not None:
                            try:
                                _live_text_buf_task.cancel()
                            except Exception:
                                pass
                        _live_text_buf_task = asyncio.create_task(
                            asyncio.sleep(_live_text_coalesce_ms / 1000.0)
                        )

                        def _done(task):
                            if not task.cancelled():
                                asyncio.create_task(_flush_live_text_buf())
                        _live_text_buf_task.add_done_callback(_done)

                    elif mtype == "turn_complete":
                        if _live_text_buf_task is not None:
                            try:
                                _live_text_buf_task.cancel()
                            except Exception:
                                pass
                            _live_text_buf_task = None
                        if _live_text_buf.strip():
                            asyncio.create_task(_flush_live_text_buf())
                    elif mtype == "error":
                        logger.error(f"[LIVE SIDECAR ERROR] {data}")
                    elif mtype == "status":
                        logger.info(f"[LIVE SIDECAR] {data}")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"读取sidecar失败: {e}")

        _live_sidecar_reader_task = asyncio.create_task(read_from_sidecar())

        # ── VAD → 麦克风开关监听 ───────────────────────────────────────────────
        async def _watch_mm_state_for_mic():
            prev = None
            try:
                while _live_sidecar_proc is not None and _live_sidecar_proc.returncode is None:
                    mm_manager = _mm_manager_getter() if _mm_manager_getter else None
                    st = getattr(mm_manager, 'state', None) if mm_manager else None
                    if st != prev:
                        if st == "LISTENING_TO_USER":
                            logger.info("[VAD→Live] 用户开始说话，发送 mic_on")
                            try:
                                await send_command_to_sidecar(
                                    json.dumps({"type": "mic_on"}, ensure_ascii=False)
                                )
                            except Exception as _e:
                                logger.warning(f"[VAD→Live] mic_on 发送失败: {_e}")
                        elif prev == "LISTENING_TO_USER" and st != "LISTENING_TO_USER":
                            logger.info(f"[VAD→Live] 用户停止说话（{prev}→{st}），发送 turn_end + mic_off")
                            try:
                                await send_command_to_sidecar(
                                    json.dumps({"type": "turn_end"}, ensure_ascii=False)
                                )
                                await asyncio.sleep(0.05)
                                await send_command_to_sidecar(
                                    json.dumps({"type": "mic_off"}, ensure_ascii=False)
                                )
                            except Exception as _e:
                                logger.warning(f"[VAD→Live] turn_end/mic_off 发送失败: {_e}")
                        prev = st
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"mm状态监听失败: {e}")

        _mm_state_watch_task = asyncio.create_task(_watch_mm_state_for_mic())
    except Exception as e:
        logger.error(f"启动sidecar失败: {e}")


async def stop_live_api_sidecar():
    global _live_sidecar_proc, _live_sidecar_reader_task, _mm_state_watch_task, _live_text_buf_task
    if _live_sidecar_reader_task is not None:
        _live_sidecar_reader_task.cancel()
        try:
            await _live_sidecar_reader_task
        except Exception:
            pass
        _live_sidecar_reader_task = None
    if _mm_state_watch_task is not None:
        _mm_state_watch_task.cancel()
        try:
            await _mm_state_watch_task
        except Exception:
            pass
        _mm_state_watch_task = None
    try:
        await send_command_to_sidecar(json.dumps({"type": "mic_off"}))
    except Exception:
        pass
    if _live_sidecar_proc is not None:
        try:
            if _live_sidecar_proc.stdin:
                try:
                    _live_sidecar_proc.stdin.close()
                except Exception:
                    pass
            _live_sidecar_proc.terminate()
            await _live_sidecar_proc.wait()
        except Exception:
            pass
        _live_sidecar_proc = None


async def send_command_to_sidecar(command):
    """向 sidecar stdin 发送一条 JSON 命令。command 可为 str 或 dict。"""
    if _live_sidecar_proc is None or _live_sidecar_proc.stdin is None:
        return
    try:
        try:
            _ = json.loads(command) if isinstance(command, str) else None
            payload = (command if isinstance(command, str) else json.dumps(command, ensure_ascii=False)) + "\n"
        except Exception:
            payload = json.dumps(command, ensure_ascii=False) + "\n"
        if _live_sidecar_proc.returncode is not None:
            return
        try:
            _live_sidecar_proc.stdin.write(payload.encode('utf-8'))
            await _live_sidecar_proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            logger.error("发送命令到sidecar失败: Connection lost")
    except Exception as e:
        logger.error(f"发送命令到sidecar失败: {e}")


async def live_send_test_text(text: str = "テスト：何か一言お願いします。"):
    try:
        payload = json.dumps({"type": "text", "data": text}, ensure_ascii=False)
        await send_command_to_sidecar(payload)
        logger.info("[LIVE QUICK TEST] 已发送测试文本")
    except Exception as e:
        logger.error(f"[LIVE QUICK TEST] 发送失败: {e}")
