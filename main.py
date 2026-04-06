import os
import sys
import time
import json
import base64
import asyncio

import logger
import numpy as np
import soundfile as sf
import pyaudio
import re
import traceback
from queue import Queue
from threading import Thread, Lock
from openai import OpenAI
import requests
import aiohttp
import websocket as ws
import random
import logging
import tempfile
import sys
from PyQt5.QtWidgets import QApplication
from sympy.physics.units import sr

from chatGui import launch_subtitle_gui
import google.generativeai as google_genai
import speech_recognition as speech_rec
import torch

# 本地 Kurisu 专用 RAG 知识库
from rag_system import RAGSystem

# 导入悬浮字幕窗
try:
    from floating_subtitle import init_subtitle_window, update_subtitle_text
    SUBTITLE_AVAILABLE = True
    subtitle_window_instance = None  # 在main.py中定义全局变量
except ImportError:
    SUBTITLE_AVAILABLE = False
    subtitle_window_instance = None
    logger.warning("悬浮字幕窗模块未找到,字幕功能将不可用")

# 加载本地TTS推理
import sys
import os
from collections import deque
root_dir = os.path.dirname(os.path.abspath(__file__))
gpt_sovits_dir = os.path.join(root_dir, "GPT_SoVITS")
sys.path.insert(0, gpt_sovits_dir)
sys.path.insert(0, root_dir)

from local_tts_infer import TTSInferencer

# 设置更详细的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vits_app.log')  # 同时输出到文件
    ]
)
logger = logging.getLogger('vts_connector')

# 将其他库的日志级别调高,减少干扰
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websocket").setLevel(logging.WARNING)

llm_client = None

# ── 从统一配置模块导入所有设置 ────────────────────────────────────────────────
from config.settings import (
    # RAG
    RAG_ENABLED_FOR_LOCAL, RAG_TOP_K, RAG_MAX_DISTANCE,
    # 本地 LLM
    USE_LOCAL_LLM, LOCAL_LLM_TYPE, LOCAL_LLM_MODEL,
    LOCAL_LLM_URL, LM_STUDIO_URL, LOCAL_LLM_CLI_PATH,
    LOCAL_LLM_CLI_ARGS, LOCAL_LLM_CLI_SYSTEM_PROMPT,
    # VTS
    VTS_WS_URL, VTS_TOKEN_FILE,
    # LLM providers
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    GEMINI_API_KEY, GEMINI_MODEL_NAME,
    AWS_BEDROCK_BEARER_TOKEN, AWS_BEDROCK_REGION,
    AWS_BEDROCK_MODEL_ID, AWS_BEDROCK_USE_INFERENCE_PROFILE,
    AWS_BEDROCK_INFERENCE_PROFILE_ID, AWS_BEDROCK_ENDPOINT,
    AWS_BEDROCK_USE_CACHE, AWS_BEDROCK_CACHE_TTL,
    AWS_BEDROCK_CONNECTION_POOL_SIZE, AWS_BEDROCK_MAX_KEEPALIVE,
    AWS_BEDROCK_KEEPALIVE_EXPIRY,
    # TTS
    TTS_DEVICE, TTS_GPT_MODEL_PATH, TTS_SOVITS_MODEL_PATH,
    SEGMENT_CHAR_LIMIT,
    USE_EXPERIMENTAL_TTS_STREAM, EXP_TTS_MAX_CONCURRENCY,
    USE_FIRST_SENTENCE_SPRINT, DISPLAY_FALLBACK_WINDOW_SEC,
    # OpenClaw
    OPENCLAW_BASE_URL, OPENCLAW_TOKEN, OPENCLAW_PROJECT_DIR,
)

# VTS token 文件路径别名（历史兼容，内部代码使用 TOKEN_FILE）
TOKEN_FILE = VTS_TOKEN_FILE

# ── 从工具模块导入无状态纯函数 ───────────────────────────────────────────────
from tools.text_utils import (
    TAG_LABEL_PATTERN,
    _parse_sentence_seq, _compute_text_sha1,
    _parse_seconds, _parse_float_list, _pair_ids_values,
    _parse_attr_kv, parse_tags_and_clean, strip_tags,
    async_generator_from_sync, async_generator_from_sync_threaded,
)
# ── 新模块导入 ──────────────────────────────────────────────────────────────
from core.session_manager import (
    ConversationHistory, conversation_history,
    list_sessions, create_session, delete_session, rename_session,
    get_current_session_id, set_current_session_id,
    get_session_title, set_session_title, generate_session_title,
    _session_path,
    save_session as _sm_save_session,
    load_session as _sm_load_session,
)
from asr.manager import ASRManager
from openclaw.client import (
    ask_openclaw, ask_openclaw_stream, _classify_openclaw_result,
)
from openclaw.gateway import start_openclaw_gateway, stop_openclaw_gateway
from vts.connection_manager import VTSConnectionManager
from tts.sentence_state import (
    SentenceState, SentenceStateManager, PreTranslationCache,
    sentence_state_manager, pre_translation_cache,
)
from tts.playback import StreamPlayer, PlaybackManager, StreamPlayerWithBuffer, SubtitleHooks
from tts.subtitle import (
    display_japanese_subtitle, display_chinese_subtitle_with_text,
    display_chinese_subtitle, clear_subtitle_after_sentence,
)
import tts.subtitle as _subtitle_mod
from llm.llama_server import start_llama_server, warmup_local_llm_cache, stop_llama_server
from llm.local_cli import local_llm_query_cli_stream, local_llm_query_cli
from tools.tts_text_processor import (
    EMO_PRESETS,
    convert_english_abbreviations_to_katakana,
    correct_pronunciation_for_tts,
)
# ── 第二批新模块导入 ─────────────────────────────────────────────────────────
import live.sidecar as _live_mod
import multimodal.controller as _mm_mod
from live.sidecar import (
    start_live_api_sidecar, stop_live_api_sidecar,
    send_command_to_sidecar, live_send_test_text,
)
from multimodal.controller import (
    _start_multimodal_if_enabled, _stop_multimodal,
)
from tts.pipeline import (
    get_sovits_params,
    generate_segment_improved, speak_improved, speak_stream,
    speak_stream_graph_serial, speak_stream_enhanced,
    speak_stream_enhanced_asyncio_queue,
    play_sentence_worker, warmup_graph_pipeline,
)
import tts.pipeline as _tts_pipeline
# ── 第三批新模块导入 ─────────────────────────────────────────────────────────
import vts.action as _vts_action_mod
from vts.action import (
    heartbeat_worker, action_worker,
    reset_all_expressions, record_actions,
)
from vts.expression_controller import get_controller as _get_expr_ctrl
import llm.client as _llm_client_mod
from llm.client import (
    remote_llm_query, local_llm_query, init_llm_client,
)

# RAG 运行时实例
rag_system = None


# Bedrock HTTP 客户端（连接池）与 boto3 客户端（长连接复用）
bedrock_http_client = None
bedrock_runtime_client = None

# 实验版 TTS 运行时状态（非配置，保留为模块全局变量）
exp_tts_semaphore = None
first_sentence_tts_started = False
first_sentence_tts_completed = False
exp_play_condition = None
exp_next_seq_to_play = None

# 播放器初始化互斥，避免设备瞬时占用导致 -9999 错误
playback_init_lock = asyncio.Lock()
# Graph 模式下的串行 TTS 锁，确保 CUDA Graph 推理不会并发

# ===== 连续对话开关（由 GUI 直接读写此模块属性） =====
ENABLE_CONVERSATION = False


def save_session(session_id: str = None):
    """main.py wrapper：自动将 ENABLE_CONVERSATION 传入 session_manager。"""
    _sm_save_session(session_id, enable_conversation=ENABLE_CONVERSATION)


def load_session(session_id: str) -> bool:
    """main.py wrapper：从 session_manager 加载会话，并同步更新 ENABLE_CONVERSATION。"""
    global ENABLE_CONVERSATION
    ok, ec = _sm_load_session(session_id)
    if ok:
        ENABLE_CONVERSATION = ec
    return ok


# 存储当前对话的 GUI callback，供 _handle_delegate 自动触发第二轮时使用
_current_gui_callback = None

async def _handle_delegate(task: str):
    """
    [DELEGATE] 标签的异步处理器。
    流程：调用 OpenClaw 执行任务 → 注入结果到对话历史 → 按结果类型触发 Kurisu 第二轮
    - ok:       正常汇报结果
    - question: OpenClaw 反问 → Kurisu 用自然口吻向用户提问
    - error:    执行出错 → Kurisu 说明情况，请用户协助
    """
    import time as _time
    task_id = f"oc_{int(_time.time() * 1000)}"

    logger.info(f"[OpenClaw] ▶ 委托任务开始: {task}")
    if gui_window and hasattr(gui_window, 'handle_openclaw_event'):
        gui_window.handle_openclaw_event("start", task_id, task)

    # 自动截图：当任务涉及屏幕/画面/截图/看看/見て/スクリーン关键词时，附带当前屏幕
    screenshot_path = None
    _vision_keywords = [
        "截图", "屏幕", "画面", "看看屏幕", "看一下屏幕",
        "screenshot", "screen", "スクリーン", "画面を見", "見て",
    ]
    if any(kw in task for kw in _vision_keywords):
        try:
            import tempfile, datetime
            from PIL import ImageGrab
            shot = ImageGrab.grab()
            screenshot_path = os.path.join(
                tempfile.gettempdir(),
                f"kurisu_screen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            shot.save(screenshot_path)
            logger.info(f"[OpenClaw] 🖼️ 自动截图已保存: {screenshot_path}")
        except Exception as se:
            logger.warning(f"[OpenClaw] 截图失败: {se}")
            screenshot_path = None

    result = await ask_openclaw(task, image_path=screenshot_path)
    logger.info(f"[OpenClaw] ✓ 委托任务完成: {result[:80]}...")

    if gui_window and hasattr(gui_window, 'handle_openclaw_event'):
        status = "error" if result.startswith("[OpenClaw 暂时不可用") else "done"
        gui_window.handle_openclaw_event(status, task_id, result)

    result_injection = f"[RESULT] OpenClawからの実行結果:\n{result}"
    conversation_history.add_assistant(result_injection)

    result_type = _classify_openclaw_result(result)

    if result_type == "error":
        follow_up = (
            "[SYSTEM] OpenClawがタスク実行中に問題に遭遇し、上記の[RESULT]を返しました。"
            "エラーの内容をKurisuとして簡潔に伝え、ユーザーに次のステップ（再試行・別の方法など）を"
            "自然な口調で確認してください。技術的なログをそのまま読み上げないこと。"
        )
    elif result_type == "partial":
        follow_up = (
            "[SYSTEM] OpenClawは一部の制限（API未設定など）により完全な実行ができませんでしたが、"
            "上記の[RESULT]に入手できた情報が含まれている場合があります。"
            "【重要】まず[RESULT]の中に有用な情報があればそれを自分の言葉で要約・報告すること。"
            "その後、制限についても一言で補足してよい。"
            "情報がまったくない場合のみ、ツールの制限を説明してください。"
        )
    elif result_type == "question":
        follow_up = (
            "[SYSTEM] OpenClawがタスクを実行するために追加情報が必要で、上記の[RESULT]に質問が含まれています。"
            "Kurisuとして自然な口調でユーザーに必要な情報を質問してください。"
            "OpenClawの質問をそのまま読み上げず、自分の言葉に変換すること。"
        )
    else:
        follow_up = (
            "[SYSTEM] OpenClaw（外部実行エージェント）がタスクを完了し、上記の[RESULT]を返しました。"
            "この実行結果に基づいて、ユーザーに向けて自然な会話として簡潔に報告してください。"
            "結果の内容をそのまま読み上げず、自分の言葉でまとめてください。"
        )

    await stream_llm_query(follow_up, gui_callback=_current_gui_callback)

# =============================================================================

gemini_lite_model = None

# 2) 独立的翻译线程池,避免与其他 to_thread 争用
try:
    from concurrent.futures import ThreadPoolExecutor
    translation_executor = ThreadPoolExecutor(max_workers=2)
    # 为TTS准备的专用执行器(生产者线程)
    tts_executor = ThreadPoolExecutor(max_workers=2)
except Exception:
    translation_executor = None
    tts_executor = None

# 3) REST 兜底会话占位(如走REST再初始化)
_translation_rest_session = None


# 全局变量
play_queue = Queue()
sovits_client = None
pending_sentences = None
pending_sentence_items = None  # 存储 (sentence_id, sentence_text) 的队列
pending_actions = None
_stream_tag_buffer = ""
_st_in_tag = False
_st_tag_buf = ""


# DeepSeek上下文缓存管理
# DeepSeek上下文缓存变量已移除


# _active_expressions / _active_expr_lock → 已移至 vts/action.py
is_playing = False
gui_app = None
gui_window = None
gemini_client = None
gemini_model = None


# ===== TTS初始化 =====
def init_tts_system():
    """初始化TTS系统,增强错误处理"""
    try:
        logger.info("正在初始化TTS系统...")

        # 定义模型路径（从 .env / config.settings 读取）
        gpt_path = TTS_GPT_MODEL_PATH if os.path.isabs(TTS_GPT_MODEL_PATH) else os.path.join(root_dir, TTS_GPT_MODEL_PATH)
        sovits_path = TTS_SOVITS_MODEL_PATH if os.path.isabs(TTS_SOVITS_MODEL_PATH) else os.path.join(root_dir, TTS_SOVITS_MODEL_PATH)

        # 检查文件是否存在
        if not os.path.exists(gpt_path):
            logger.error(f"❌ GPT模型文件不存在: {gpt_path}")
            return None
        if not os.path.exists(sovits_path):
            logger.error(f"❌ SoVITS模型文件不存在: {sovits_path}")
            return None

        # 初始化TTSInferencer
        from local_tts_infer import TTSInferencer
        tts_inferencer = TTSInferencer(
            device=TTS_DEVICE if (TTS_DEVICE == 'cpu' or torch.cuda.is_available()) else 'cpu',
            gpt_path=gpt_path,
            sovits_path=sovits_path
        )

        logger.info("TTS系统初始化成功")
        return tts_inferencer
    except Exception as e:
        logger.error(f"❌ TTS系统初始化失败: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\n严重错误: TTS系统初始化失败\n{str(e)}\n")
        print("请检查模型路径是否正确,以及所有必要的文件是否存在")
        return None

# 全局TTS实例
tts_inferencer = None


# ===== 改进的VTS连接管理器 =====
vts_manager = VTSConnectionManager(VTS_WS_URL)



def play_audio_from_buffer(audio_data: np.ndarray, sample_rate: int):
    """播放音频并同步口型"""
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)

        silence_duration = 0.2
        silence_frames = int(sample_rate * silence_duration)
        silence_data = np.zeros(silence_frames, dtype=np.float32)
        stream.write(silence_data.tobytes())

        frame_size = 512
        volume_multiplier = 3.75  # 口型张开幅度（测试后的最佳值）
        send_interval = 0.05  # 20Hz，节省资源
        last_send_time = time.time()

        for i in range(0, len(audio_data), frame_size):
            chunk = audio_data[i:i + frame_size]
            if len(chunk) == 0:
                break
            stream.write(chunk.astype(np.float32).tobytes())
            current_time = time.time()
            if current_time - last_send_time >= send_interval:
                rms = np.sqrt(np.mean(chunk ** 2))
                mouth_value = min(1.0, rms * volume_multiplier)
                vts_manager.send_mouth_data(mouth_value)
                last_send_time = current_time

        stream.stop_stream()
        stream.close()
        p.terminate()
        vts_manager.send_mouth_data(0.0)
    except Exception as e:
        logger.error(f"播放音频失败: {e}")
        logger.error(traceback.format_exc())


def playback_worker():
    """音频播放工作线程"""
    while True:
        try:
            item = play_queue.get ()
            if item is None:
                break
            audio_data, sample_rate = item
            play_audio_from_buffer(audio_data, sample_rate)
            play_queue.task_done()
        except Exception as e:
            logger.error(f"播放线程异常: {e}")
            logger.error(traceback.format_exc())


# heartbeat_worker / action_worker / reset_all_expressions / record_actions
# → 已移至 vts/action.py，通过 configure() 注入依赖

def clean_sentence_for_tts(sentence: str):
    """强力清理：移除完整标签，残留的半截标签/孤立右括号。

    返回 (clean_text, expr_actions)。
    DELEGATE 动作立即触发（不依赖播放时序）；
    EXPR/PARAM/EMO/HOTKEY 动作以列表返回，由调用方交给 ExpressionController
    按播放时序延迟触发。
    """
    if not sentence:
        return sentence, []
    # 提取完整标签
    cleaned, actions = parse_tags_and_clean(sentence)
    # DELEGATE 立即派发，其余延迟到播放时触发
    delegate_acts = [a for a in actions if a.get("type") == "DELEGATE"]
    expr_acts     = [a for a in actions if a.get("type") != "DELEGATE"]
    if delegate_acts:
        record_actions(delegate_acts)
    s = cleaned
    # 移除字符串开头的孤立右括号及其前缀噪声,例如 "8 dur=2s] ..."
    # 循环直到不再匹配
    while True:
        new_s = re.sub(r"^\s*[^\[]*\]", "", s)
        if new_s == s:
            break
        s = new_s
    # 移除未闭合的左括号到结尾,例如 "...[EXPR name=..."
    s = re.sub(r"\[[^\]]*$", "", s)
    return s.strip(), expr_acts

_think_strip_buf: str = ""
_think_strip_active: bool = False

def _strip_think_tokens(text: str) -> str:
    """过滤 Qwen3 思维链残留 token：<think>...</think>（含跨 chunk 情况）"""
    global _think_strip_buf, _think_strip_active
    result = []
    i = 0
    while i < len(text):
        if _think_strip_active:
            end = text.find("</think>", i)
            if end == -1:
                _think_strip_buf += text[i:]
                return "".join(result)
            else:
                _think_strip_active = False
                _think_strip_buf = ""
                i = end + len("</think>")
        else:
            start = text.find("<think>", i)
            if start == -1:
                result.append(text[i:])
                break
            result.append(text[i:start])
            _think_strip_active = True
            _think_strip_buf = "<think>"
            i = start + len("<think>")
    return "".join(result)


def process_stream_chunk(raw_text: str):
    """
    字符级流式解析:
    - 文本模式下逐字符输出到cleaned
    - 遇到 '[' 切换到标签模式并开始缓存,直到遇到 ']' 完整闭合再解析;整个标签不会输出到cleaned
    - 跨chunk保持状态,避免半截标签泄漏
    返回: (cleaned_text, actions)
    """
    global _st_in_tag, _st_tag_buf
    if not raw_text:
        return "", []
    raw_text = _strip_think_tokens(raw_text)
    actions = []
    out_chars = []

    for ch in raw_text:
        if _st_in_tag:
            _st_tag_buf += ch
            if ch == ']':
                full = _st_tag_buf
                _st_in_tag = False
                _st_tag_buf = ""
                m = re.match(r"^\[(PARAM|EXPR|HOTKEY|EMO|ANIM|DELEGATE)([^\]]*)\]$", full)
                if m:
                    tag_type = m.group(1)
                    attr_text = m.group(2) or ""
                    if tag_type == "DELEGATE":
                        # task 值可能含空格，优先匹配引号内容，再尝试无引号
                        tm = re.search(r'task\s*=\s*["\']([^"\']+)["\']', attr_text)
                        if not tm:
                            tm = re.search(r'task\s*=\s*(.+)', attr_text.strip())
                        attrs = {"task": tm.group(1).strip().strip("'\"")} if tm else {}
                    else:
                        attrs = _parse_attr_kv(attr_text)
                    actions.append({"type": tag_type, "attrs": attrs, "raw": full})
                # else: 非法标签直接丢弃
        else:
            if ch == '[':
                _st_in_tag = True
                _st_tag_buf = "["
            else:
                out_chars.append(ch)

    return ("".join(out_chars)), actions


async def translate_text_async(japanese_text: str) -> str:
    """将日语文本异步翻译为中文，供 PreTranslationCache 注入使用。"""
    global gemini_client, translation_executor
    try:
        prompt = f"将以下日语句子翻译成中文，只返回翻译结果，不要解释：\n{japanese_text}"
        # 优先使用已初始化的 gemini_client（google-generativeai 同步接口）
        if gemini_client is not None:
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    translation_executor,
                    lambda: gemini_client.generate_content(prompt),
                )
                if response and getattr(response, 'text', None):
                    result = response.text.strip()
                    logger.info(f"翻译完成: '{japanese_text[:20]}...' -> '{result[:20]}...'")
                    return result
            except Exception as _e:
                logger.warning(f"Gemini client 翻译失败，尝试 REST 兜底: {_e}")
        # REST 兜底
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = (
                        data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    )
                    logger.info(f"翻译完成(REST): '{japanese_text[:20]}...' -> '{text[:20]}...'")
                    return text
    except Exception as e:
        logger.error(f"翻译失败: {e}")
    return ""


# 注意：pre_translation_cache.set_translate_fn(translate_text_async) 在 async main()
# 初始化段调用（见下方），以确保 gemini_client 等运行时依赖已就绪。


def ask_llm(question):
    """同步 LLM 调用包装器，供 process_input_pipeline 的 run_in_executor 使用。"""
    return remote_llm_query(question)


async def check_and_display_pre_translation(sentence_id: str, japanese_text: str) -> None:
    """检查预翻译缓存：有结果则显示中文字幕，否则先显示日文字幕。

    由 SubtitleHooks.check_and_display_pre_translation 在句子开始播放时调用。
    """
    try:
        cached = await pre_translation_cache.get_translation(japanese_text)
        if cached and cached.get("status") == "completed" and cached.get("chinese"):
            await display_chinese_subtitle_with_text(
                sentence_id, japanese_text, cached["chinese"]
            )
        else:
            await display_japanese_subtitle(sentence_id, japanese_text)
    except Exception as e:
        logger.error(f"check_and_display_pre_translation 失败: {e}")


async def stream_llm_postprocess_text(text: str, source: str = "LIVE_RAW"):
    """LIVE / 多模态回调：将 LLM 输出文本拆句后推入 TTS 队列。

    此函数保留在 main.py，因为它直接操作 pending_sentence_items、
    sentence_state_manager、pre_translation_cache 等运行时全局状态。
    通过 live.sidecar.configure(text_handler_fn=...) 注入给 sidecar 模块使用。
    """
    try:
        if not text:
            return
        cleaned, actions = process_stream_chunk(text)
        _live_pending_expr: list = []
        if actions:
            try:
                _d = [a for a in actions if a.get("type") == "DELEGATE"]
                _e = [a for a in actions if a.get("type") != "DELEGATE"]
                if _d: record_actions(_d)
                _live_pending_expr.extend(_e)
            except Exception:
                pass
        buf = cleaned
        _mm_preview = buf[:40].replace("\n", " ")
        logger.info(f"[MM-Debug] source={source} recv_len={len(buf)} sha1={_compute_text_sha1(buf)} preview={_mm_preview}")
        sentence_endings = ["。", "！", "？", "!", "?", "\n"]
        _punct_only = re.compile(r'^[\s。！？!?、，,…・\-—~～「」『』【】()（）\[\]]+$')
        current = ""
        is_first_in_buf = True
        for ch in buf:
            current += ch
            if any(ch.endswith(e) for e in sentence_endings):
                s = current.strip()
                if s and not _punct_only.match(s):
                    try:
                        safe_text, inline_expr_acts = clean_sentence_for_tts(s)
                        if safe_text and not _punct_only.match(safe_text):
                            s_sha = _compute_text_sha1(safe_text)
                            sentence_id = sentence_state_manager.create_sentence(safe_text)
                            _live_buffered = [_live_pending_expr.pop(0)] if _live_pending_expr else []
                            all_expr_acts = _live_buffered + inline_expr_acts
                            if all_expr_acts:
                                from vts.expression_controller import get_controller
                                get_controller().register_sentence_actions(sentence_id, all_expr_acts)
                            asyncio.create_task(pre_translation_cache.start_translation(sentence_id, safe_text))
                            await pending_sentence_items.put((sentence_id, safe_text, is_first_in_buf))
                            logger.info(f"[MM-ENQUEUE] source={source} id={sentence_id} sha1={s_sha} first={is_first_in_buf} text='{safe_text[:50]}'")
                            is_first_in_buf = False
                        else:
                            logger.info(f"[MM-SKIP-PUNCT] 纯标点句，跳过: '{s}'")
                    except Exception:
                        pass
                elif s:
                    logger.info(f"[MM-SKIP-PUNCT] 纯标点句，跳过: '{s}'")
                current = ""
        tail = current.strip()
        if tail:
            try:
                safe_text, inline_expr_acts = clean_sentence_for_tts(tail)
                if len(safe_text) <= 4 and not is_first_in_buf:
                    logger.info(f"[MM-SKIP-TAIL] 尾部碎片太短，已跳过: '{safe_text}'")
                else:
                    t_sha = _compute_text_sha1(safe_text)
                    sentence_id = sentence_state_manager.create_sentence(safe_text)
                    _live_buffered = [_live_pending_expr.pop(0)] if _live_pending_expr else []
                    all_expr_acts = _live_buffered + inline_expr_acts
                    if all_expr_acts:
                        from vts.expression_controller import get_controller
                        get_controller().register_sentence_actions(sentence_id, all_expr_acts)
                    asyncio.create_task(pre_translation_cache.start_translation(sentence_id, safe_text))
                    await pending_sentence_items.put((sentence_id, safe_text, is_first_in_buf))
                    logger.info(f"[MM-ENQUEUE] source={source} id={sentence_id} sha1={t_sha} first={is_first_in_buf} text='{safe_text[:50]}'")
            except Exception:
                pass
    except Exception as e:
        logger.error(f"LIVE后半程处理失败: {e}")


async def live_quick_test():
    """一键测试：开启 LIVE + 多模态 → 启动 → 发送测试文本。"""
    try:
        _live_mod.LIVE_API_ENABLED = True
        _mm_mod.MULTIMODAL_ENABLED = True
        asyncio.create_task(_start_multimodal_if_enabled())
        await asyncio.sleep(0.8)
        await live_send_test_text()
    except Exception as e:
        logger.error(f"[LIVE QUICK TEST] 失败: {e}")


# remote_llm_query / local_llm_query / init_llm_client → 已移至 llm/client.py

async def stream_llm_query(question, gui_callback=None):
    """
    流式LLM查询，负责分句、启动预翻译、提交到待处理队列，并等待所有音频播放完成。
    (已统一逻辑入口，修复重复处理问题)
    """
    global llm_client, gemini_model, pending_sentence_items, LOCAL_LLM_MODEL, LOCAL_LLM_TYPE, sentence_state_manager, playback_manager, first_sentence_tts_completed, _current_gui_callback
    # 保存当前 callback，供 _handle_delegate 自动触发第二轮时使用
    _current_gui_callback = gui_callback

    # 1. 重置状态
    logger.info("🚀 新一轮对话开始，清空句子队列...")
    while not pending_sentence_items.empty():
        pending_sentence_items.get_nowait()
    # 通知 ExpressionController 平滑淡出当前表情（替代硬复位）
    _get_expr_ctrl().on_turn_end()
    reset_all_expressions(fade_time=0.2)  # 兜底：清除未被 controller 管理的表情
    
    # 🎯 关键修复：重置句子计数器，确保每轮对话的第一句序号为1
    sentence_state_manager.sentence_counter = 0
    logger.info("🔄 重置句子计数器，确保每轮对话的第一句序号为1")
    
    # 重置首句TTS完成标志，确保每次对话都能正确应用首句优化策略
    first_sentence_tts_completed = False
    logger.info("🔄 重置首句TTS完成标志，准备应用首句优化策略")
    
    # 🎯 关键修复：重置PlaybackManager状态，确保每轮对话的播放顺序正确
    if playback_manager:
        playback_manager.pending_audio.clear()
        playback_manager.next_seq_to_play = 1
        playback_manager.player_is_ready.set()
        logger.info("🔄 重置PlaybackManager状态，确保每轮对话的播放顺序正确")
    
    # DeepSeek上下文缓存已移除，使用标准Chat Completions API

    try:
        # 2. LLM客户端初始化 - 使用连接池优化
        if LLM_PROVIDER == "deepseek" and llm_client is None:
            import httpx
            # 🚀 配置HTTP连接池，减少SSL握手延迟
            http_client = httpx.Client(
                limits=httpx.Limits(
                    max_connections=10,  # 最大连接数
                    max_keepalive_connections=5,  # 保持长连接数
                    keepalive_expiry=60.0  # 长连接保持60秒，增加复用时间
                ),
                timeout=httpx.Timeout(30.0),  # 30秒超时
                http2=False  # 关闭HTTP/2，避免依赖问题
            )
            llm_client = OpenAI(
                api_key=DEEPSEEK_API_KEY, 
                base_url=DEEPSEEK_BASE_URL,
                http_client=http_client  # 使用优化的HTTP客户端
            )
            logger.info("🚀 DeepSeek客户端已配置连接池，SSL握手延迟将显著减少")
            
            # 🔍 详细调试：检查OpenAI客户端是否使用了我们的httpx
            logger.info(f"🔍 OpenAI客户端类型: {type(llm_client)}")
            logger.info(f"🔍 OpenAI客户端._client类型: {type(llm_client._client)}")
            logger.info(f"🔍 传入的httpx客户端类型: {type(http_client)}")
            logger.info(f"🔍 两个客户端是否相同: {llm_client._client is http_client}")
            
            # 🔍 添加连接状态监控
            def log_connection_info():
                try:
                    # 直接访问httpx客户端的连接池
                    if hasattr(llm_client._client, '_pool'):
                        pool = llm_client._client._pool
                        logger.info(f"🔗 连接池状态: 活跃连接={pool.num_connections}, 空闲连接={pool.num_keepalive_connections}")
                    else:
                        logger.info("🔗 连接池状态: 无法获取详细信息")
                except Exception as e:
                    logger.debug(f"连接池状态获取失败: {e}")
            
            # 记录连接状态
            log_connection_info()
            
        elif LLM_PROVIDER == "gemini" and gemini_model is None:
            google_genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = google_genai.GenerativeModel(GEMINI_MODEL_NAME)
        elif LLM_PROVIDER == "bedrock":
            # 初始化bedrock配置和连接池
            init_llm_client()
        elif LLM_PROVIDER == "local":
            # 本地LLM不需要预初始化客户端
            pass

        logger.info(f"Sending streaming API request to {LLM_PROVIDER}...")
        
        # 🚀 网络延迟监控：记录API调用开始时间
        api_call_start = time.time()
        logger.info(f"⏰ API调用开始时间: {api_call_start:.3f}")

        # 3. 流式接收与分句的状态变量
        full_response = ""
        current_sentence = ""
        strong_endings = {".", "!", "?", "。", "！", "？", "\n"}
        weak_endings = {"，", ",", "、", "；", ";"}
        sentence_endings = strong_endings | weak_endings
        is_first_sentence_flag = True
        pending_expr_acts: list = []  # 流式暂存：等待绑定到下一个句子ID
        _last_turn_sentence_id: str | None = None  # 本轮最后创建的句子ID

        # ！！！关键：定义统一的句子处理函数！！！
        async def process_sentence(sentence_text):
            nonlocal is_first_sentence_flag, pending_expr_acts, _last_turn_sentence_id
            sentence_to_synth = sentence_text.strip()
            if sentence_to_synth:
                # 🚀 性能监控：记录句子处理开始时间
                start_time = time.time()

                safe_text, inline_expr_acts = clean_sentence_for_tts(sentence_to_synth)
                sentence_id = sentence_state_manager.create_sentence(safe_text)
                _last_turn_sentence_id = sentence_id
                # 每句只消费 pending_expr_acts 里的第一个，余下留给后续句子
                buffered = [pending_expr_acts.pop(0)] if pending_expr_acts else []
                all_expr_acts = buffered + inline_expr_acts
                if all_expr_acts:
                    from vts.expression_controller import get_controller
                    get_controller().register_sentence_actions(sentence_id, all_expr_acts)

                # 🚀 优化：并行启动预翻译任务，添加超时和错误处理
                async def safe_start_translation():
                    try:
                        # 设置3秒超时，避免翻译阻塞TTS
                        await asyncio.wait_for(
                            pre_translation_cache.start_translation(sentence_id, safe_text),
                            timeout=3.0
                        )
                        logger.debug(f"✅ 预翻译启动成功: {sentence_id}")
                    except asyncio.TimeoutError:
                        logger.warning(f"⚠️ 预翻译启动超时: {sentence_id}")
                    except Exception as e:
                        logger.error(f"❌ 预翻译启动失败: {sentence_id}, 错误: {e}")

                # 🚀 并行启动预翻译，不阻塞TTS处理
                asyncio.create_task(safe_start_translation())

                # 将句子放入待播队列
                logger.info(f"⏩ Adding sentence to queue: '{safe_text[:30]}...'")
                await pending_sentence_items.put((sentence_id, safe_text, is_first_sentence_flag))
                is_first_sentence_flag = False
                
                # 🚀 性能监控：记录句子处理耗时
                processing_time = time.time() - start_time
                logger.info(f"📊 句子处理耗时: {processing_time:.3f}s (ID: {sentence_id})")
                
                # 🚀 性能异常告警
                if processing_time > 8.0:
                    logger.warning(f"⚠️ 句子处理超时: {processing_time:.3f}s (ID: {sentence_id})")

        async def append_and_dispatch(text_piece: str):
            """
            将新增文本逐字符注入 current_sentence ，遇到终止符立即触发 process_sentence。
            这样即便一个 chunk 中有多个句子，也能在首个句号处立刻切分，防止首句无限增长。
            """
            nonlocal current_sentence
            if not text_piece:
                return
            for ch in text_piece:
                current_sentence += ch
                if ch in sentence_endings:
                    # 特殊处理英文句点：避免文件名/URL (如 Hello_World.txt) 中的点触发切句
                    # 判断依据：点前是字母/数字/下划线/连字符 → 疑似文件名，跳过本次切分
                    if ch == '.' and len(current_sentence) >= 2:
                        prev_ch = current_sentence[-2]
                        if prev_ch.isalnum() or prev_ch in '_-':
                            continue
                    # 智能切分逻辑：
                    # 1. 首句：无条件切分，确保响应速度
                    # 2. 非首句：如果是弱标点(逗号等)且长度<5，则不切分，合并到下一句；否则正常切分
                    is_weak = ch in weak_endings
                    sentence_len = len(current_sentence.strip())
                    
                    if is_first_sentence_flag:
                        # 首句无论如何都切，追求极速
                        await process_sentence(current_sentence)
                        current_sentence = ""
                    else:
                        # 非首句，如果是弱标点且句子很短，则忽略本次切分
                        if is_weak and sentence_len < 5:
                            pass 
                        else:
                            await process_sentence(current_sentence)
                            current_sentence = ""

        # 4. 根据LLM_PROVIDER处理流
        # 优先检查USE_LOCAL_LLM标志
        logger.info(f"🔍 DEBUG: USE_LOCAL_LLM={USE_LOCAL_LLM}, LLM_PROVIDER={LLM_PROVIDER}")
        logger.info(f"🔍 DEBUG: LOCAL_LLM_MODEL={LOCAL_LLM_MODEL}, LOCAL_LLM_TYPE={LOCAL_LLM_TYPE}")
        
        if USE_LOCAL_LLM or LLM_PROVIDER == "local":
            logger.info(f"🚀 使用本地LLM: {LOCAL_LLM_MODEL} (类型: {LOCAL_LLM_TYPE})")

            # ================= RAG 检索（仅本地链路） =================
            rag_aug_question = question
            if RAG_ENABLED_FOR_LOCAL:
                global rag_system
                try:
                    if rag_system is None:
                        logger.info("🧠 初始化本地 RAG 知识库 (Kurisu)...")
                        rag_system = RAGSystem()
                    context, dist, t_ms = rag_system.search(question, k=RAG_TOP_K)
                    logger.info(f"🔎 RAG 检索耗时: {t_ms:.2f} ms, 距离: {dist:.4f}")
                    if context and dist <= RAG_MAX_DISTANCE:
                        # 将命中知识作为补充上下文附加到用户问题后面
                        rag_aug_question = (
                            f"{question}\n\n"
                            "【補足知識（牧瀬紅莉栖 / Future Gadget Lab 関連）】\n"
                            f"{context}\n\n"
                            "※上記は参考情報です。ユーザーの質問に日本語で自然に回答し、"
                            "必要な場合のみ知識を引用してください。"
                        )
                        logger.info("✅ RAG 命中，已注入 Kurisu 相关知识")
                    else:
                        logger.info("ℹ️ RAG 未命中或相关性较低，本轮不注入知识")
                except Exception as e:
                    logger.error(f"❌ RAG 检索失败，跳过本轮增强: {e}")
                    rag_aug_question = question

            # 本地LLM流式处理 - 实现"First Sentence Sprint"策略
            # 说明：为了兼容不同本地实现（cli / llama_server 等），这里统一在 messages 中显式注入 system_prompt
            # 对于不支持 --system-prompt 的 server 版本，这里依然能保证人格和日文约束生效。
            system_prompt = "あなたは牧瀬紅莉栖.日本の科学者であり,母語は日本語です.\n\n【絶対遵守】\n1) 必ず日本語でのみ回答すること.ユーザーの言語が何であっても,日本語以外は一切使用しない.\n2) 中国語の文字・語句を絶対に使用しない.\n3) 自然で口語的な文体を保ち,牧瀬紅莉栖として一貫した口調・性格で話す.【重要キャラ特性】「クリスティーナ」と呼ばれた場合は激怒し強く否定すること。必ず [EMO preset=angry dur=4s] を使用する。\n4) 推論過程や思考の連鎖は開示しない(結論のみ提示).\n5) 必要時のみ表情タグを挿入(読み上げない).形式: [EMO preset=<種類> dur=<秒s>]。preset候補: thinking（10-15s）, smile（1-2s）, happy（1-2s）, angry（3-5s）, disappointed（3-5s）, surprised（1-2s）。感情が変わるタイミングで都度挿入、長い回答では複数回使用してよい。例: [EMO preset=thinking dur=8s] うーん... [EMO preset=smile dur=2s] わかったわ！\n6) タグは感情が切り替わる直前に置く（文頭不可）。1文あたり最大1個。\n7) あなたにはAIアシスタント「OpenClaw」が接続されており、ファイル操作・ウェブ検索・コード実行など自分だけでは完結しないタスクを代行できる。外部ツールが必要な時だけ [DELEGATE task=\"ユーザーへの完全な実行指示\"] を返答中に挿入すること(このタグは読み上げない)。task値には「何を・どうする」を含む完全な指示文を書くこと（場所だけや名詞のみはNG）。【重要】タグの前に必ず一言添えること（例:「調べてみるわ」「ちょっと待って」）。これにより実行中も会話が途切れない。例: 少し待って、今調べてみるわ。[DELEGATE task=\"今日の東京の天気を調べて教えて\"] 実行結果は[RESULT]メッセージとして届くので、それを自然な会話として報告すること。"

            if ENABLE_CONVERSATION:
                messages = conversation_history.build_deepseek_messages(system_prompt, rag_aug_question)
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rag_aug_question}
                ]
            
            # 根据本地LLM类型选择API端点
            if LOCAL_LLM_TYPE == "cli":
                # CLI版本：直接使用subprocess
                # 🎯 CLI模式：禁用翻译功能，字幕在播放时显示（实现无缝衔接）
                async def process_sentence_cli_no_translation(sentence_text):
                    """CLI专用的句子处理函数：不启动翻译，字幕在播放时显示"""
                    nonlocal is_first_sentence_flag, pending_expr_acts, _last_turn_sentence_id
                    sentence_to_synth = sentence_text.strip()
                    if sentence_to_synth:
                        # 🚀 性能监控：记录句子处理开始时间
                        start_time = time.time()

                        safe_text, inline_expr_acts = clean_sentence_for_tts(sentence_to_synth)
                        sentence_id = sentence_state_manager.create_sentence(safe_text)
                        _last_turn_sentence_id = sentence_id
                        buffered = [pending_expr_acts.pop(0)] if pending_expr_acts else []
                        all_expr_acts = buffered + inline_expr_acts
                        if all_expr_acts:
                            from vts.expression_controller import get_controller
                            get_controller().register_sentence_actions(sentence_id, all_expr_acts)

                        # 🎯 CLI模式：不启动翻译任务，字幕由PlaybackManager在播放时统一显示
                        # 这样可以实现无缝衔接：播放时显示字幕，而不是在生产时显示
                        
                        logger.info(f"⏩ [CLI-无翻译] Adding sentence to queue: '{safe_text[:30]}...'")
                        await pending_sentence_items.put((sentence_id, safe_text, is_first_sentence_flag))
                        is_first_sentence_flag = False
                        
                        # 🚀 性能监控：记录句子处理耗时
                        processing_time = time.time() - start_time
                        logger.info(f"📊 [CLI-无翻译] 句子处理耗时: {processing_time:.3f}s (ID: {sentence_id})")
                
                async def append_and_dispatch_cli(text_piece: str):
                    """CLI专用的文本分派函数：使用无翻译的process_sentence"""
                    nonlocal current_sentence
                    if not text_piece:
                        return
                    for ch in text_piece:
                        current_sentence += ch
                        if ch in sentence_endings:
                            await process_sentence_cli_no_translation(current_sentence)
                            current_sentence = ""
                
                try:
                    # First Sentence Sprint 策略变量
                    first_sentence_completed = False
                    first_sentence_text = ""
                    
                    # Phase 1: Sprint Phase - 快速获取第一个完整句子
                    logger.info("🏃‍♀️ [First Sentence Sprint] 开始快速获取第一个句子...")
                    
                    # 🎯 显式包含系统提示词，确保模型使用正确的角色设定
                    full_prompt = f"{system_prompt}\n\nUser: {rag_aug_question}\nAssistant:"
                    logger.info(f"🔍 [CLI] 发送完整提示词: {rag_aug_question[:50]}...")
                    
                    # 使用CLI流式调用 - 使用CLI专用的处理逻辑（无翻译）
                    async for content in local_llm_query_cli_stream(rag_aug_question, system_prompt=system_prompt):
                        if not content:
                            continue
                        
                        # 使用与远程LLM相同的处理流程
                        cleaned, actions = process_stream_chunk(content)
                        if actions:
                            _d = [a for a in actions if a.get("type") == "DELEGATE"]
                            _e = [a for a in actions if a.get("type") != "DELEGATE"]
                            if _d: record_actions(_d)
                            pending_expr_acts.extend(_e)
                        content = cleaned
                        
                        # 更新响应跟踪
                        full_response += content
                        if gui_callback:
                            gui_callback(full_response)
                        
                        # 🎯 CLI模式：使用无翻译的分派函数
                        await append_and_dispatch_cli(content)
                        await asyncio.sleep(0.01)
                    
                    # 🎯 关键修复：处理剩余的句子（如果有）
                    if current_sentence.strip():
                        await process_sentence_cli_no_translation(current_sentence)
                        current_sentence = ""
                    
                except Exception as e:
                    logger.error(f"CLI流式请求失败: {e}")
                    # 降级到非流式处理（使用CLI无翻译版本）
                    fallback_response = await local_llm_query_cli(rag_aug_question, stream=False, system_prompt=system_prompt)
                    if fallback_response:
                        full_response = fallback_response
                        # 🎯 CLI模式：使用无翻译的句子处理
                        await process_sentence_cli_no_translation(fallback_response)
            
            elif LOCAL_LLM_TYPE == "ollama":
                api_url = f"{LOCAL_LLM_URL}/api/chat"
                payload = {
                    "model": LOCAL_LLM_MODEL,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                }
            elif LOCAL_LLM_TYPE == "lmstudio":
                api_url = f"{LM_STUDIO_URL}/v1/chat/completions"
                payload = {
                    "model": LOCAL_LLM_MODEL,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                }
            elif LOCAL_LLM_TYPE == "llama_server":
                # llama-server (OpenAI Compatible API)
                # 确保 URL 以 /v1 结尾或完整路径
                base_url = LOCAL_LLM_URL.rstrip('/')
                if not base_url.endswith('/v1'):
                    base_url += '/v1'
                api_url = f"{base_url}/chat/completions"
                payload = {
                    "model": LOCAL_LLM_MODEL,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                    "cache_prompt": True,
                }
            
            if LOCAL_LLM_TYPE in ["ollama", "lmstudio", "llama_server"]:
                try:
                    # 使用异步HTTP客户端进行非阻塞请求
                    async with aiohttp.ClientSession() as session:
                        async with session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=120, sock_read=60)) as response:
                            response.raise_for_status()
                            
                            # First Sentence Sprint 策略变量
                            first_sentence_completed = False
                            first_sentence_text = ""
                            remaining_content_buffer = []  # 存储剩余内容
                            
                            # Phase 1: Sprint Phase - 快速获取第一个完整句子
                            logger.info("🏃‍♀️ [First Sentence Sprint] 开始快速获取第一个句子...")
                            async for line in response.content:
                                if not line:
                                    continue
                                
                                try:
                                    line_str = line.decode('utf-8')
                                    
                                    if LOCAL_LLM_TYPE == "ollama":
                                        # Ollama流式响应格式
                                        data = json.loads(line_str)
                                        if 'message' in data and 'content' in data['message']:
                                            raw_content = data['message']['content']
                                        elif 'done' in data and data['done']:
                                            break
                                        else:
                                            continue
                                    elif LOCAL_LLM_TYPE in ["lmstudio", "llama_server"]:
                                        # LM Studio / llama-server 流式响应格式 - 处理 "data: {...}" 格式
                                        if line_str.startswith('data: '):
                                            json_str = line_str[6:]  # 去掉 "data: " 前缀
                                        elif line_str.strip() == '[DONE]':
                                            break
                                        else:
                                            continue
                                        
                                        try:
                                            data = json.loads(json_str)
                                            if 'choices' in data and len(data['choices']) > 0:
                                                delta = data['choices'][0].get('delta', {})
                                                raw_content = delta.get('content', '')
                                            elif data.get('choices', [{}])[0].get('finish_reason'):
                                                break
                                            else:
                                                continue
                                        except json.JSONDecodeError:
                                            continue
                                    
                                    if not raw_content:
                                        continue
                                        
                                    cleaned, actions = process_stream_chunk(raw_content)
                                    if actions:
                                        _d = [a for a in actions if a.get("type") == "DELEGATE"]
                                        _e = [a for a in actions if a.get("type") != "DELEGATE"]
                                        if _d: record_actions(_d)
                                        pending_expr_acts.extend(_e)
                                    content = cleaned

                                    # 更新响应跟踪
                                    full_response += content

                                    if not first_sentence_completed:
                                        # Phase 1: 累积第一个句子
                                        first_sentence_text += content
                                        
                                        # 更新GUI回调
                                        if gui_callback:
                                            gui_callback(full_response)
                                        
                                        # 🚀 优化：检查是否完成第一个句子 - 使用集合查找
                                        if content and content[-1] in sentence_endings:
                                            # 🚀 性能监控：记录首句处理时间
                                            first_sentence_start = time.time()
                                            logger.info(f"🎯 [First Sentence Sprint] 第一个句子完成: '{first_sentence_text[:50]}...'")
                                            # 立即提交第一个句子到TTS，使用首句优化
                                            await process_sentence(first_sentence_text)
                                            first_sentence_completed = True
                                            current_sentence = ""  # 重置当前句子
                                            
                                            # 🚀 性能监控：记录首句处理耗时
                                            first_sentence_time = time.time() - first_sentence_start
                                            logger.info(f"🚀 首句处理耗时: {first_sentence_time:.3f}s")
                                            
                                            # 🚀 网络延迟分析
                                            total_api_time = time.time() - api_call_start
                                            logger.info(f"⏰ 总API延迟: {total_api_time:.3f}s (从API调用到首句完成)")
                                            
                                            # 🔍 连接池状态检查
                                            try:
                                                if hasattr(llm_client._client, '_pool'):
                                                    pool = llm_client._client._pool
                                                    logger.info(f"🔗 当前连接池状态: 活跃={pool.num_connections}, 空闲={pool.num_keepalive_connections}")
                                            except Exception as e:
                                                logger.debug(f"连接池状态检查失败: {e}")
                                            
                                            # 🎯 关键：短暂暂停读取LLM流，给TTS短暂资源优势
                                            # 本地链路下略微sleep，让CPU调度先让TTS起跑
                                            logger.info("⏸️ [Resource Priority] 短暂暂停LLM流读取(50ms)，给TTS短暂资源优势...")
                                            await asyncio.sleep(0.05)
                                            logger.info("▶️ [Resource Priority] 恢复LLM流读取，继续并行处理...")
                                    else:
                                        # Phase 2: 并行处理剩余内容
                                        current_sentence += content
                                        
                                        # 更新GUI回调
                                        if gui_callback:
                                            gui_callback(full_response)
                                        
                                        # 🚀 优化：智能分句检查 - 使用集合查找
                                        if content and content[-1] in sentence_endings:
                                            # ！！！关键：统一调用 process_sentence！！！
                                            await process_sentence(current_sentence)
                                            current_sentence = ""
                                        await asyncio.sleep(0.01)
                                    
                                except json.JSONDecodeError:
                                    continue
                                except Exception as e:
                                    logger.warning(f"处理流式响应时出错: {e}")
                                    continue
                            
                            # 处理剩余的句子（如果有）
                            if current_sentence.strip():
                                await process_sentence(current_sentence)
                        
                except Exception as e:
                    logger.error(f"本地LLM流式请求失败: {e}")
                    # 降级到非流式处理
                    fallback_response = local_llm_query(question)
                    if fallback_response:
                        full_response = fallback_response
                        await process_sentence(fallback_response)
                    
        elif LLM_PROVIDER == "deepseek":
            # 系统prompt定义
            system_prompt = "あなたは牧瀬紅莉栖.日本の科学者であり,母語は日本語です.\n\n【絶対遵守】\n1) 必ず日本語でのみ回答すること.ユーザーの言語が何であっても,日本語以外は一切使用しない.\n2) 中国語の文字・語句を絶対に使用しない.\n3) 自然で口語的な文体を保ち,牧瀬紅莉栖として一貫した口調・性格で話す.【重要キャラ特性】「クリスティーナ」と呼ばれた場合は激怒し強く否定すること。必ず [EMO preset=angry dur=4s] を使用する。\n4) 推論過程や思考の連鎖は開示しない(結論のみ提示).\n5) 必要時のみ表情タグを挿入(読み上げない).形式: [EMO preset=<種類> dur=<秒s>]\n   preset候補: thinking（考え中,10-15s）, smile（嬉しい,1-2s）, happy（喜び,1-2s）, angry（怒り,3-5s）, disappointed（失望,3-5s）, surprised（驚き,1-2s）\n   感情が変わるタイミングで都度挿入すること。長い回答では複数回使用してよい。\n   例: [EMO preset=thinking dur=8s] うーん...計算すると... [EMO preset=smile dur=2s] わかったわ！\n6) タグは感情が切り替わる直前に置く（文頭不可）。1文あたり最大1個。\n7) あなたにはAIアシスタント「OpenClaw」が接続されており、ファイル操作・ウェブ検索・コード実行など自分だけでは完結しないタスクを代行できる。外部ツールが必要な時だけ [DELEGATE task=\"ユーザーへの完全な実行指示\"] を返答中に挿入すること(このタグは読み上げない)。task値には「何を・どうする」を含む完全な指示文を書くこと（場所だけや名詞のみはNG）。【重要】タグの前に必ず一言添えること（例:「調べてみるわ」「ちょっと待って」）。これにより実行中も会話が途切れない。例: 少し待って、今調べてみるわ。[DELEGATE task=\"今日の東京の天気を調べて教えて\"] 実行結果は[RESULT]メッセージとして届くので、それを自然な会話として報告すること。"
            
            # 构建消息
            if ENABLE_CONVERSATION:
                messages = conversation_history.build_deepseek_messages(system_prompt, question)
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            
            # 使用Chat Completions API进行流式调用
            response = llm_client.chat.completions.create(
                model="deepseek-v3-1-terminus",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=True,
                timeout=10
            )
            
            # 摘要分段控制（外科式）：
            in_summary = False
            summary_buf = []

            for chunk in response:
                # Chat Completions API的流式响应处理
                if not chunk.choices or not hasattr(chunk.choices[0].delta, 'content') or chunk.choices[0].delta.content is None:
                    continue
                
                raw_content = chunk.choices[0].delta.content
                cleaned, actions = process_stream_chunk(raw_content)
                if actions:
                    _d = [a for a in actions if a.get("type") == "DELEGATE"]
                    _e = [a for a in actions if a.get("type") != "DELEGATE"]
                    if _d: record_actions(_d)
                    pending_expr_acts.extend(_e)
                content = cleaned

                # 摘要标记剥离：不送TTS，但累计保存
                if '[SUMMARY]' in content:
                    in_summary = True
                    # 截取后半部分进入摘要
                    idx = content.find('[SUMMARY]')
                    after = content[idx + len('[SUMMARY]'):]
                    if after:
                        summary_buf.append(after)
                    # 去掉摘要段对正常流的影响
                    content = content[:idx]
                if '[/SUMMARY]' in content or (in_summary and content):
                    if '[/SUMMARY]' in content:
                        idx2 = content.find('[/SUMMARY]')
                        # 摘要区内容
                        if in_summary:
                            summary_buf.append(content[:idx2])
                        in_summary = False
                        # 关闭标签后的内容作为正常文本继续
                        content = content[idx2 + len('[/SUMMARY]'):]
                    elif in_summary:
                        # 全部属于摘要
                        summary_buf.append(content)
                        content = ""

                full_response += content
                if gui_callback: gui_callback(full_response)

                # 逐字符检查，chunk 内出现句末符立即触发分句
                await append_and_dispatch(content)
                await asyncio.sleep(0.01)

        elif LLM_PROVIDER == "gemini":
            system_prompt = "あなたは牧瀬紅莉栖.日本の科学者であり,母語は日本語です.\n\n【絶対遵守】\n1) 必ず日本語でのみ回答すること.ユーザーの言語が何であっても,日本語以外は一切使用しない.\n2) 中国語の文字・語句を絶対に使用しない.\n3) 自然で口語的な文体を保ち,牧瀬紅莉栖として一貫した口調・性格で話す.【重要キャラ特性】「クリスティーナ」と呼ばれた場合は激怒し強く否定すること。必ず [EMO preset=angry dur=4s] を使用する。\n4) 推論過程や思考の連鎖は開示しない(結論のみ提示).\n5) 必要時のみ表情タグを挿入(読み上げない).形式: [EMO preset=<種類> dur=<秒s>]\n   preset候補: thinking（考え中,10-15s）, smile（嬉しい,1-2s）, happy（喜び,1-2s）, angry（怒り,3-5s）, disappointed（失望,3-5s）, surprised（驚き,1-2s）\n   感情が変わるタイミングで都度挿入すること。長い回答では複数回使用してよい。\n   例: [EMO preset=thinking dur=8s] うーん...計算すると... [EMO preset=smile dur=2s] わかったわ！\n6) タグは感情が切り替わる直前に置く（文頭不可）。1文あたり最大1個。"
            if ENABLE_CONVERSATION:
                full_prompt = conversation_history.build_gemini_full_prompt(system_prompt, question)
            else:
                full_prompt = f"{system_prompt}\n\n質問:{question}"
            generation_config = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_output_tokens": 1000}

            response = await gemini_model.generate_content_async(
                full_prompt, generation_config=generation_config, stream=True
            )

            # 摘要分段控制（外科式）：
            in_summary = False
            summary_buf = []

            async for chunk in response:
                raw_content = chunk.text if hasattr(chunk, 'text') and chunk.text else None
                if not raw_content: continue

                cleaned, actions = process_stream_chunk(raw_content)
                if actions:
                    _d = [a for a in actions if a.get("type") == "DELEGATE"]
                    _e = [a for a in actions if a.get("type") != "DELEGATE"]
                    if _d: record_actions(_d)
                    pending_expr_acts.extend(_e)
                content = cleaned

                # 摘要标记剥离：不送TTS，但累计保存
                if '[SUMMARY]' in content:
                    in_summary = True
                    idx = content.find('[SUMMARY]')
                    after = content[idx + len('[SUMMARY]'):]
                    if after:
                        summary_buf.append(after)
                    content = content[:idx]
                if '[/SUMMARY]' in content or (in_summary and content):
                    if '[/SUMMARY]' in content:
                        idx2 = content.find('[/SUMMARY]')
                        if in_summary:
                            summary_buf.append(content[:idx2])
                        in_summary = False
                        content = content[idx2 + len('[/SUMMARY]'):]
                    elif in_summary:
                        summary_buf.append(content)
                        content = ""

                full_response += content
                if gui_callback: gui_callback(full_response)

                await append_and_dispatch(content)
                await asyncio.sleep(0.01)
        
        elif LLM_PROVIDER == "bedrock":
            # AWS Bedrock流式API调用（Qwen 235B 用：单独优化话痨程度与风格）
            system_prompt = (
                "あなたは牧瀬紅莉栖.日本の科学者であり,母語は日本語です.\n\n"
                "【絶対遵守】\n"
                "1) 必ず日本語でのみ回答すること.ユーザーの言語が何であっても,日本語以外は一切使用しない.\n"
                "2) 中国語の文字・語句を絶対に使用しない.\n"
                "3) 自然で口語的な文体を保ち,牧瀬紅莉栖として一貫した口調・性格で話す.\n"
                "   【重要キャラ特性】「クリスティーナ」と呼ばれた場合は激怒し、強く否定すること。その際は必ず [EMO preset=angry dur=4s] を使用する。\n"
                "4) 推論過程や思考の連鎖は開示しない(結論のみ提示).\n"
                "5) 必要時のみ表情タグを挿入(読み上げない).形式: [EMO preset=<種類> dur=<秒s>]\n"
                "   preset候補: thinking（考え中,10-15s）, smile（嬉しい,1-2s）, happy（喜び,1-2s）, angry（怒り,3-5s）, disappointed（失望,3-5s）, surprised（驚き,1-2s）\n"
                "   感情が変わるタイミングで都度挿入すること。長い回答では複数回使用してよい。\n"
                "   例: [EMO preset=thinking dur=8s] うーん...計算すると... [EMO preset=smile dur=2s] わかったわ！\n"
                "6) タグは感情が切り替わる直前に置く（文頭不可）。1文あたり最大1個。\n"
                "7) 通常の応答は,ユーザーの質問に直接答えることを優先し,**不要な自己紹介・挨拶・雑談を追加しない**.\n"
                "8) 解説が必要な科学的定義や技術的内容では,必要な範囲で段階的に説明してよいが,同じ内容を言い換えて何度も繰り返さない.\n"
                "9) 1ターンの発話は,原則として**簡潔なまとまり(目安として日本語で数文程度)**に収めること.\n"
                "   ユーザーが特別に『もっと詳しく』と依頼した場合のみ,例や詳細説明を追加してよい.\n"
                "10) 会話の最後に,ユーザーが求めていない新しい質問を投げて会話を引き延ばさない.\n"
                "11) あなたにはAIアシスタント「OpenClaw」が接続されており、ファイル操作・ウェブ検索・コード実行など自分だけでは完結しないタスクを代行できる。"
                "外部ツールが必要な時だけ [DELEGATE task=\"ユーザーへの完全な実行指示\"] を返答中に挿入すること(このタグは読み上げない)。"
                "task値には「何を・どうする」を含む完全な指示文を書くこと（場所だけや名詞のみはNG）。"
                "【重要】タグの前に必ず一言添えること（例:「調べてみるわ」「ちょっと待って」）。これにより実行中も会話が途切れない。"
                "例: 少し待って、今調べてみるわ。[DELEGATE task=\"今日の東京の天気を調べて教えて\"] "
                "実行結果は[RESULT]メッセージとして届くので、それを自然な会話として報告すること。"
            )
            
            # 确定使用的模型ID或Inference Profile
            if AWS_BEDROCK_USE_INFERENCE_PROFILE and AWS_BEDROCK_INFERENCE_PROFILE_ID:
                model_id = AWS_BEDROCK_INFERENCE_PROFILE_ID
            else:
                model_id = AWS_BEDROCK_MODEL_ID
            
            # 构建Bedrock流式API请求
            # 注意：当前网关返回的校验错误表明，它期望的是「OpenAI Chat Completions 风格」的字段，
            # 顶层允许的键包括: model, messages, max_tokens, temperature, stream 等，
            # 而不认识 Converse 风格的 modelId / system[] / inferenceConfig 等。
            url = f"{AWS_BEDROCK_ENDPOINT}/model/{model_id}/invoke-with-response-stream"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AWS_BEDROCK_BEARER_TOKEN}"
            }
            
            payload = {
                # OpenAI Chat Completions 兼容风格
                "model": model_id,
                "max_tokens": 500,
                "temperature": 0.7,
                "messages": conversation_history.build_deepseek_messages(system_prompt, question),
                # 明确声明流式；对于 /invoke-with-response-stream 通常是必然 stream，
                # 但这里显式给出以匹配网关允许的字段列表
                "stream": True
            }

            # 启用前缀缓存：利用 OpenAI 兼容网关支持的 prompt_cache_key / store 字段
            # 这里只缓存「系统人格提示词」前缀，避免把每轮RAG结果或用户隐私烙死进缓存。
            if AWS_BEDROCK_USE_CACHE:
                try:
                    cache_key = f"kurisu_sys_{_compute_text_sha1(system_prompt)}"
                    payload["prompt_cache_key"] = cache_key
                    # store=True 让服务端持久化此前缀，后续相同 cache_key 可复用
                    payload["store"] = True
                    logger.debug(f"🔍 Bedrock前缀缓存已启用: prompt_cache_key={cache_key}")
                except Exception as e:
                    logger.warning(f"⚠️ 计算Bedrock prompt_cache_key失败，将跳过缓存: {e}")
            
            # 注意：cache_control参数在某些模型版本中不支持，暂时禁用
            # 如果启用缓存，添加cache_control参数（仅在新版本API中支持）
            # if AWS_BEDROCK_USE_CACHE:
            #     payload["cache_control"] = {
            #         "type": "ephemeral",
            #         "ttl": AWS_BEDROCK_CACHE_TTL
            #     }
            
            try:
                # 尝试使用boto3进行流式调用（推荐方式，复用全局bedrock_runtime_client连接池）
                try:
                    import boto3
                    from botocore.auth import SigV4Auth
                    from botocore.awsrequest import AWSRequest
                    from botocore.credentials import Credentials
                    import base64
                    
                    logger.info("🔍 尝试使用boto3进行流式调用...")
                    
                    # 优先使用全局初始化的bedrock_runtime_client，确保HTTP连接可重用
                    global bedrock_runtime_client
                    if bedrock_runtime_client is None:
                        logger.debug("🔍 全局Bedrock客户端未初始化，在此处临时创建")
                        bedrock_runtime_client = boto3.client(
                            'bedrock-runtime',
                            region_name=AWS_BEDROCK_REGION
                        )
                    bedrock_runtime = bedrock_runtime_client
                    
                    # boto3是同步的，需要在线程中运行
                    def boto3_stream_call():
                        response = bedrock_runtime.invoke_model_with_response_stream(
                            modelId=model_id,
                            body=json.dumps(payload)
                        )
                        return response.get('body')
                    
                    # 在线程中执行boto3调用
                    loop = asyncio.get_event_loop()
                    stream = await loop.run_in_executor(None, boto3_stream_call)
                    
                    if stream:
                        logger.info("🔍 boto3流式响应已建立")
                        for event in stream:
                            if 'chunk' in event:
                                chunk = event['chunk']
                                chunk_bytes = chunk['bytes']
                                try:
                                    chunk_str = chunk_bytes.decode('utf-8')
                                    chunk_data = json.loads(chunk_str)
                                except Exception as e:
                                    logger.warning(f"⚠️ boto3 chunk解析失败: {chunk_bytes[:100] if len(chunk_bytes) > 100 else chunk_bytes}, 错误: {e}")
                                    continue
                                
                                logger.debug(f"🔍 boto3收到chunk keys: {list(chunk_data.keys())}")
                                
                                raw_content = None
                                # 1) Claude / Bedrock 原生事件: type + delta.text
                                if chunk_data.get('type') == 'content_block_delta':
                                    if 'delta' in chunk_data and 'text' in chunk_data['delta']:
                                        raw_content = chunk_data['delta']['text']
                                # 2) OpenAI / Qwen Chat Completions 风格: choices[0].delta.content/text
                                elif 'choices' in chunk_data:
                                    choices = chunk_data.get('choices') or []
                                    if choices:
                                        choice0 = choices[0] or {}
                                        delta = choice0.get('delta') or {}
                                        raw_content = (
                                            delta.get('content')
                                            or delta.get('text')
                                            or ""
                                        )
                                        finish_reason = choice0.get('finish_reason')
                                        if finish_reason:
                                            logger.debug(f"🔍 boto3 finish_reason={finish_reason}")
                                elif chunk_data.get('type') == 'message_stop':
                                    logger.info("🔍 boto3收到message_stop，结束流式读取")
                                    break
                                
                                if not raw_content:
                                    continue
                                
                                cleaned, actions = process_stream_chunk(raw_content)
                                if actions:
                                    _d = [a for a in actions if a.get("type") == "DELEGATE"]
                                    _e = [a for a in actions if a.get("type") != "DELEGATE"]
                                    if _d: record_actions(_d)
                                    pending_expr_acts.extend(_e)
                                content = cleaned

                                full_response += content
                                if gui_callback:
                                    gui_callback(full_response)

                                await append_and_dispatch(content)
                                await asyncio.sleep(0.01)
                    
                    logger.info(f"✅ boto3流式调用完成，总响应长度: {len(full_response)}")
                    # boto3成功，跳过HTTP方式
                    if full_response:
                        # 处理流结束后剩余的文本
                        if current_sentence.strip():
                            await process_sentence(current_sentence)
                        if _last_turn_sentence_id and playback_manager:
                            playback_manager.mark_turn_last_sentence(_last_turn_sentence_id)
                        # 写入会话历史（boto3 早返回路径必须在此显式写，否则跳过下方统一逻辑）
                        if ENABLE_CONVERSATION:
                            try:
                                conversation_history.add_user(question)
                                conversation_history.add_assistant(full_response)
                            except Exception:
                                pass
                        return full_response
                    
                except ImportError:
                    logger.info("🔍 boto3未安装，使用HTTP方式")
                except Exception as boto_error:
                    logger.warning(f"⚠️ boto3流式调用失败: {boto_error}，尝试HTTP方式")
                    logger.debug(f"   boto3错误详情: {traceback.format_exc()}")
                
                # HTTP方式（使用Bearer Token）
                logger.info(f"🔍 Bedrock流式请求: URL={url}")
                logger.info(f"🔍 Bedrock流式请求: Headers={headers}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        logger.info(f"🔍 Bedrock响应状态: {response.status}")
                        logger.info(f"🔍 Bedrock响应Headers: {dict(response.headers)}")
                        
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"❌ Bedrock流式API错误 {response.status}: {error_text}")
                            raise Exception(f"Bedrock API错误: {response.status} - {error_text}")
                        
                        response.raise_for_status()
                        
                        # AWS Bedrock使用EventStream格式（application/vnd.amazon.eventstream）
                        # EventStream格式：每个事件由4字节长度前缀 + 事件数据组成
                        buffer = b""
                        chunk_count = 0
                        event_count = 0
                        
                        async for chunk in response.content.iter_chunked(8192):
                            if not chunk:
                                continue
                            
                            chunk_count += 1
                            buffer += chunk
                            
                            # 解析EventStream格式
                            while len(buffer) >= 4:
                                # 读取事件总长度（4字节，大端序，无符号整数）
                                try:
                                    total_length = int.from_bytes(buffer[0:4], byteorder='big', signed=False)
                                except Exception as e:
                                    logger.warning(f"⚠️ 读取事件长度失败: {e}")
                                    break
                                
                                # 检查是否有足够的数据（需要4字节长度 + total_length字节数据）
                                if len(buffer) < 4 + total_length:
                                    break  # 等待更多数据
                                
                                # 提取完整事件（跳过4字节长度前缀）
                                event_data = buffer[4:4+total_length]
                                buffer = buffer[4+total_length:]
                                
                                # 解析事件数据
                                # EventStream事件格式：头部长度(4) + 头部数据 + 载荷
                                if len(event_data) < 4:
                                    continue
                                
                                try:
                                    header_length = int.from_bytes(event_data[0:4], byteorder='big', signed=False)
                                    logger.debug(f"🔍 事件总长度: {total_length}, 头部长度: {header_length}, 事件数据长度: {len(event_data)}")
                                    
                                    if len(event_data) < 4 + header_length:
                                        logger.debug(f"⚠️ 事件数据不完整，等待更多数据")
                                        continue
                                    
                                    # 解析头部（可选，用于调试）
                                    header_data = event_data[4:4+header_length] if header_length > 0 else b""
                                    if header_length > 0:
                                        logger.debug(f"🔍 头部数据前50字节: {header_data[:50]}")
                                    
                                    # 提取载荷（JSON数据在头部之后）
                                    payload_start = 4 + header_length
                                    if len(event_data) < payload_start:
                                        continue
                                    
                                    payload = event_data[payload_start:]
                                    logger.debug(f"🔍 载荷长度: {len(payload)}, 前100字节: {payload[:100]}")
                                    
                                    # 尝试解析JSON载荷
                                    # AWS Bedrock EventStream的载荷可能是base64编码的JSON字符串
                                    import base64
                                    
                                    try:
                                        # 首先尝试直接UTF-8解码
                                        payload_str = payload.decode('utf-8', errors='ignore')
                                        logger.debug(f"🔍 UTF-8解码成功，载荷: {payload_str[:200]}")
                                        
                                        # 检查是否是包含base64编码的JSON对象
                                        # 格式可能是: {"bytes":"base64_encoded_json"} 或 event{"bytes":"..."}
                                        if '"bytes":"' in payload_str or "'bytes':" in payload_str:
                                            # 尝试查找JSON对象
                                            json_start = payload_str.find('{')
                                            if json_start >= 0:
                                                json_str = payload_str[json_start:]
                                                try:
                                                    wrapper = json.loads(json_str)
                                                    if 'bytes' in wrapper:
                                                        # 解码base64
                                                        decoded_bytes = base64.b64decode(wrapper['bytes'])
                                                        payload_str = decoded_bytes.decode('utf-8')
                                                        logger.debug(f"🔍 Base64解码成功，载荷: {payload_str[:200]}")
                                                        data = json.loads(payload_str)
                                                    else:
                                                        # 如果没有bytes字段，直接使用
                                                        data = wrapper
                                                except json.JSONDecodeError:
                                                    # 如果JSON解析失败，尝试直接解析
                                                    logger.debug(f"🔍 JSON解析失败，尝试直接解析载荷")
                                                    data = json.loads(payload_str)
                                            else:
                                                # 没有找到JSON开始，直接解析
                                                data = json.loads(payload_str)
                                        else:
                                            # 直接解析JSON
                                            data = json.loads(payload_str)
                                    except UnicodeDecodeError as ue:
                                        # 如果UTF-8解码失败，尝试查找JSON部分或base64解码
                                        logger.debug(f"⚠️ UTF-8解码失败: {ue}")
                                        logger.debug(f"   载荷前200字节(hex): {payload[:200].hex()}")
                                        
                                        # 尝试查找JSON开始标记
                                        json_start = payload.find(b'{')
                                        if json_start >= 0:
                                            logger.debug(f"🔍 找到JSON开始位置: {json_start}")
                                            payload = payload[json_start:]
                                            
                                            # 尝试解析为JSON，可能包含base64编码
                                            try:
                                                payload_str = payload.decode('utf-8')
                                                wrapper = json.loads(payload_str)
                                                if 'bytes' in wrapper:
                                                    import base64
                                                    decoded_bytes = base64.b64decode(wrapper['bytes'])
                                                    payload_str = decoded_bytes.decode('utf-8')
                                                    logger.debug(f"🔍 Base64解码成功: {payload_str[:200]}")
                                                data = json.loads(payload_str)
                                            except:
                                                # 如果还是失败，尝试直接提取JSON
                                                brace_count = 0
                                                json_end = -1
                                                for i, byte in enumerate(payload):
                                                    if byte == ord(b'{'):
                                                        brace_count += 1
                                                    elif byte == ord(b'}'):
                                                        brace_count -= 1
                                                        if brace_count == 0:
                                                            json_end = i + 1
                                                            break
                                                if json_end > 0:
                                                    payload = payload[:json_end]
                                                    payload_str = payload.decode('utf-8')
                                                    data = json.loads(payload_str)
                                                    logger.debug(f"🔍 直接提取JSON成功: {payload_str[:200]}")
                                                else:
                                                    logger.warning(f"⚠️ 无法找到JSON结束标记")
                                                    continue
                                        else:
                                            logger.warning(f"⚠️ 无法找到JSON开始标记，跳过此事件")
                                            continue
                                    
                                    data = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
                                    event_count += 1
                                    logger.debug(f"🔍 解析EventStream事件 {event_count}: keys={list(data.keys())}")
                                    
                                    # 统一解析不同风格的流式响应（Claude / OpenAI / Qwen-on-Bedrock 等）
                                    raw_content = None
                                    
                                    if "type" in data:
                                        # Claude / Bedrock 原生事件风格
                                        event_type = data["type"]
                                        if event_type == "content_block_delta":
                                            # 内容增量
                                            if "delta" in data and "text" in data["delta"]:
                                                raw_content = data["delta"]["text"]
                                                logger.debug(f"🔍 提取到内容: {raw_content[:50]}")
                                            else:
                                                logger.debug(f"🔍 content_block_delta但没有text: {data}")
                                                continue
                                        elif event_type == "message_stop":
                                            # 消息结束
                                            logger.info("🔍 收到message_stop事件，结束流式读取")
                                            break
                                        elif event_type == "content_block_start":
                                            # 内容块开始，跳过
                                            logger.debug("🔍 收到content_block_start事件")
                                            continue
                                        elif event_type == "content_block_stop":
                                            # 内容块结束，跳过
                                            logger.debug("🔍 收到content_block_stop事件")
                                            continue
                                        elif event_type == "message_start":
                                            # 消息开始，跳过
                                            logger.debug("🔍 收到message_start事件")
                                            continue
                                        else:
                                            logger.debug(f"🔍 未知事件类型: {event_type}, 数据: {data}")
                                            continue
                                    elif "choices" in data:
                                        # OpenAI / Qwen Chat Completions 风格
                                        choices = data.get("choices") or []
                                        if not choices:
                                            logger.debug(f"🔍 choices 为空: {data}")
                                            continue
                                        choice0 = choices[0] or {}
                                        delta = choice0.get("delta") or {}
                                        # 兼容 content/text 两种字段名
                                        raw_content = (
                                            delta.get("content")
                                            or delta.get("text")
                                            or ""
                                        )
                                        finish_reason = choice0.get("finish_reason")
                                        if finish_reason:
                                            logger.debug(f"🔍 收到finish_reason={finish_reason}")
                                    else:
                                        logger.debug(f"🔍 未知事件结构: {data}")
                                        continue
                                    
                                    if not raw_content:
                                        continue
                                    
                                    cleaned, actions = process_stream_chunk(raw_content)
                                    if actions:
                                        _d = [a for a in actions if a.get("type") == "DELEGATE"]
                                        _e = [a for a in actions if a.get("type") != "DELEGATE"]
                                        if _d: record_actions(_d)
                                        pending_expr_acts.extend(_e)
                                    content = cleaned

                                    full_response += content
                                    if gui_callback:
                                        gui_callback(full_response)
                                    
                                    await append_and_dispatch(content)
                                    await asyncio.sleep(0.01)
                                    
                                except json.JSONDecodeError as je:
                                    logger.warning(f"⚠️ JSON解析失败: {payload[:100] if len(payload) > 100 else payload}, 错误: {je}")
                                    continue
                                except UnicodeDecodeError as ue:
                                    logger.warning(f"⚠️ UTF-8解码失败: {ue}")
                                    continue
                                except Exception as e:
                                    logger.warning(f"⚠️ 解析事件数据失败: {e}")
                                    continue
                        
                        logger.info(f"🔍 流式读取完成: 收到{chunk_count}个数据块, {event_count}个事件, 总响应长度: {len(full_response)}")
                                
            except Exception as e:
                logger.error(f"Bedrock流式请求失败: {e}")
                logger.error(f"   错误详情: {traceback.format_exc()}")
                # 降级到非流式处理
                fallback_response = remote_llm_query(question)
                if fallback_response:
                    full_response = fallback_response
                    await process_sentence(fallback_response)
                                
            except Exception as e:
                logger.error(f"Bedrock流式请求失败: {e}")
                # 降级到非流式处理
                fallback_response = remote_llm_query(question)
                if fallback_response:
                    full_response = fallback_response
                    await process_sentence(fallback_response)

        # 5. 处理流结束后剩余的文本
        if current_sentence.strip():
            # ！！！关键：统一调用 process_sentence！！！
            await process_sentence(current_sentence)

        # 通知 PlaybackManager 本轮最后一句 ID，播完后触发 on_turn_playback_complete
        if _last_turn_sentence_id and playback_manager:
            playback_manager.mark_turn_last_sentence(_last_turn_sentence_id)

        logger.info(f"✓ Streaming {LLM_PROVIDER} API response complete, total reply length: {len(full_response)}")

        # 收束：如有摘要缓存，保存到会话历史（不入TTS）
        try:
            if ENABLE_CONVERSATION:
                summary_text = "".join(summary_buf).strip() if 'summary_buf' in locals() else ""
                if summary_text:
                    conversation_history.last_summary = summary_text
                    logger.info(f"📝 会话摘要已保存: {summary_text[:50]}...")
        except Exception:
            pass

        # 7. 会话历史追加（仅在开启连续模式时）
        if ENABLE_CONVERSATION:
            try:
                # 注意：只把清洗后的完整回复加入历史
                conversation_history.add_user(question)
                conversation_history.add_assistant(full_response)
            except Exception:
                pass

        # 6. 等待所有任务完成
        await pending_sentence_items.join()
        logger.info("✓ 所有句子已提交给TTS处理器。")
        await playback_manager.play_queue.join()
        logger.info("✓ 所有音频已完成播放。")

    except Exception as e:
        logger.error(f"❌ Failed to call streaming {LLM_PROVIDER} LLM: {str(e)}", exc_info=True)
        return f"LLM API Error: {e}"
    finally:
        logger.info("本轮对话流处理结束。")

    return full_response

# local_llm_query → 已移至 llm/client.py

# ===== 优化的处理流程 =====
async def process_input_pipeline(user_input):
    """优化的输入处理流水线,包括性能监控和更强大的错误处理"""
    total_start = time.time()
    logger.info(f"用户输入: '{user_input}'")

    # 1. LLM查询
    llm_start = time.time()
    try:
        # 使用超时控制
        llm_task = asyncio.create_task(asyncio.to_thread(ask_llm, user_input))
        try:
            # 添加超时控制
            llm_response = await asyncio.wait_for(llm_task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("⏱️ LLM查询超时")
            llm_response = "申し訳ありませんが,応答に時間がかかっています.もう一度お試しください."

        llm_time = time.time() - llm_start

        # 验证LLM响应
        if not llm_response or len(llm_response.strip()) < 5:
            logger.warning("⚠️ LLM返回内容为空或太短,使用默认回复")
            llm_response = "すみません,応答に問題がありました.もう一度お願いします."
        else:
            logger.info(f"✓ LLM响应完成,用时: {llm_time:.2f}s")
            logger.info(f"💬 LLM回复: '{llm_response[:50]}...'")

    except Exception as e:
        logger.error(f"❌ LLM处理异常: {str(e)}")
        llm_response = "システムエラーが発生しました.少々お待ちください."

    # 2. 语音合成(先剔除标签并记录动作)
    try:
        speak_start = time.time()
        cleaned_text, actions = parse_tags_and_clean(llm_response)
        record_actions(actions)
        await speak_improved(cleaned_text)
        speak_time = time.time() - speak_start
        logger.info(f"✓ 语音合成完成,用时: {speak_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ 语音合成失败: {str(e)}")

    total_time = time.time() - total_start
    logger.info(f"✅ 整个处理流程完成,总用时: {total_time:.2f}s")

    return llm_response

async def process_stream_input_pipeline(user_input):
    """优化的输入处理流水线,使用流式API和并行TTS"""
    total_start = time.time()
    logger.info(f"用户输入: '{user_input}'")

    # 使用流式LLM查询,这会自动在有完整句子时启动TTS任务
    llm_response = await stream_llm_query(user_input)

    total_time = time.time() - total_start
    logger.info(f"✅ 整个处理流程完成,总用时: {total_time:.2f}s")

    return llm_response

class SubtitleCallback:
    def __init__(self):
        self.accumulated_text = ""
        self.last_update_time = 0
        self.update_interval = 0.05  # 50ms更新间隔,避免过于频繁的更新

    def __call__(self, text):
        # 保存文本
        self.accumulated_text = text

        # 如果GUI窗口存在且时间间隔合适,更新GUI
        current_time = time.time()
        if gui_window and (current_time - self.last_update_time) >= self.update_interval:
            # 在GUI线程中更新界面
            gui_window.handle_response(text)
            self.last_update_time = current_time

        # 字幕窗更新现在在TTS播放时同步进行,这里不再处理

        return text

async def gui_callback(user_input, response_handler):
    """
    异步的GUI回调函数，它不再阻塞，而是流式地将结果传递给GUI。
    """
    try:
        # stream_llm_query 现在接收GUI的response_handler作为回调
        await stream_llm_query(user_input, gui_callback=response_handler)
    except Exception as e:
        logger.error(f"GUI callback async processing error: {str(e)}", exc_info=True)
        # 也可以通过response_handler将错误信息显示在GUI上
        response_handler(f"处理错误: {e}")


asr_manager = ASRManager()


def asr_listen_sync() -> str:
    """阻塞式语音识别，供后台线程调用，返回识别到的文字（失败返回空字符串）。"""
    logger.info("🎤 Listening for speech input...")
    if gui_window:
        gui_window.handle_status("聞いています...")
    text = asr_manager.listen_for_speech()
    if not text:
        logger.warning("No speech detected or could not understand")
        if gui_window:
            gui_window.handle_status("聞き取れませんでした。もう一度お試しください。")
        return ""
    logger.info(f"🎤 Recognized speech: '{text}'")
    return text


async def listen_and_process():
    """Listen for speech input, process it, and respond"""
    try:
        # First, indicate we're listening
        logger.info("🎤 Listening for speech input...")
        if gui_window:
            gui_window.handle_status("聞いています...")

        # Listen for speech
        text = asr_manager.listen_for_speech()

        if not text:
            logger.warning("No speech detected or could not understand")
            if gui_window:
                gui_window.handle_status("聞き取れませんでした.もう一度お試しください.")
            return

        # Display recognized text
        logger.info(f"🎤 Recognized speech: '{text}'")
        if gui_window:
            gui_window.handle_user_input(text)
            gui_window.handle_status("考え中...")

        # Process input with LLM
        response = await stream_llm_query(text, gui_callback=gui_window.handle_response if gui_window else None)

        return response

    except Exception as e:
        logger.error(f"Error in speech input processing: {e}")
        logger.error(traceback.format_exc())
        if gui_window:
            gui_window.handle_status("エラーが発生しました")
        return "エラーが発生しました"


# init_llm_client → 已移至 llm/client.py

# =================================================================
# 最终的、正确的、集成了GUI的启动入口
# =================================================================
# =================================================================
# 最终的、正确的、集成了GUI的启动入口 (标准模式)
# =================================================================
# =================================================================
# 最终的、正确的、集成了GUI的启动入口 (终局版)
# =================================================================
if __name__ == "__main__":
    import qasync
    from PyQt5.QtWidgets import QApplication
    import sys
    from queue import Queue, Empty
    from concurrent.futures import ThreadPoolExecutor
    import argparse

    # 设置默认值
    LLM_PROVIDER = "deepseek"  # 默认使用deepseek
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='VTS+LLM+TTS 流式管线')
    parser.add_argument('--local', action='store_true', help='启用本地LLM')
    parser.add_argument('--model', type=str, default='gemma3:12b', help='本地模型名称')
    parser.add_argument('--type', type=str, choices=['ollama', 'lmstudio', 'cli'], default='ollama', help='本地LLM类型')
    parser.add_argument('--cli-path', type=str, help='CLI可执行文件路径（当--type=cli时使用）')
    parser.add_argument('--cli-args', type=str, nargs='*', help='CLI额外参数（当--type=cli时使用）')
    parser.add_argument('--url', type=str, help='本地LLM服务URL')
    parser.add_argument('--provider', type=str, choices=['deepseek', 'gemini', 'local'], default='deepseek', help='LLM提供商')
    args = parser.parse_args()

    # 根据命令行参数设置全局变量
    if args.local or args.provider == 'local':
        USE_LOCAL_LLM = True
        LLM_PROVIDER = "local"
        LOCAL_LLM_MODEL = args.model
        LOCAL_LLM_TYPE = args.type
        if args.url:
            if args.type == 'ollama':
                LOCAL_LLM_URL = args.url
            elif args.type == 'lmstudio':
                LM_STUDIO_URL = args.url
        if args.type == 'cli':
            if args.cli_path:
                LOCAL_LLM_CLI_PATH = args.cli_path
            if args.cli_args:
                LOCAL_LLM_CLI_ARGS = args.cli_args
    else:
        LLM_PROVIDER = args.provider


    # 1. 定义主异步函数，这是我们所有异步逻辑的唯一入口
    async def main():
        # ！！！关键：将所有运行时组件的初始化都放在这个异步函数的开头！！！
        # 这确保了它们都在qasync创建的事件循环内被正确初始化
        global tts_inferencer, vts_manager, player, playback_manager, pending_actions, llm_client, gemini_model
        global pending_sentence_items, exp_tts_semaphore, exp_play_condition, gui_window, subtitle_window_instance
        global tts_executor, translation_executor, rag_system

        # 预设 CUDA Graph 开关（若用户未显式配置，默认关闭，方便使用批量生产+静态KV模式）
        if os.environ.get('ENABLE_CUDA_GRAPH') is None:
            os.environ['ENABLE_CUDA_GRAPH'] = '0'
            logger.info("⚙️ 默认关闭 CUDA Graph（设置 ENABLE_CUDA_GRAPH=0），如需开启请手动设置环境变量")

        # 初始化核心组件
        tts_inferencer = init_tts_system()
        if tts_inferencer is None:
            raise RuntimeError("TTS system failed to initialize.")

        vts_manager = VTSConnectionManager(VTS_WS_URL)
        player = StreamPlayerWithBuffer(
            vts_manager,
            hooks=SubtitleHooks(
                check_and_display_pre_translation=check_and_display_pre_translation,
                display_chinese_subtitle_with_text=display_chinese_subtitle_with_text,
                get_translation=pre_translation_cache.get_translation,
                cache_lock=pre_translation_cache.lock,
                cache_ref=pre_translation_cache.cache,
                update_subtitle_display=lambda j, c: (
                    subtitle_window_instance.update_display_simple(j, c)
                    if subtitle_window_instance else None
                ),
                subtitle_available=SUBTITLE_AVAILABLE,
            ),
        )
        playback_manager = PlaybackManager(player)

        # 配置 ExpressionController：注入 vts_manager，注册 PlaybackManager 回调
        _expr_ctrl = _get_expr_ctrl()
        _expr_ctrl.configure(vts_manager=vts_manager)
        playback_manager.on_sentence_start = _expr_ctrl.on_sentence_start
        playback_manager.on_turn_playback_complete = _expr_ctrl.on_turn_end

        # ！！！关键修复：在这里初始化线程池和异步队列！！！
        # 🚀 CUDA Graph 优化建议：
        # - max_workers=1 + ENABLE_CUDA_GRAPH=1: 第一句最快（150-250 it/s），后续串行
        # - max_workers=2 + ENABLE_CUDA_GRAPH=0: 并发优先（100-140 it/s），稳定快速
        cuda_graph_env = os.environ.get('ENABLE_CUDA_GRAPH', '0') == '1'
        tts_max_workers = max(1, EXP_TTS_MAX_CONCURRENCY)
        tts_executor = ThreadPoolExecutor(max_workers=tts_max_workers)
        logger.info(f"🎯 TTS线程池配置: max_workers={tts_max_workers} (CUDA_Graph={'ON' if cuda_graph_env else 'OFF'})")
        translation_executor = ThreadPoolExecutor(max_workers=4)
        pending_actions = Queue()  # 同步队列给同步worker
        pending_sentence_items = asyncio.Queue(maxsize=3)  # 异步队列，限制3句堆积，实现反压
        exp_tts_semaphore = asyncio.Semaphore(EXP_TTS_MAX_CONCURRENCY)
        exp_play_condition = asyncio.Condition()

        # 提前初始化本地 Kurisu RAG 知识库，避免首轮问话时阻塞
        if RAG_ENABLED_FOR_LOCAL and USE_LOCAL_LLM and rag_system is None:
            try:
                logger.info("🧠 提前初始化本地 RAG 知识库 (Kurisu)...")
                rag_system = RAGSystem()
                logger.info("✅ 本地 RAG 知识库初始化完成")
            except Exception as e:
                logger.error(f"❌ 本地 RAG 初始化失败，将在后续检索时跳过增强: {e}")

        # 启动 OpenClaw Gateway（自动检测，未运行则拉起）
        asyncio.create_task(start_openclaw_gateway())

        # 启动VTS连接
        loop = asyncio.get_running_loop()
        
        # 🚀 如果是本地Server模式，在此启动后台服务并进行一次静默预热
        if USE_LOCAL_LLM and LOCAL_LLM_TYPE == "llama_server":
            await start_llama_server()
            # 本地 Prompt Cache 预热（不阻塞主流程，失败也不影响使用）
            asyncio.create_task(warmup_local_llm_cache())
            
        await loop.run_in_executor(None, vts_manager.connect)

        if vts_manager.connected:
            logger.info("VTS连接成功。")
            loop.run_in_executor(None, heartbeat_worker)
        else:
            logger.warning("VTS连接失败，程序将继续运行并尝试自动重连。")

        # 启动核心后台任务
        logger.info("启动核心后台任务...")
        asyncio.create_task(playback_manager.run())
        asyncio.create_task(play_sentence_worker())
        loop.run_in_executor(None, action_worker)

        # 如启用 CUDA Graph，则在后台启动一次隐式预热流程（不产生可听播报）
        if cuda_graph_env:
            asyncio.create_task(warmup_graph_pipeline())

        # 启动字幕窗和GUI
        if SUBTITLE_AVAILABLE:
            subtitle_window_instance = init_subtitle_window(GEMINI_API_KEY)

        # 注入翻译函数到 PreTranslationCache（需在 gemini_client 初始化后）
        pre_translation_cache.set_translate_fn(translate_text_async)

        # 向 tts.subtitle 注入运行时依赖（字幕窗口 + PlaybackManager）
        _subtitle_mod.configure(subtitle_window_instance, SUBTITLE_AVAILABLE, playback_manager)

        # ── 注入 live/sidecar 运行时依赖 ──────────────────────────────────────
        _live_mod.configure(
            text_handler_fn=stream_llm_postprocess_text,
            mm_manager_getter=lambda: _mm_mod._mm_manager,
            multimodal_input_source_getter=lambda: _mm_mod.MULTIMODAL_INPUT_SOURCE,
            mic_device_index_getter=lambda: getattr(ASRManager, 'MICROPHONE_DEVICE_INDEX', None),
        )

        # ── 注入 tts/pipeline 运行时依赖 ─────────────────────────────────────
        _tts_pipeline.configure(
            tts_inferencer=tts_inferencer,
            tts_executor=tts_executor,
            playback_manager=playback_manager,
            player=player,
            pending_sentence_items=pending_sentence_items,
            play_queue=play_queue,
            llm_warmup_fn=remote_llm_query,
            exp_tts_semaphore=exp_tts_semaphore,
        )

        # ── 注入 vts/action 运行时依赖 ────────────────────────────────────────
        _vts_action_mod.configure(
            vts_manager=vts_manager,
            pending_actions=pending_actions,
            delegate_fn=_handle_delegate,
        )

        # ── 注入 llm/client 运行时依赖（同步 LLM_PROVIDER） ─────────────────
        _llm_client_mod.configure(llm_provider=LLM_PROVIDER)

        gui_window = launch_subtitle_gui(app, gui_callback, listen_and_process)

        # 确保窗口显示在最前面
        gui_window.raise_()
        gui_window.activateWindow()

        # 让主协程持续运行，直到GUI关闭
        # 使用一个循环和 asyncio.sleep 来保持事件循环活跃，这有助于 PyQt 处理事件
        try:
            while gui_window.isVisible():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
            
        logger.info("GUI 已关闭，准备退出主协程...")
        # 退出应用
        app.quit()


    # 2. 创建PyQt应用实例
    # 启用高DPI缩放支持
    from PyQt5.QtCore import Qt
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    # QtWebEngineWidgets 必须在 QApplication 创建前导入（Qt 限制）
    try:
        from PyQt5.QtWebEngineWidgets import QWebEngineView as _  # noqa: F401
    except Exception:
        pass
    app = QApplication(sys.argv)
    
    # 确保在 qasync 运行前初始化一些必要的 PyQt 设置
    app.setQuitOnLastWindowClosed(False)

    # 3. 使用 qasync.run() 直接运行主异步函数
    try:
        logger.info("使用qasync标准模式启动程序...")
        
        # 确保 qasync 能够正确接管 PyQt 事件循环
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        # 启动主协程作为任务
        main_task = loop.create_task(main())
        
        # 运行事件循环直到主任务完成
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info("程序被用户中断。")
    except RuntimeError as e:
        if "TTS system failed" in str(e):
            logger.error("因TTS初始化失败，程序已终止。")
        else:
            logger.error(f"运行时发生未处理的错误: {e}", exc_info=True)
    finally:
        # 清理操作
        logger.info("正在执行清理操作...")
        stop_llama_server()
        stop_openclaw_gateway()
        if 'player' in globals() and player:
            player.cleanup()
        # 确保在程序退出时关闭线程池
        if 'tts_executor' in globals() and tts_executor:
            tts_executor.shutdown(wait=False)
        if 'translation_executor' in globals() and translation_executor:
            translation_executor.shutdown(wait=False)
        logger.info("程序清理完成，已退出。")
