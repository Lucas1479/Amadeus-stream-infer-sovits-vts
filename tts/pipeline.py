"""tts/pipeline.py — TTS 合成 + 播放调度管线

负责：
  - TTS 参数选择（get_sovits_params）
  - 各种 speak_* 合成策略（改进版 / 流式 / Graph串行 / 增强异步队列）
  - 句子调度工作线程（play_sentence_worker）
  - CUDA Graph 管线预热（warmup_graph_pipeline）

运行时状态（模块级，供 main.py 读写）：
  exp_tts_semaphore, exp_play_condition, graph_tts_lock

依赖注入（通过 configure() 在 async main() 完成初始化后调用）：
  tts_inferencer, tts_executor, playback_manager, player,
  pending_sentence_items, play_queue, llm_warmup_fn
"""

import asyncio
import logging
import os
import time
import traceback

import numpy as np

from config.settings import (
    SEGMENT_CHAR_LIMIT,
    USE_EXPERIMENTAL_TTS_STREAM,
    USE_FIRST_SENTENCE_SPRINT,
)
from tools.text_utils import (
    _compute_text_sha1,
    _parse_sentence_seq,
    async_generator_from_sync,
)
from tools.tts_text_processor import correct_pronunciation_for_tts

logger = logging.getLogger(__name__)

# ===== 运行时状态（供外部模块读写）=====
exp_tts_semaphore = None
exp_play_condition = None
graph_tts_lock = asyncio.Lock()

# ===== 依赖注入 —— configure() 填充这些 =====
_tts_inferencer = None
_tts_executor = None
_playback_manager = None
_player = None
_pending_sentence_items = None
_play_queue = None
_llm_warmup_fn = None   # remote_llm_query，供 warmup_graph_pipeline 使用


def configure(
    tts_inferencer=None,
    tts_executor=None,
    playback_manager=None,
    player=None,
    pending_sentence_items=None,
    play_queue=None,
    llm_warmup_fn=None,
):
    """在 async main() 完成运行时初始化后调用，注入所有运行时依赖。"""
    global _tts_inferencer, _tts_executor, _playback_manager, _player
    global _pending_sentence_items, _play_queue, _llm_warmup_fn
    if tts_inferencer is not None:
        _tts_inferencer = tts_inferencer
    if tts_executor is not None:
        _tts_executor = tts_executor
    if playback_manager is not None:
        _playback_manager = playback_manager
    if player is not None:
        _player = player
    if pending_sentence_items is not None:
        _pending_sentence_items = pending_sentence_items
    if play_queue is not None:
        _play_queue = play_queue
    if llm_warmup_fn is not None:
        _llm_warmup_fn = llm_warmup_fn


# =============================================================================
# TTS 参数选择
# =============================================================================

def get_sovits_params(text: str, is_first_sentence: bool = False):
    """根据文本长度和是否为首句返回合适的推理参数。

    CUDA Graph 开关仅由环境变量 ENABLE_CUDA_GRAPH 控制，静态 KV Cache 始终开启。
    """
    length = len(text.strip())
    cuda_graph_env = os.environ.get("ENABLE_CUDA_GRAPH", "0") == "1"

    if is_first_sentence:
        max_sec_override = max(3.5, min(8.0, length * 0.25 or 3.5))
        return {
            "text_language": "日文",
            "prompt_language": "日文",
            "top_k": 5,
            "top_p": 1,
            "temperature": 0.6,
            "sample_steps": 8,
            "if_sr": False,
            "how_to_cut": "不切",
            "speed": 1.1,
            "pause_second": 0.1,
            "if_freeze": False,
            "enable_cuda_graph": cuda_graph_env,
            "enable_static_kv": True,
            "max_sec_override": max_sec_override,
        }

    if length < 45:
        return {
            "text_language": "日文",
            "prompt_language": "日文",
            "top_k": 5,
            "top_p": 1,
            "temperature": 0.6,
            "sample_steps": 16,
            "if_sr": False,
            "how_to_cut": "不切",
            "speed": 1,
            "pause_second": 0.2,
            "if_freeze": False,
            "enable_cuda_graph": cuda_graph_env,
            "enable_static_kv": True,
        }

    return {
        "text_language": "日文",
        "prompt_language": "日文",
        "top_k": 5,
        "top_p": 1,
        "temperature": 0.6,
        "sample_steps": 32,
        "if_sr": False,
        "how_to_cut": "凑四句一切",
        "speed": 1,
        "pause_second": 0.35,
        "if_freeze": False,
        "enable_cuda_graph": cuda_graph_env,
        "enable_static_kv": True,
    }


# =============================================================================
# 合成策略
# =============================================================================

_REF_AUDIO = "./reference audio/kurisu_reference.wav"
_REF_TEXT = "そういえば,正式に自己紹介していませんでしたね……牧瀬紅莉栖です.改めてまして,よろしく"


async def generate_segment_improved(text_segment):
    """改进的分段生成函数（本地推理），增加性能监控和错误处理。"""
    if _tts_inferencer is None:
        logger.error("TTS推理器未初始化，无法生成语音")
        return np.zeros(16000, dtype=np.float32), 16000

    params = get_sovits_params(text_segment)
    params['ref_audio_path'] = _REF_AUDIO
    params['prompt_text'] = _REF_TEXT

    start_time = time.time()
    infer_start = time.time()
    logger.info(f"开始TTS推理: '{text_segment[:20]}...'")

    try:
        sr, audio_data = await asyncio.to_thread(
            _tts_inferencer.infer,
            text_segment,
            params['ref_audio_path'],
            params['prompt_text'],
            params["text_language"],
            params["prompt_language"],
            params["how_to_cut"],
            params.get("top_k", 20),
            params["top_p"],
            params["temperature"],
            params["speed"],
            params["sample_steps"],
            params.get("ref_free", False),
            params["pause_second"],
            params.get("if_freeze", False),
            None,
            params.get("if_sr", False),
            params.get("enable_cuda_graph", False),
            params.get("enable_static_kv", True),
            params.get("max_sec_override"),
        )
        infer_time = time.time() - infer_start
        if audio_data is None or not isinstance(audio_data, np.ndarray):
            logger.warning("生成的音频无效或类型不正确")
            return np.zeros(16000, dtype=np.float32), 16000
        logger.info(f"TTS推理完成，用时: {infer_time:.2f}s，音频长度: {len(audio_data)}")
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        logger.info(f"完成合成 -> {text_segment[:20]}... (总用时: {time.time() - start_time:.2f}s)")
        return audio_data, sr
    except Exception as e:
        logger.error(f"本地TTS推理失败: {e}\n{traceback.format_exc()}")
        return np.zeros(16000, dtype=np.float32), 16000


async def speak_improved(text):
    """改进的语音处理流程（非流式），增加性能监控。"""
    import re
    overall_start = time.time()
    sentences = re.split(r"[.!?\\n]", text)
    segments, current = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) < SEGMENT_CHAR_LIMIT:
            current += s + "."
        else:
            segments.append(current)
            current = s + "."
    if current:
        segments.append(current)
    logger.info(f"文本分段完成: {len(segments)}段，用时: {time.time() - overall_start:.2f}s")

    tasks = [generate_segment_improved(seg) for seg in segments]
    results = await asyncio.gather(*tasks)
    for audio_data, sample_rate in (r for r in results if r is not None):
        if _play_queue is not None:
            _play_queue.put((audio_data, sample_rate))
    logger.info(f"语音处理完成，总用时: {time.time() - overall_start:.2f}s")


async def speak_stream(text):
    """简化的流式语音处理流程 —— 移除翻译阻塞。"""
    if _tts_inferencer is None:
        logger.error("TTS推理器未初始化，无法生成语音")
        return

    overall_start = time.time()
    if text and text[-1] not in {',', '.', ',', '.', '?', '!', '?', '!'}:
        text += "."
    text = correct_pronunciation_for_tts(text)
    params = get_sovits_params(text)
    params['ref_audio_path'] = _REF_AUDIO
    params['prompt_text'] = "そういえば,正式に自己紹介していませんでしたね……牧瀬紅莉栖です.改めてまして,よろしく"

    try:
        first_chunk = True
        sentence_id = None  # speak_stream 直接使用 player，无 sentence_id 语义

        async for sr, audio_chunk in async_generator_from_sync(
            lambda: _tts_inferencer.infer_stream(
                text=text,
                ref_audio_path=params['ref_audio_path'],
                prompt_text=params['prompt_text'],
                text_language=params["text_language"],
                prompt_language=params["prompt_language"],
                how_to_cut=params["how_to_cut"],
                top_p=params["top_p"],
                top_k=params.get("top_k", 20),
                temperature=params["temperature"],
                sample_steps=params["sample_steps"],
                speed=params["speed"],
                if_sr=params["if_sr"],
                pause_second=params["pause_second"],
            )
        ):
            if first_chunk:
                if _player is not None:
                    _player.initialize(sr)
                first_chunk = False
            if audio_chunk is not None and len(audio_chunk) > 0 and _player is not None:
                _player.add_to_buffer(audio_chunk, text, sentence_id)
    except Exception as e:
        logger.error(f"流式处理失败: {e}\n{traceback.format_exc()}")
    logger.info(f"流式处理完成，总用时: {time.time() - overall_start:.2f}s")


async def speak_stream_graph_serial(text, sentence_id, is_first_sentence=False):
    """Graph 模式专用：全局锁确保串行推理，播放仍可与下一句推理并行。"""
    global graph_tts_lock
    if graph_tts_lock is None:
        graph_tts_lock = asyncio.Lock()
    logger.info(f"[Graph Serial] 等待获取串行推理锁: {sentence_id}")
    async with graph_tts_lock:
        start_time = time.time()
        logger.info(f"[Graph Serial] 已获取锁，开始串行推理: {sentence_id}")
        try:
            await speak_stream_enhanced_asyncio_queue(
                text,
                sentence_id,
                is_first_sentence=is_first_sentence,
                force_graph=True,
            )
        finally:
            elapsed = time.time() - start_time
            logger.info(f"[Graph Serial] 释放串行推理锁: {sentence_id}，TTS耗时 {elapsed:.2f}s")


async def speak_stream_enhanced(text, sentence_id, is_first_sentence=False, chunk_size_seconds=None):
    """增强的流式语音处理，支持状态管理和首句优化，第一句使用真正的流式播放。"""
    try:
        _sha = _compute_text_sha1(text)
    except Exception:
        _sha = "sha_err"
    logger.info(f"[TTS-FUNC-ENTER] func=speak_stream_enhanced id={sentence_id} sha1={_sha} first={is_first_sentence}")

    if _tts_inferencer is None:
        logger.error("TTS推理器未初始化，无法生成语音")
        return

    overall_start = time.time()
    if text and text[-1] not in {',', '.', ',', '.', '?', '!', '?', '!', '。', '！', '？', '、', '，'}:
        text += "。"
    text = correct_pronunciation_for_tts(text)
    processed_text = text

    logger.info(f"开始流式处理文本: '{text[:50]}...' (首句: {is_first_sentence})")
    params = get_sovits_params(text, is_first_sentence)
    params['ref_audio_path'] = _REF_AUDIO
    params['prompt_text'] = _REF_TEXT

    try:
        first_chunk = True
        audio_chunks = []
        chunk_count = 0
        tracking_streaming = chunk_size_seconds is not None and chunk_size_seconds > 0
        chunk_flush_threshold = 1 if tracking_streaming else 2

        def create_stream_generator():
            return _tts_inferencer.infer_stream(
                text=processed_text,
                ref_audio_path=params['ref_audio_path'],
                prompt_text=params['prompt_text'],
                text_language=params["text_language"],
                prompt_language=params["prompt_language"],
                how_to_cut=params["how_to_cut"],
                top_p=params["top_p"],
                top_k=params.get("top_k", 20),
                temperature=params["temperature"],
                sample_steps=params["sample_steps"],
                speed=params["speed"],
                if_sr=params["if_sr"],
                pause_second=params["pause_second"],
                chunk_size_seconds=chunk_size_seconds,
                max_sec_override=params.get("max_sec_override"),
            )

        async for sr, audio_chunk, text_item in async_generator_from_sync(create_stream_generator):
            if first_chunk:
                first_chunk = False
                logger.info(f"[流式播放] 开始流式合成第一个句子: {sentence_id}")
            if audio_chunk is not None and len(audio_chunk) > 0:
                audio_chunks.append(audio_chunk)
                chunk_count += 1
                logger.debug(f"[流式合成] 收集音频块: {len(audio_chunk)} samples")
                if len(audio_chunks) >= chunk_flush_threshold and _playback_manager is not None:
                    partial_audio = np.concatenate(audio_chunks)
                    audio_chunks = []
                    payload_text = text if chunk_count <= 2 else ""
                    await _playback_manager.add_streaming_chunk(
                        partial_audio, sentence_id, payload_text,
                        is_first_chunk=(chunk_count <= 2),
                        is_last_chunk=False,
                    )

        if audio_chunks and _playback_manager is not None:
            remaining_audio = np.concatenate(audio_chunks)
            payload_text = text if chunk_count <= 2 else ""
            await _playback_manager.add_streaming_chunk(
                remaining_audio, sentence_id, payload_text,
                is_first_chunk=(chunk_count <= 2),
                is_last_chunk=True,
            )
            logger.info(f"[流式播放] 播放剩余音频块")

        logger.info("[流式播放] 第一句流式合成完成，播放由PlaybackManager管理")
    except Exception as e:
        logger.error(f"流式处理失败: {e}\n{traceback.format_exc()}")
    logger.info(f"流式处理完成，总用时: {time.time() - overall_start:.2f}s")


async def speak_stream_enhanced_asyncio_queue(
    text, sentence_id, is_first_sentence=False, *, force_graph: bool = False
):
    """新版生产者：合成一句完整的音频，然后提交给 PlaybackManager。"""
    try:
        _sha = _compute_text_sha1(text)
    except Exception:
        _sha = "sha_err"
    logger.info(
        f"[TTS-FUNC-ENTER] func=speak_stream_enhanced_asyncio_queue "
        f"id={sentence_id} sha1={_sha} first={is_first_sentence}"
    )

    if _tts_inferencer is None:
        logger.error("TTS推理器未初始化，无法合成语音")
        return

    def tts_producer(loop, queue, producer_text, producer_params):
        """在后台线程中运行的 TTS 生产者。"""
        try:
            logger.info(f"[后台线程] TTS生产者线程已启动: {sentence_id}")
            for item in _tts_inferencer.infer_stream(text=producer_text, **producer_params):
                loop.call_soon_threadsafe(queue.put_nowait, item)
            logger.info(f"[后台线程] TTS生产者线程正常完成: {sentence_id}")
        except Exception as e:
            logger.error(f"[后台线程] TTS生产者线程出错 ({sentence_id}): {e}")
            logger.error(f"--- 后台线程详细Traceback ---\n{traceback.format_exc()}")
            loop.call_soon_threadsafe(queue.put_nowait, ("__ERROR__", e))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("__DONE__", None))

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    params = get_sovits_params(text, is_first_sentence)
    if force_graph:
        params["enable_cuda_graph"] = True
        params["enable_static_kv"] = True
        logger.info(f"[Graph Serial] 强制启用CUDA Graph参数: sentence_id={sentence_id}")
    params['ref_audio_path'] = _REF_AUDIO
    params['prompt_text'] = _REF_TEXT
    processed_text = correct_pronunciation_for_tts(text)

    loop.run_in_executor(_tts_executor, tts_producer, loop, queue, processed_text, params)
    logger.info(f"[监控] TTS生产者任务已提交到后台线程: {sentence_id}")

    audio_chunks = []
    while True:
        item = await queue.get()
        if isinstance(item, tuple) and isinstance(item[0], str) and item[0].startswith("__"):
            signal, data = item
            if signal == "__DONE__":
                break
            if signal == "__ERROR__":
                logger.error(f"主线程收到来自后台的错误信号: {data}")
                return
        sr, audio_chunk, text_item = item
        if audio_chunk is not None and len(audio_chunk) > 0:
            audio_chunks.append(audio_chunk)
            if len(audio_chunks) == 1:
                logger.info(f"[TTS-CHUNK] 首个音频块生成: {sentence_id} ({len(audio_chunk)} samples)")

    if audio_chunks and _playback_manager is not None:
        full_audio_data = np.concatenate(audio_chunks)
        logger.info(f"[监控] TTS合成完成，准备提交播放列表: {sentence_id}")
        sentence_seq = _parse_sentence_seq(sentence_id)
        if USE_FIRST_SENTENCE_SPRINT:
            if sentence_seq == 2:
                logger.info("[串音修复] 第二句合成完成，等待第一句播放完成后再播放")
                await _playback_manager.player_is_ready.wait()
                logger.info("[串音修复] 第一句播放完成，第二句开始播放")
        else:
            if sentence_seq == 2:
                logger.info("[串音修复] 批量模式：第二句等待第一句")
                await _playback_manager.player_is_ready.wait()
                logger.info("[串音修复] 第一句播放完成，第二句开始播放")
        await _playback_manager.add_to_playlist(full_audio_data, sentence_id, text)
    else:
        logger.warning(f"TTS任务未生成任何音频数据: {sentence_id}")


# =============================================================================
# 句子调度工作线程
# =============================================================================

async def play_sentence_worker():
    """从 pending_sentence_items 队列取出句子，创建对应 TTS 合成任务。

    职责简化：仅调度，播放完全交给 PlaybackManager。
    """
    logger.info("句子处理工作线程已启动")
    while True:
        try:
            sentence_id, sentence, is_first = await _pending_sentence_items.get()
            logger.info(
                f"从队列取出句子，准备TTS任务: '{sentence[:30]}...' "
                f"(ID: {sentence_id}, 首句: {is_first})"
            )
            try:
                _sha = _compute_text_sha1(sentence)
            except Exception:
                _sha = "sha_err"
            if not hasattr(play_sentence_worker, "_intent_counts"):
                play_sentence_worker._intent_counts = {}
            cnt = play_sentence_worker._intent_counts.get(sentence_id, 0) + 1
            play_sentence_worker._intent_counts[sentence_id] = cnt
            logger.info(f"[TTS-START-INTENT] id={sentence_id} sha1={_sha} first={is_first} intent_count={cnt}")

            graph_enabled = os.environ.get('ENABLE_CUDA_GRAPH', '0') == '1'
            if graph_enabled:
                logger.info("[Graph Serial Mode] CUDA Graph已启用，切换串行TTS生产")
                tts_coro = speak_stream_graph_serial(sentence, sentence_id, is_first)
            elif USE_EXPERIMENTAL_TTS_STREAM:
                logger.info("[Batch Synthesis] 批量合成+并发处理")
                tts_coro = speak_stream_enhanced_asyncio_queue(sentence, sentence_id, is_first)
            else:
                tts_coro = speak_stream_enhanced(sentence, sentence_id, is_first)

            asyncio.create_task(tts_coro)
            _pending_sentence_items.task_done()

        except asyncio.CancelledError:
            logger.info("句子处理工作线程被取消")
            break
        except Exception as e:
            logger.error(f"句子处理工作线程异常: {e}", exc_info=True)


# =============================================================================
# CUDA Graph 预热
# =============================================================================

async def warmup_graph_pipeline():
    """在启用 CUDA Graph 时进行一次隐式预热（不产生可听播放）。"""
    try:
        if os.environ.get('ENABLE_CUDA_GRAPH', '0') != '1':
            return
        if _tts_inferencer is None:
            return
        logger.info("[Graph Warmup] 启动 CUDA Graph 预热流程...")

        if _llm_warmup_fn is not None:
            try:
                warmup_question = "ごく簡単に自己紹介を一文だけでして。"
                _llm_warmup_fn(warmup_question)
                logger.info("[Graph Warmup] LLM 预热已触发")
            except Exception as e:
                logger.warning(f"[Graph Warmup] LLM 预热失败: {e}")

        try:
            warmup_text = "これはテストです。"
            params = get_sovits_params(warmup_text, is_first_sentence=True)
            params['ref_audio_path'] = _REF_AUDIO
            params['prompt_text'] = _REF_TEXT
            logger.info("[Graph Warmup] 开始本地 TTS 预热推理...")
            start_t = time.time()
            await asyncio.to_thread(
                _tts_inferencer.infer,
                warmup_text,
                params['ref_audio_path'],
                params['prompt_text'],
                params["text_language"],
                params["prompt_language"],
                params["how_to_cut"],
                params.get("top_k", 20),
                params["top_p"],
                params["temperature"],
                params["speed"],
                params["sample_steps"],
                params.get("ref_free", False),
                params["pause_second"],
                params.get("if_freeze", False),
                None,
                params.get("if_sr", False),
                True,   # enable_cuda_graph
                True,   # enable_static_kv
                params.get("max_sec_override"),
            )
            logger.info(f"[Graph Warmup] 本地 TTS 预热完成，用时 {time.time() - start_t:.2f}s（音频已丢弃）")
        except Exception as e:
            logger.warning(f"[Graph Warmup] TTS 预热失败: {e}")

        logger.info("[Graph Warmup] 预热流程结束，后续对话将直接复用 Graph / 静态 KV。")
    except Exception as e:
        logger.warning(f"[Graph Warmup] 预热协程异常: {e}")
