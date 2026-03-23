"""
音频播放引擎
- StreamPlayer：底层音频流 + 口型同步（pyaudio）
- PlaybackManager：基于句子序号的顺序播放调度器
- StreamPlayerWithBuffer：带缓冲区 + 字幕集成的播放器（包含子类扩展）

注意：StreamPlayerWithBuffer 中的字幕回调通过 SubtitleHooks dataclass 注入，
避免直接引用 main.py 全局变量。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Any

import numpy as np
import pyaudio
import torch

from tools.text_utils import _parse_sentence_seq
from config.settings import USE_FIRST_SENTENCE_SPRINT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 字幕回调容器（依赖注入，替代直接引用 main.py 全局）
# ---------------------------------------------------------------------------
@dataclass
class SubtitleHooks:
    """
    StreamPlayerWithBuffer 所需的字幕 / 翻译回调。
    main.py 在构造 StreamPlayerWithBuffer 时传入。
    """
    # async fn(sentence_id, japanese_text) -> None
    check_and_display_pre_translation: Callable[..., Coroutine] | None = None
    # async fn(sentence_id, japanese_text, chinese_text) -> None
    display_chinese_subtitle_with_text: Callable[..., Coroutine] | None = None
    # async fn(japanese_text) -> dict | None  (pre_translation_cache.get_translation)
    get_translation: Callable[..., Coroutine] | None = None
    # async context manager lock for pre_translation_cache.cache access
    cache_lock: asyncio.Lock | None = None
    # ref to cache dict itself (for fuzzy match)
    cache_ref: dict | None = None
    # sync fn(japanese_text, chinese_text) -> None
    update_subtitle_display: Callable[[str, str], None] | None = None
    # bool: whether subtitle window is available
    subtitle_available: bool = False


# ---------------------------------------------------------------------------
# StreamPlayer
# ---------------------------------------------------------------------------
class StreamPlayer:
    def __init__(self, vts_manager):
        self.vts_manager = vts_manager
        self.chunk_size = 512
        self.volume_multiplier = 3.75
        self.send_interval = 0.05
        self.is_playing = False
        self.pyaudio_instance = None
        self.stream = None
        self.last_send_time = 0

    def initialize(self, sample_rate: int) -> None:
        if self.pyaudio_instance is None:
            self.pyaudio_instance = pyaudio.PyAudio()

        current_rate = getattr(self, "_current_rate", None)
        if self.stream is not None and self.is_playing and current_rate == sample_rate:
            self.last_send_time = time.time()
            return

        if self.stream is not None:
            self.stop()

        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True,
        )
        self._current_rate = sample_rate
        self.is_playing = True
        self.last_send_time = time.time()

    def play_chunk(self, audio_chunk) -> None:
        if not self.is_playing or self.stream is None:
            return

        if torch.is_tensor(audio_chunk):
            audio_chunk = audio_chunk.cpu().detach().numpy()
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        for i in range(0, len(audio_chunk), self.chunk_size):
            if not self.is_playing:
                break
            chunk = audio_chunk[i : i + self.chunk_size]
            if len(chunk) == 0:
                break
            self.stream.write(chunk.tobytes())

            current_time = time.time()
            if current_time - self.last_send_time >= self.send_interval:
                rms = np.sqrt(np.mean(chunk ** 2))
                mouth_value = min(1.0, rms * self.volume_multiplier)
                self.vts_manager.send_mouth_data(mouth_value)
                self.last_send_time = current_time

    def stop(self) -> None:
        self.is_playing = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.vts_manager.send_mouth_data(0.0)

    def cleanup(self) -> None:
        self.stop()
        if self.pyaudio_instance is not None:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None


# ---------------------------------------------------------------------------
# PlaybackManager
# ---------------------------------------------------------------------------
class PlaybackManager:
    """
    中心化顺序播放调度器，确保 TTS 句子按文本顺序输出。
    """

    def __init__(self, player_instance: StreamPlayer):
        self.player = player_instance
        self.play_queue: asyncio.Queue = asyncio.Queue()
        self.player_is_ready = asyncio.Event()
        self.player_is_ready.set()
        self.current_playing_id = None
        self.logger = logging.getLogger("PlaybackManager")

        self.pending_audio: dict[int, tuple] = {}
        self.next_seq_to_play = 1
        self.play_condition = asyncio.Condition()

    async def add_to_playlist(
        self, full_audio_data, sentence_id: str, japanese_text: str
    ) -> None:
        sentence_seq = _parse_sentence_seq(sentence_id)
        async with self.play_condition:
            self.pending_audio[sentence_seq] = (full_audio_data, sentence_id, japanese_text)
            self.logger.info(
                f"🎶 [监控] 句子 '{sentence_id}' (序号: {sentence_seq}) 的音频已完成并加入播放列表。"
            )
            self.logger.info(
                f"🔍 [调试] 当前pending_audio: {list(self.pending_audio.keys())}, "
                f"next_seq_to_play: {self.next_seq_to_play}"
            )
            self.play_condition.notify()

    async def add_streaming_chunk(
        self,
        audio_chunk,
        sentence_id: str,
        japanese_text: str,
        is_first_chunk: bool = False,
        is_last_chunk: bool = False,
    ) -> None:
        """流式播放模式：立即播放第一句音频块。"""
        sentence_seq = _parse_sentence_seq(sentence_id)

        if sentence_seq == 1:
            self.logger.info(f"🎵 [流式播放] 立即播放第一句音频块: {sentence_id}")
            await self.player_is_ready.wait()
            self.player_is_ready.clear()
            self.current_playing_id = sentence_id
            self.logger.info(f"🔍 [调试] 第一句流式播放开始，player_is_ready已清除")
            self.logger.info(
                f"[PLAYBACK] 接收到首个音频块并开始流式播放: {sentence_id} "
                f"(大小: {len(audio_chunk)} samples)"
            )

            asyncio.create_task(
                self.player.play_audio_chunk_and_signal_completion(
                    audio_chunk,
                    sentence_id,
                    japanese_text,
                    self.player_is_ready,
                    is_last_chunk=is_last_chunk,
                )
            )

            if is_first_chunk:
                self.next_seq_to_play = 2
                self.logger.info("🔄 [顺序优化] 第一句开始播放，立即更新下一个播放序号为2")
                if USE_FIRST_SENTENCE_SPRINT:
                    self.player_is_ready.set()
                    self.logger.info("🎵 [批量TTS时机] 首句冲刺模式：第一句开始播放，立即允许批量TTS开始")
                else:
                    self.logger.info("🎵 [批量模式] 批量生产模式：第一句播放，批量TTS时机由PlaybackManager控制")
                async with self.play_condition:
                    self.play_condition.notify()

            if is_last_chunk:
                self.next_seq_to_play = 2
                self.logger.info("🔄 [顺序确认] 第一句流式播放完成，确保序号更新为2")
                async with self.play_condition:
                    self.play_condition.notify()
        else:
            self.logger.warning(
                f"⚠️ [流式播放] 非第一句尝试使用流式播放，回退到正常播放: {sentence_id}"
            )
            await self.add_to_playlist(audio_chunk, sentence_id, japanese_text)

    async def run(self) -> None:
        """播放主循环，按句子序号顺序消费 pending_audio。"""
        self.logger.info("🎬 [监控] 播放管理器 'run' 循环已启动，等待播放任务...")
        while True:
            try:
                async with self.play_condition:
                    while self.next_seq_to_play not in self.pending_audio:
                        self.logger.debug(
                            f"⏳ 等待句子序号 {self.next_seq_to_play} 完成..."
                        )
                        self.logger.info(
                            f"🔍 [调试] 当前pending_audio: {list(self.pending_audio.keys())}, "
                            f"next_seq_to_play: {self.next_seq_to_play}"
                        )
                        await self.play_condition.wait()

                    full_audio_data, sentence_id, japanese_text = self.pending_audio[
                        self.next_seq_to_play
                    ]
                    del self.pending_audio[self.next_seq_to_play]
                    self.next_seq_to_play += 1
                    self.logger.info(
                        f"🔍 [调试] 播放序号 {self.next_seq_to_play - 1}, "
                        f"更新next_seq_to_play为: {self.next_seq_to_play}"
                    )

                await self.player_is_ready.wait()
                self.player_is_ready.clear()
                self.current_playing_id = sentence_id
                self.logger.info(f"▶️ [监控] 开始播放句子: {sentence_id}")
                self.logger.info(
                    f"[PLAYBACK] 接收到首个音频块并开始播放: {sentence_id} "
                    f"(大小: {len(full_audio_data)} samples)"
                )

                asyncio.create_task(
                    self.player.play_full_audio_and_signal_completion(
                        full_audio_data,
                        sentence_id,
                        japanese_text,
                        self.player_is_ready,
                    )
                )

            except Exception as e:
                self.logger.error(f"❌ PlaybackManager 'run' 循环出错: {e}", exc_info=True)
                if not self.player_is_ready.is_set():
                    self.player_is_ready.set()
                await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# StreamPlayerWithBuffer
# ---------------------------------------------------------------------------
class StreamPlayerWithBuffer(StreamPlayer):
    """
    带前置缓冲区 + 字幕集成的播放器。

    字幕/翻译操作通过 SubtitleHooks 依赖注入，
    main.py 在构造时传入：
        player = StreamPlayerWithBuffer(vts_manager, hooks=SubtitleHooks(...))
    """

    def __init__(self, vts_manager, buffer_size: float = 0.3, hooks: SubtitleHooks | None = None):
        super().__init__(vts_manager)
        self.buffer_size = buffer_size
        self.buffer: list = []
        self.buffer_samples = 0
        self.sample_rate = 0
        self.current_sentence = ""
        self.current_translation = ""
        self.subtitle_display_index = 0
        self.subtitle_start_time = 0
        self.logger = logging.getLogger("StreamPlayer")
        self._hooks = hooks or SubtitleHooks()

    def initialize(self, sample_rate: int) -> None:
        super().initialize(sample_rate)
        self.sample_rate = sample_rate
        self.buffer_samples = int(sample_rate * self.buffer_size)

    async def play_full_audio_and_signal_completion(
        self,
        full_audio_data,
        sentence_id: str,
        japanese_text: str,
        completion_event: asyncio.Event,
    ) -> None:
        try:
            sample_rate = 24000
            self.initialize(sample_rate)

            hooks = self._hooks
            if hooks.subtitle_available and hooks.update_subtitle_display:
                hooks.update_subtitle_display(japanese_text, "")
            if hooks.check_and_display_pre_translation:
                asyncio.create_task(
                    hooks.check_and_display_pre_translation(sentence_id, japanese_text)
                )

            def sync_play_and_lipsync(player_instance, _loop):
                player_instance.logger.info(
                    f"👄 [监控] 播放器开始物理播放和口型同步: {sentence_id}"
                )
                player_instance.logger.info(
                    f"[PLAYBACK-PHYSICAL] 开始物理播放和口型同步: {sentence_id}"
                )
                player_instance.vts_manager.send_mouth_data(0.0)
                player_instance.last_send_time = time.time()

                for i in range(0, len(full_audio_data), player_instance.chunk_size):
                    chunk = full_audio_data[i : i + player_instance.chunk_size]
                    if len(chunk) == 0:
                        break
                    player_instance.stream.write(chunk.tobytes())
                    current_time = time.time()
                    if current_time - player_instance.last_send_time >= player_instance.send_interval:
                        rms = np.sqrt(np.mean(chunk ** 2))
                        mouth_value = min(1.0, rms * player_instance.volume_multiplier)
                        player_instance.vts_manager.send_mouth_data(mouth_value)
                        player_instance.last_send_time = current_time

                player_instance.vts_manager.send_mouth_data(0.0)
                player_instance.logger.info(
                    f"✅ [监控] 播放器完成物理播放: {sentence_id}"
                )

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, sync_play_and_lipsync, self, loop)

        except Exception as e:
            self.logger.error(f"❌ 播放器 'play_full_audio' 出错: {e}", exc_info=True)
        finally:
            completion_event.set()
            self.logger.info(
                f"🟢 [监控] 播放器已就绪，为 '{sentence_id}' 设置了完成信号。"
            )

    async def play_audio_chunk_and_signal_completion(
        self,
        audio_chunk,
        sentence_id: str,
        japanese_text: str,
        completion_event: asyncio.Event,
        is_last_chunk: bool = False,
    ) -> None:
        try:
            sample_rate = 24000
            self.initialize(sample_rate)

            hooks = self._hooks
            if hooks.subtitle_available and hooks.update_subtitle_display:
                hooks.update_subtitle_display(japanese_text, "")
            if hooks.check_and_display_pre_translation:
                asyncio.create_task(
                    hooks.check_and_display_pre_translation(sentence_id, japanese_text)
                )

            def sync_play_chunk(player_instance, _loop):
                player_instance.logger.info(
                    f"🎵 [流式播放] 播放音频块: {sentence_id}"
                )
                player_instance.logger.info(
                    f"[PLAYBACK-PHYSICAL] 开始物理播放和口型同步: {sentence_id}"
                )
                player_instance.vts_manager.send_mouth_data(0.0)
                player_instance.last_send_time = time.time()

                for i in range(0, len(audio_chunk), player_instance.chunk_size):
                    chunk = audio_chunk[i : i + player_instance.chunk_size]
                    if len(chunk) == 0:
                        break
                    player_instance.stream.write(chunk.tobytes())
                    current_time = time.time()
                    if current_time - player_instance.last_send_time >= player_instance.send_interval:
                        rms = np.sqrt(np.mean(chunk ** 2))
                        mouth_value = min(1.0, rms * player_instance.volume_multiplier)
                        player_instance.vts_manager.send_mouth_data(mouth_value)
                        player_instance.last_send_time = current_time

                player_instance.vts_manager.send_mouth_data(0.0)
                player_instance.logger.info(
                    f"✅ [流式播放] 音频块播放完成: {sentence_id}"
                )

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, sync_play_chunk, self, loop)

        except Exception as e:
            self.logger.error(f"❌ 流式播放音频块出错: {e}", exc_info=True)
        finally:
            if is_last_chunk:
                audio_duration = len(audio_chunk) / 24000
                await asyncio.sleep(audio_duration + 0.05)
                completion_event.set()
                self.logger.info(
                    f"🟢 [流式播放] 第一句播放完成，为 '{sentence_id}' 设置了完成信号。"
                )
                self.logger.info(
                    f"🔍 [调试] 音频时长: {audio_duration:.3f}s, "
                    f"player_is_ready状态: {completion_event.is_set()}"
                )
                self.logger.info("🎵 [优化策略] 第一句播放完成，第二句可以立即开始播放")
                self.logger.info("🔄 [流式播放] 第一句播放完成，序号更新由PlaybackManager处理")

    def set_subtitle_content(self, japanese_text: str, chinese_text: str) -> None:
        self.current_sentence = japanese_text
        self.current_translation = chinese_text
        self.subtitle_display_index = 0
        self.subtitle_start_time = time.time()
        logger.info(
            f"📝 设置字幕内容 - 日文: '{japanese_text}', 中文: '{chinese_text}'"
        )

    async def _check_and_update_translation(
        self, sentence_id: str, japanese_text: str
    ) -> None:
        hooks = self._hooks
        try:
            normalized = japanese_text.strip()
            translation_data = None
            if hooks.get_translation:
                translation_data = await hooks.get_translation(normalized)

            if translation_data:
                if translation_data["status"] == "completed":
                    chinese_text = translation_data["chinese"]
                    if hooks.display_chinese_subtitle_with_text:
                        await hooks.display_chinese_subtitle_with_text(
                            sentence_id, normalized, chinese_text
                        )
                    logger.info(f"📝 预翻译完成并更新: '{chinese_text[:30]}...'")
                elif translation_data["status"] == "translating":
                    logger.info(f"⏳ 翻译进行中，非阻塞等待: {sentence_id}")
                    asyncio.create_task(
                        self._wait_for_translation(sentence_id, normalized)
                    )
                else:
                    logger.warning(f"⚠️ 翻译失败: {sentence_id}")
            else:
                logger.warning(f"⚠️ 未找到翻译缓存: {sentence_id}")
                if hooks.cache_lock and hooks.cache_ref is not None:
                    async with hooks.cache_lock:
                        for cached_text, cached_data in hooks.cache_ref.items():
                            if cached_text.strip() == normalized or self._is_similar_text(
                                cached_text.strip(), normalized
                            ):
                                logger.info(
                                    f"🔍 找到匹配的缓存: '{cached_text[:30]}...'"
                                )
                                if cached_data["status"] == "completed":
                                    chinese_text = cached_data["chinese"]
                                    if hooks.display_chinese_subtitle_with_text:
                                        await hooks.display_chinese_subtitle_with_text(
                                            sentence_id, normalized, chinese_text
                                        )
                                    logger.info(
                                        f"📝 模糊匹配翻译完成并更新: '{chinese_text[:30]}...'"
                                    )
                                    return
                                break
        except Exception as e:
            logger.error(f"检查翻译失败: {e}")

    def _is_similar_text(self, text1: str, text2: str) -> bool:
        if not text1 or not text2:
            return False
        kana = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
        clean1 = "".join(c for c in text1 if c.isalnum() or c in kana)
        clean2 = "".join(c for c in text2 if c.isalnum() or c in kana)
        if not clean1 or not clean2:
            return False
        if abs(len(clean1) - len(clean2)) / max(len(clean1), len(clean2)) > 0.2:
            return False
        matches = sum(1 for a, b in zip(clean1, clean2) if a == b)
        return matches / max(len(clean1), len(clean2)) > 0.8

    async def _wait_for_translation(
        self, sentence_id: str, japanese_text: str
    ) -> None:
        hooks = self._hooks
        normalized = japanese_text.strip()
        max_wait = 12.0
        interval = 0.1
        waited = 0.0

        while waited < max_wait:
            if hooks.get_translation:
                data = await hooks.get_translation(normalized)
                if data and data["status"] == "completed":
                    chinese_text = data["chinese"]
                    if hooks.display_chinese_subtitle_with_text:
                        await hooks.display_chinese_subtitle_with_text(
                            sentence_id, normalized, chinese_text
                        )
                    logger.info(
                        f"📝 等待翻译完成并更新: '{chinese_text[:30]}...'"
                    )
                    return
                elif data and data["status"] == "failed":
                    logger.warning(f"⚠️ 翻译失败: '{normalized[:30]}...'")
                    return
            await asyncio.sleep(interval)
            waited += interval

        logger.warning(f"⚠️ 翻译超时: '{normalized[:30]}...'")

    def add_to_buffer(
        self,
        audio_chunk,
        subtitle_text: str = None,
        sentence_id: str = None,
    ) -> None:
        if torch.is_tensor(audio_chunk):
            audio_chunk = audio_chunk.cpu().detach().numpy()
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        if subtitle_text:
            self._pending_subtitle_text = subtitle_text
        if sentence_id:
            self._pending_sentence_id = sentence_id

        self.buffer.append(audio_chunk)

        total_samples = sum(len(c) for c in self.buffer)
        if total_samples >= self.buffer_samples:
            self.play_buffer()

    def play_buffer(self) -> None:
        if not self.buffer:
            return
        combined = np.concatenate(self.buffer)
        self.buffer = []

        if len(combined) > 0:
            threshold = 0.008
            non_zero = np.where(np.abs(combined) > threshold)[0]
            if len(non_zero) > 0:
                start_idx = max(0, non_zero[0] - int(0.01 * self.sample_rate))
                combined = combined[start_idx:]

            hooks = self._hooks
            if hasattr(self, "_pending_subtitle_text"):
                pending_sid = getattr(self, "_pending_sentence_id", None)

                if not hasattr(self, "_current_playing_sentence_id"):
                    self._current_playing_sentence_id = None
                current_id = self._current_playing_sentence_id

                allow_update = False
                if current_id is None and pending_sid is not None:
                    self._current_playing_sentence_id = pending_sid
                    allow_update = True
                elif pending_sid is not None and current_id == pending_sid:
                    allow_update = True
                elif pending_sid is not None:
                    logger.info(
                        f"🔁 切换当前播放ID: {current_id} → {pending_sid}"
                    )
                    self._current_playing_sentence_id = pending_sid
                    self._last_switch_time = time.time()
                    self._current_playing_sentence = self._pending_subtitle_text
                    allow_update = True

                if allow_update and hooks.subtitle_available and hooks.update_subtitle_display:
                    hooks.update_subtitle_display(self._pending_subtitle_text, "")
                    logger.info(
                        f"📝 显示字幕: '{self._pending_subtitle_text[:30]}...'"
                    )
                    try:
                        asyncio.create_task(
                            self._check_and_update_translation(
                                pending_sid, self._pending_subtitle_text
                            )
                        )
                    except Exception:
                        pass

                delattr(self, "_pending_subtitle_text")
                if hasattr(self, "_pending_sentence_id"):
                    delattr(self, "_pending_sentence_id")

        super().play_chunk(combined)

    def play_chunk(self, audio_chunk) -> None:
        if not self.is_playing or self.stream is None:
            return
        if torch.is_tensor(audio_chunk):
            audio_chunk = audio_chunk.cpu().detach().numpy()
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        for i in range(0, len(audio_chunk), self.chunk_size):
            if not self.is_playing:
                break
            chunk = audio_chunk[i : i + self.chunk_size]
            if len(chunk) == 0:
                break
            self.stream.write(chunk.tobytes())
            current_time = time.time()
            if current_time - self.last_send_time >= self.send_interval:
                rms = np.sqrt(np.mean(chunk ** 2))
                mouth_value = min(1.0, rms * self.volume_multiplier)
                self.vts_manager.send_mouth_data(mouth_value)
                self.last_send_time = current_time

    def _update_subtitle_display(self, current_time: float) -> None:
        hooks = self._hooks
        if not self.current_sentence or not hooks.subtitle_available:
            return
        elapsed = current_time - self.subtitle_start_time
        length = len(self.current_sentence)
        if length <= 20:
            char_dur = 0.06
        elif length <= 50:
            char_dur = 0.08
        else:
            char_dur = 0.10
        expected = min(int(elapsed / char_dur), length)
        if expected > self.subtitle_display_index:
            display_text = self.current_sentence[:expected]
            display_trans = self.current_translation[:expected] if self.current_translation else ""
            self.subtitle_display_index = expected
            try:
                if hooks.update_subtitle_display:
                    hooks.update_subtitle_display(display_text, display_trans)
            except Exception as e:
                logger.warning(f"字幕更新失败: {e}")
