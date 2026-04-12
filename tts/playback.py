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
        # 句子开始播放回调：fn(sentence_id: str) -> None
        self.on_sentence_start: Callable[[str], None] | None = None
        # 轮次最后一句播完回调：fn() -> None
        self.on_turn_playback_complete: Callable[[], None] | None = None
        self._turn_last_sentence_id: str | None = None

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

            if is_first_chunk and self.on_sentence_start is not None:
                try:
                    self.on_sentence_start(sentence_id)
                except Exception as _cb_err:
                    self.logger.warning(f"on_sentence_start 回调异常: {_cb_err}")

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

    def mark_turn_last_sentence(self, sentence_id: str) -> None:
        """标记本轮最后一句 ID；该句播完后触发 on_turn_playback_complete。"""
        self._turn_last_sentence_id = sentence_id

    async def play_s1_stream(
        self,
        chunk_queue: asyncio.Queue,
        sentence_id: str,
        japanese_text: str,
    ) -> None:
        """句1专用流式播放：从 chunk_queue 串行消费 chunk 并立即播放。

        chunk_queue 协议：
          - 元素为 np.ndarray（float32 音频数据）
          - put(None) 表示 EOF，收到后停止播放并置位 player_is_ready

        设计要点：
          - 每个 chunk 用 run_in_executor 同步写 pyaudio，await 保证串行，
            不存在多线程同时写同一 stream 的问题
          - player_is_ready 仅在 finally 中置位一次，与 PlaybackManager.run()
            的句2+ 路径契约完全相同
          - next_seq_to_play 在此处更新为 2 并通知 play_condition，确保
            PlaybackManager.run() 能在 pending_audio 里等到正确序号
        """
        # ── 等前一轮播完，占用播放器 ─────────────────────────────────────
        await self.player_is_ready.wait()
        self.player_is_ready.clear()
        self.current_playing_id = sentence_id
        self.logger.info(f"🎵 [S1流式] 开始流式播放: {sentence_id}")

        # ── 触发 on_sentence_start 回调（表情切换等） ────────────────────
        if self.on_sentence_start is not None:
            try:
                self.on_sentence_start(sentence_id)
            except Exception as _e:
                self.logger.warning(f"on_sentence_start 回调异常: {_e}")

        # ── 通知 PlaybackManager.run() 下一个期望序号为 2 ─────────────────
        self.next_seq_to_play = 2
        async with self.play_condition:
            self.play_condition.notify()
        self.logger.info("🔄 [S1流式] next_seq_to_play 已更新为 2")

        # ── 字幕 / 预翻译 ──────────────────────────────────────────────────
        hooks = self.player._hooks
        if hooks.subtitle_available and hooks.update_subtitle_display:
            hooks.update_subtitle_display(japanese_text, "")
        if hooks.check_and_display_pre_translation:
            asyncio.create_task(
                hooks.check_and_display_pre_translation(sentence_id, japanese_text)
            )

        # ── 初始化 pyaudio stream ──────────────────────────────────────────
        sample_rate = 24000
        self.player.initialize(sample_rate)
        player = self.player
        player.vts_manager.send_mouth_data(0.0)
        player.last_send_time = time.time()
        loop = asyncio.get_running_loop()
        first_chunk_played = False

        # ── 串行播放所有 chunk ─────────────────────────────────────────────
        try:
            while True:
                audio_chunk = await chunk_queue.get()
                if audio_chunk is None:
                    # EOF 哨兵：所有 chunk 已发送完毕
                    break

                # 确保 float32
                if audio_chunk.dtype != np.float32:
                    audio_chunk = audio_chunk.astype(np.float32)

                if not first_chunk_played:
                    first_chunk_played = True
                    self.logger.info(
                        f"[PLAYBACK-S1流式] 首个chunk开始物理播放: {sentence_id} "
                        f"({len(audio_chunk)} samples)"
                    )

                # 捕获当前 chunk 的引用（避免闭包捕获循环变量）
                _chunk = audio_chunk

                def _sync_play(chunk=_chunk):
                    for i in range(0, len(chunk), player.chunk_size):
                        c = chunk[i : i + player.chunk_size]
                        if not len(c):
                            break
                        player.stream.write(c.tobytes())
                        t = time.time()
                        if t - player.last_send_time >= player.send_interval:
                            rms = float(np.sqrt(np.mean(c ** 2)))
                            player.vts_manager.send_mouth_data(
                                min(1.0, rms * player.volume_multiplier)
                            )
                            player.last_send_time = t

                # await 保证当前 chunk 物理写完才取下一个（串行）
                await loop.run_in_executor(None, _sync_play)
                self.logger.debug(
                    f"[S1流式] chunk 播放完成: {len(audio_chunk)} samples"
                )

            player.vts_manager.send_mouth_data(0.0)
            self.logger.info(f"✅ [S1流式] 全部音频播放完毕: {sentence_id}")

        except Exception as e:
            self.logger.error(f"❌ [S1流式] 播放出错: {e}", exc_info=True)

        finally:
            player.vts_manager.send_mouth_data(0.0)
            # 留少量时间让声卡硬件缓冲排空（stream.write 已阻塞，只差最后一帧）
            await asyncio.sleep(0.08)

            # ── 本轮最后一句回调 ──────────────────────────────────────────
            if sentence_id == self._turn_last_sentence_id:
                self._turn_last_sentence_id = None
                if self.on_turn_playback_complete is not None:
                    try:
                        self.on_turn_playback_complete()
                    except Exception as _e:
                        self.logger.warning(f"on_turn_playback_complete 回调异常: {_e}")

            self.player_is_ready.set()
            self.logger.info(f"🟢 [S1流式] player_is_ready 已置位: {sentence_id}")

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
                # 检查刚完成的句子是否是本轮最后一句
                _just_finished = self.current_playing_id
                if _just_finished and _just_finished == self._turn_last_sentence_id:
                    self._turn_last_sentence_id = None
                    if self.on_turn_playback_complete is not None:
                        try:
                            self.on_turn_playback_complete()
                        except Exception as _cb_err:
                            self.logger.warning(f"on_turn_playback_complete 回调异常: {_cb_err}")
                self.player_is_ready.clear()
                self.current_playing_id = sentence_id
                self.logger.info(f"▶️ [监控] 开始播放句子: {sentence_id}")
                self.logger.info(
                    f"[PLAYBACK] 接收到首个音频块并开始播放: {sentence_id} "
                    f"(大小: {len(full_audio_data)} samples)"
                )

                if self.on_sentence_start is not None:
                    try:
                        self.on_sentence_start(sentence_id)
                    except Exception as _cb_err:
                        self.logger.warning(f"on_sentence_start 回调异常: {_cb_err}")

                asyncio.create_task(
                    self.player.play_full_audio_and_signal_completion(
                        full_audio_data,
                        sentence_id,
                        japanese_text,
                        self.player_is_ready,
                    )
                )

                # 若当前句是本轮最后一句，创建独立监听任务：
                # 主循环在最后一句结束后会卡在 play_condition.wait() 等新音频，
                # 导致 on_turn_playback_complete 永远不被触发。
                # 这里额外用一个 task 等待 player_is_ready，确保回调一定触发。
                if sentence_id == self._turn_last_sentence_id:
                    _watched_id = sentence_id

                    async def _fire_on_last_done(watched=_watched_id):
                        await self.player_is_ready.wait()
                        if self._turn_last_sentence_id != watched:
                            return  # 主循环已先触发，避免重复
                        self._turn_last_sentence_id = None
                        if self.on_turn_playback_complete is not None:
                            try:
                                self.on_turn_playback_complete()
                            except Exception as _cb_err:
                                self.logger.warning(
                                    f"on_turn_playback_complete(watcher) 回调异常: {_cb_err}"
                                )

                    asyncio.create_task(_fire_on_last_done())

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
                # stream.write() 是阻塞调用，run_in_executor 返回时音频数据已全部
                # 送入 OS 声卡驱动缓冲区，只需预留极短的硬件缓冲余量即可。
                # 原来的 asyncio.sleep(audio_duration + 0.05) 会在物理播放结束后
                # 再额外等待整个句子的时长，造成句间静音间隙，已删除。
                await asyncio.sleep(0.08)
                completion_event.set()
                audio_duration = len(audio_chunk) / 24000
                self.logger.info(
                    f"🟢 [流式播放] 第一句播放完成，为 '{sentence_id}' 设置了完成信号。"
                    f"（音频时长 {audio_duration:.3f}s）"
                )

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
