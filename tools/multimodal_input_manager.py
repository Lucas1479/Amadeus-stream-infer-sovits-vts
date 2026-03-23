import asyncio
import time
import os
from typing import Optional, Tuple, Literal, Dict, Any

import numpy as np
import cv2

try:
    import mss  # for screen capture
except Exception:
    mss = None

try:
    import pyaudio  # for audio capture
except Exception:
    pyaudio = None


InputSource = Literal["webcam", "screen"]
ManagerState = Literal[
    "IDLE",
    "TEXT_INPUT",
    "CONTINUOUS_VISUAL_MONITORING",
    "LISTENING_TO_USER",
    "USER_TURN_ENDED",
    "ANALYZING_CONTEXT",
    "AGENT_SPEAKING",
]


class MultimodalInputManager:
    """
    管理多模态输入（视频/屏幕+音频）的异步管理器：
    - 状态管理（IDLE/TEXT_INPUT/CONTINUOUS_VISUAL_MONITORING）
    - 输入源切换（摄像头/屏幕）
    - 麦克风音频采集（保留系统音频采集占位）
    - 智能触发器（能量VAD / 运动检测 / 场景变化）
    - 触发后打包：关键帧 + 5s 低清视频片段（含音频）
    - 对外通过 asyncio.Queue 输出打包结果
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state: ManagerState = "IDLE"
        self.input_source_type: InputSource = self.config.get("input_source", "webcam")

        # 视频参数
        self.webcam_index: int = int(self.config.get("webcam_index", 0))
        self.screen_monitor_index: int = int(self.config.get("screen_monitor_index", 1))
        self.target_fps: int = int(self.config.get("target_fps", 3))  # 降低帧率以减少资源消耗
        self.keyframe_resolution: Tuple[int, int] = tuple(self.config.get("keyframe_resolution", (1280, 720)))
        self.clip_resolution: Tuple[int, int] = tuple(self.config.get("clip_resolution", (640, 360)))
        self.clip_fps: int = int(self.config.get("clip_fps", 15))
        self.clip_seconds: float = float(self.config.get("clip_seconds", 5.0))
        self.attach_video_clip: bool = bool(self.config.get("attach_video_clip", True))  # 默认附带短视频，利于LIVE理解上下文

        # 触发阈值
        self.vad_energy_threshold: float = float(self.config.get("vad_energy_threshold", 300.0))
        self.motion_threshold: float = float(self.config.get("motion_threshold", 25.0))  # MSE or diff metric
        self.scene_hist_threshold: float = float(self.config.get("scene_hist_threshold", 0.6))  # 直方图差异阈值（越大越容易触发）
        self.trigger_cooldown_sec: float = float(self.config.get("trigger_cooldown_sec", 2.0))  # 从10秒缩短到2秒
        self.min_trigger_interval: float = float(self.config.get("min_trigger_interval", 1.0))  # 从6秒缩短到1秒
        # 语音门控（滞回&延时释放）
        self.vad_upper: float = float(self.config.get("vad_upper", self.vad_energy_threshold))
        self.vad_lower: float = float(self.config.get("vad_lower", max(0.0, self.vad_energy_threshold * 0.7)))
        self.vad_hangover_ms: int = int(self.config.get("vad_hangover_ms", 200))

        # 队列
        self.output_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

        # 运行标志
        self._running: bool = False
        self._monitor_task: Optional[asyncio.Task] = None

        # 视频设备/屏幕
        self._cap: Optional[cv2.VideoCapture] = None
        self._sct: Optional[mss.mss] = None if mss is None else mss.mss()
        self._screen_bbox = None
        self._screen_region = None  # dict: {"left":int, "top":int, "width":int, "height":int}

        # 音频
        self._pa: Optional[pyaudio.PyAudio] = None
        self._mic_stream = None
        self.audio_rate: int = int(self.config.get("audio_rate", 16000))
        self.audio_channels: int = int(self.config.get("audio_channels", 1))
        self.audio_frames_per_buffer: int = int(self.config.get("audio_frames_per_buffer", 1024))
        # 麦克风设备索引（None = 系统默认）
        _mic_idx = self.config.get("input_device_index", None)
        self.mic_input_device_index: Optional[int] = int(_mic_idx) if _mic_idx is not None else None
        self._last_audio_energy: float = 0.0

        # 视频触发辅助
        self._prev_gray: Optional[np.ndarray] = None
        self._last_key_hist: Optional[np.ndarray] = None
        self._last_trigger_ts: float = 0.0
        self._in_flight: bool = False
        self._voice_active: bool = False
        self._last_below_ts: float = 0.0

        # 帧环形缓存（语音门控：平时只缓存，不上传）
        from collections import deque
        self.cache_seconds: float = float(self.config.get("cache_seconds", 3.0))
        self.cache_stride: int = int(self.config.get("cache_stride", 2))  # 每隔N帧缓存一帧
        # 估算缓存容量：cache_seconds * (target_fps / stride)
        cache_len = max(1, int(self.cache_seconds * max(1, self.target_fps // max(1, self.cache_stride))))
        self._frame_cache = deque(maxlen=cache_len)

        # 会话级缓存：用户说话期间的帧
        self._session_frames = []  # list of np.ndarray (low-res)
        self._session_start_ts: float = 0.0

        # 外部控制：Agent发声时屏蔽触发
        self._agent_speaking: bool = False

        # 临时文件目录
        self._tmp_dir: str = self.config.get("tmp_dir", os.path.join(os.getcwd(), "TEMP"))
        os.makedirs(self._tmp_dir, exist_ok=True)

    # --------------- 公共API ---------------
    def set_input_source(self, source_type: InputSource):
        if source_type in ("webcam", "screen"):
            self.input_source_type = source_type
            print(f"Input source set to: {source_type}")
        else:
            raise ValueError("Invalid source type. Use 'webcam' or 'screen'.")

    def set_screen_region(self, region: Optional[Dict[str, int]]):
        """设置屏幕截取区域（None 表示全屏）。
        region 需包含 left, top, width, height。
        运行中也可调用，立即生效。
        """
        if region is not None:
            required = ("left", "top", "width", "height")
            if not all(k in region for k in required):
                raise ValueError("region must contain left, top, width, height")
        # 容错：将区域限制在当前监视器边界内，否则回退全屏
        if region is not None and self._screen_bbox is not None:
            try:
                bx = int(self._screen_bbox.get("left", 0))
                by = int(self._screen_bbox.get("top", 0))
                bw = int(self._screen_bbox.get("width", 0)) or int(self._screen_bbox.get("right", 0)) - bx
                bh = int(self._screen_bbox.get("height", 0)) or int(self._screen_bbox.get("bottom", 0)) - by
                l = max(bx, int(region["left"]))
                t = max(by, int(region["top"]))
                r = min(bx + bw, l + max(1, int(region["width"])))
                b = min(by + bh, t + max(1, int(region["height"])))
                if r <= l or b <= t:
                    # 非法，回退全屏
                    self._screen_region = None
                else:
                    self._screen_region = {"left": l, "top": t, "width": r - l, "height": b - t}
            except Exception:
                self._screen_region = None
        else:
            self._screen_region = region

    def set_attach_video_clip(self, enabled: bool):
        self.attach_video_clip = bool(enabled)

    def set_trigger_cooldown(self, seconds: float):
        try:
            self.trigger_cooldown_sec = float(seconds)
        except Exception:
            pass

    # Agent说话期的屏蔽开关（供外部在TTS开始/结束时调用）
    def notify_agent_speaking(self, speaking: bool):
        self._agent_speaking = bool(speaking)
        if speaking:
            self.state = "AGENT_SPEAKING"
        else:
            # 结束发声后回到持续监控态
            if self.state == "AGENT_SPEAKING":
                self.state = "CONTINUOUS_VISUAL_MONITORING"

    async def start_monitoring(self):
        if self._running:
            return
        print(f"Starting continuous monitoring from {self.input_source_type}...")
        self.state = "CONTINUOUS_VISUAL_MONITORING"
        self._running = True
        await self._open_resources()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self):
        print("Stopping monitoring...")
        self._running = False
        self.state = "IDLE"
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        await self._close_resources()

    # --------------- 内部：资源管理 ---------------
    async def _open_resources(self):
        # 视频（潜在阻塞操作放到线程）
        if self.input_source_type == "webcam":
            def _open_cam():
                try:
                    # 优先使用 DSHOW 后端以加快打开速度（Windows）
                    cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_DSHOW)
                except Exception:
                    cap = cv2.VideoCapture(self.webcam_index)
                return cap
            self._cap = await asyncio.to_thread(_open_cam)
            if self._cap is not None:
                try:
                    self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                except Exception:
                    pass
        else:
            if mss is None:
                raise RuntimeError("mss is not available for screen capture")
            self._sct = await asyncio.to_thread(mss.mss)
            monitors = self._sct.monitors
            idx = min(max(1, self.screen_monitor_index), len(monitors) - 1)
            self._screen_bbox = monitors[idx]

        # 音频（麦克风）
        if pyaudio is None:
            print("PyAudio not available; audio capture disabled.")
        else:
            self._pa = pyaudio.PyAudio()
            open_kwargs = dict(
                format=pyaudio.paInt16,
                channels=self.audio_channels,
                rate=self.audio_rate,
                input=True,
                frames_per_buffer=self.audio_frames_per_buffer,
            )
            if self.mic_input_device_index is not None:
                open_kwargs["input_device_index"] = self.mic_input_device_index
            self._mic_stream = self._pa.open(**open_kwargs)

        # 系统音频采集占位（未来实现）
        # TODO: Implement system/loopback audio capture per platform

    async def _close_resources(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    # --------------- 内部：主循环 ---------------
    async def _monitor_loop(self):
        frame_interval = 1.0 / max(1, self.target_fps)
        last_frame_time = 0.0

        try:
            while self._running:
                start_time = time.time()

                # 抓取帧（放到后台线程，避免阻塞UI）
                frame = None
                try:
                    if self.input_source_type == "webcam":
                        if self._cap is not None:
                            ok, f = await asyncio.to_thread(self._cap.read)
                            if ok:
                                frame = f
                    else:
                        if self._sct is not None:
                            bbox = self._screen_region if self._screen_region is not None else self._screen_bbox
                            if bbox is not None:
                                img = await asyncio.to_thread(self._sct.grab, bbox)
                                img = np.array(img)
                                frame = await asyncio.to_thread(cv2.cvtColor, img, cv2.COLOR_BGRA2BGR)
                except KeyboardInterrupt:
                    break
                except Exception:
                    frame = None
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # 更新帧缓存（按步进缓存，并附上简单“有趣度”评分）
                try:
                    if (len(self._frame_cache) == 0) or (len(self._frame_cache) % self.cache_stride == 0):
                        score = 0.0
                        if self._prev_gray is not None:
                            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            score = float(np.mean(cv2.absdiff(g, self._prev_gray)))
                        # 为降低写盘成本，缓存低清尺寸
                        thumb = cv2.resize(frame, self.clip_resolution)
                        self._frame_cache.append({"ts": time.time(), "frame": thumb, "score": score})
                        # 若处于用户说话期，额外记录到本次会话帧
                        if self._voice_active:
                            self._session_frames.append(thumb)
                except Exception:
                    pass

                # 拉取音频块：按需多次读取以匹配采样率，避免低fps下缓冲堆积
                audio_chunk = None
                try:
                    import math
                    reads_per_frame = max(1, int(math.ceil((self.audio_rate / max(1, self.audio_frames_per_buffer)) / max(1, self.target_fps))))
                except Exception:
                    reads_per_frame = 1
                for _ in range(reads_per_frame):
                    ch = self._read_audio_chunk()
                    if ch is not None:
                        audio_chunk = ch

                # 触发检测
                triggered = await self._smart_trigger_check(frame, audio_chunk)
                if triggered:
                    now_ts = time.time()
                    # in-flight保护 + 冷却 + 硬间隔
                    if (not self._in_flight) and ((now_ts - self._last_trigger_ts) >= max(0.0, max(self.trigger_cooldown_sec, self.min_trigger_interval))):
                        self._last_trigger_ts = now_ts
                        try:
                            self._in_flight = True
                            # 从缓存中挑选关键帧（若缓存为空则退化为当前帧）
                            keyframe_src = self._select_cached_keyframe_fallback(frame)
                            await self._capture_and_package_data(keyframe_src)
                            # VAD 触发时（进入 LISTENING_TO_USER）标记截图已完成，
                            # 避免 LISTENING_TO_USER 检查再触发一张重复截图
                            if self.state == "LISTENING_TO_USER":
                                self._screenshot_taken_for_current_turn = True
                        except Exception as e:
                            print(f"Packaging error: {e}")
                        finally:
                            # 适度延迟后释放占用，避免抖动
                            await asyncio.sleep(0.2)
                            self._in_flight = False

                # 关键改进：用户开始说话时立即截屏（而不是结束时）
                # 这样图片能在说话期间发送完成，不延迟响应
                if self.state == "LISTENING_TO_USER" and not self._in_flight and self._running:
                    if not hasattr(self, '_screenshot_taken_for_current_turn'):
                        self._screenshot_taken_for_current_turn = False
                    
                    if not self._screenshot_taken_for_current_turn:
                        try:
                            self._in_flight = True
                            self._screenshot_taken_for_current_turn = True
                            keyframe_src = self._select_cached_keyframe_fallback(frame)
                            await self._capture_and_package_data(keyframe_src)
                            print(f"[截屏] 用户开始说话，已截取当前屏幕")
                        except Exception as e:
                            print(f"Screenshot on speech start error: {e}")
                        finally:
                            await asyncio.sleep(0.1)
                            self._in_flight = False

                # 用户回合结束时，只需重置标记
                if self.state == "USER_TURN_ENDED" and not self._in_flight and self._running:
                    # 重置截屏标记，为下次对话准备
                    self._screenshot_taken_for_current_turn = False
                    # 恢复持续监控态
                    self.state = "CONTINUOUS_VISUAL_MONITORING"

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            pass

    # --------------- 内部：读取输入 ---------------
    def _read_frame(self) -> Optional[np.ndarray]:
        # 保留同步版本（仅备用；主循环已改为异步线程抓取）
        try:
            if self.input_source_type == "webcam":
                if self._cap is None:
                    return None
                ok, frame = self._cap.read()
                return frame if ok else None
            else:
                if self._sct is None or self._screen_bbox is None:
                    return None
                bbox = self._screen_region if self._screen_region is not None else self._screen_bbox
                img = np.array(self._sct.grab(bbox))
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

    def _read_audio_chunk(self) -> Optional[np.ndarray]:
        if self._mic_stream is None:
            return None
        try:
            # 放到后台线程读取音频，避免阻塞
            data = None
            try:
                data = asyncio.get_running_loop()
                data = None  # 仅用于检测是否在事件循环中
            except RuntimeError:
                pass
            if True:
                data = self._mic_stream.read(self.audio_frames_per_buffer, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            return audio
        except Exception:
            return None

    # --------------- 内部：触发逻辑 ---------------
    async def _smart_trigger_check(self, frame: np.ndarray, audio_chunk: Optional[np.ndarray]) -> bool:
        # 1) VAD: 简单能量阈值
        vad_trigger = False
        if audio_chunk is not None and audio_chunk.size > 0:
            energy = float(np.mean(np.abs(audio_chunk)))
            self._last_audio_energy = energy
            # 滞回：仅在从静音->有声的上升沿触发
            now = time.time()
            if self._agent_speaking:
                # Agent说话期间不触发用户回合
                pass
            elif not self._voice_active:
                if energy >= self.vad_upper:
                    self._voice_active = True
                    self._session_start_ts = now
                    self._session_frames = []
                    self.state = "LISTENING_TO_USER"
                    vad_trigger = True  # 上升沿触发一次
                    print(f"[VAD] 🎤 检测到用户说话 (能量: {energy:.1f})", flush=True)
            else:
                if energy < self.vad_lower:
                    if self._last_below_ts == 0.0:
                        self._last_below_ts = now
                    elif (now - self._last_below_ts) * 1000.0 >= self.vad_hangover_ms:
                        self._voice_active = False
                        self._last_below_ts = 0.0
                        # 用户回合结束
                        if self.state == "LISTENING_TO_USER":
                            self.state = "USER_TURN_ENDED"
                            print(f"[VAD] 🔇 用户停止说话 (静音持续 {self.vad_hangover_ms}ms)", flush=True)
                else:
                    self._last_below_ts = 0.0

        # 2) 运动检测：与上一帧灰度比较
        motion_trigger = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            mean_diff = float(np.mean(diff))
            if mean_diff >= self.motion_threshold:
                motion_trigger = True
        self._prev_gray = gray

        # 3) 场景变化：颜色直方图比较
        scene_trigger = False
        hist = self._compute_color_hist(frame)
        if self._last_key_hist is not None and hist is not None:
            # use correlation distance: 1 - corr
            corr = cv2.compareHist(self._last_key_hist, hist, cv2.HISTCMP_CORREL)
            distance = 1.0 - float(corr)
            if distance >= self.scene_hist_threshold:
                scene_trigger = True

        return vad_trigger or motion_trigger or scene_trigger

    def _compute_color_hist(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist
        except Exception:
            return None

    # --------------- 内部：打包流程 ---------------
    async def _capture_and_package_data(self, trigger_frame: np.ndarray):
        # 关键帧（高分辨率）
        keyframe = cv2.resize(trigger_frame, self.keyframe_resolution)
        self._last_key_hist = self._compute_color_hist(keyframe)

        # 按需录制低清视频片段（LIVE模式默认关闭，仅关键帧）
        clip_path = None
        if self.attach_video_clip:
            # 使用带预卷的短视频，提高LIVE对上下文的理解
            clip_path = await self._record_lowres_clip_with_preroll(duration=self.clip_seconds)

        packaged = self._package_for_llm(keyframe, clip_path)
        await self.output_queue.put(packaged)

    async def _record_lowres_clip(self, duration: float) -> str:
        width, height = self.clip_resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(self._tmp_dir, f"clip_{int(time.time()*1000)}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, self.clip_fps, (width, height))

        # 简单同步录制：在duration内循环采集帧，并将音频缓存到内存
        audio_buffers = []
        frame_interval = 1.0 / max(1, self.clip_fps)
        start_t = time.time()
        while time.time() - start_t < duration and self._running:
            frame = self._read_frame()
            if frame is None:
                await asyncio.sleep(0.005)
                continue
            frame_small = cv2.resize(frame, (width, height))
            writer.write(frame_small)

            audio_chunk = self._read_audio_chunk()
            if audio_chunk is not None:
                audio_buffers.append(audio_chunk.copy())

            await asyncio.sleep(frame_interval)

        writer.release()

        # 音频封装到视频：当前简化实现不将音频复用进容器，仅保留视频并返回路径。
        # 若需将音频与视频合并，可在此处用 ffmpeg 合并（占位）：
        # TODO: mux audio into mp4 via ffmpeg if needed
        # 同时可将原始PCM写入旁路文件供上层使用
        if audio_buffers:
            pcm_path = out_path.replace(".mp4", ".pcm16")
            pcm = np.concatenate(audio_buffers).astype(np.int16).tobytes()
            with open(pcm_path, "wb") as f:
                f.write(pcm)

        return out_path

    def _select_cached_keyframe_fallback(self, latest_frame: np.ndarray) -> np.ndarray:
        """选择最新的帧作为关键帧（确保截取当前屏幕，而不是历史"有趣"帧）"""
        # 关键修复：直接使用最新帧，不要从缓存中选"评分最高"的旧帧
        # 这样才能确保LLM看到的是当前屏幕，而不是几秒前的画面
        return cv2.resize(latest_frame, self.keyframe_resolution)

    async def _record_lowres_clip_with_preroll(self, duration: float) -> str:
        """写入缓存帧作为pre-roll，再补采后续，形成带预卷的短视频。"""
        width, height = self.clip_resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(self._tmp_dir, f"clip_{int(time.time()*1000)}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, self.clip_fps, (width, height))
        # 先写缓存（时间顺序）
        try:
            cached = list(self._frame_cache)
            cached.sort(key=lambda x: x["ts"])
            # 写入最多 cache_seconds 对应的帧数
            max_cached = int(self.clip_fps * min(self.cache_seconds, duration))
            for it in cached[-max_cached:]:
                fr = it["frame"]
                if fr.shape[1] != width or fr.shape[0] != height:
                    fr = cv2.resize(fr, (width, height))
                writer.write(fr)
        except Exception:
            pass
        # 再补采剩余时长
        remain = max(0.0, duration - min(self.cache_seconds, duration))
        start_t = time.time()
        while time.time() - start_t < remain and self._running:
            frame = None
            try:
                if self.input_source_type == "webcam" and self._cap is not None:
                    ok, f = await asyncio.to_thread(self._cap.read)
                    if ok:
                        frame = cv2.resize(f, (width, height))
                elif self.input_source_type == "screen" and self._sct is not None:
                    bbox = self._screen_region if self._screen_region is not None else self._screen_bbox
                    if bbox is not None:
                        img = await asyncio.to_thread(self._sct.grab, bbox)
                        img = np.array(img)
                        frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), (width, height))
            except Exception:
                frame = None
            if frame is not None:
                writer.write(frame)
            await asyncio.sleep(1.0 / max(1, self.clip_fps))
        writer.release()
        return out_path

    def _package_for_llm(self, keyframe: np.ndarray, video_clip_path: Optional[str]) -> Dict[str, Any]:
        # 这里预留：将keyframe编码为Base64、将视频转储为可上传资源，最终组装为目标LLM API需要的JSON结构
        packaged_data: Dict[str, Any] = {
            "mode": "CONTINUOUS_VISUAL_MONITORING",
            "keyframe": keyframe,  # numpy array 占位
            "video_path": video_clip_path,
            "has_video": bool(video_clip_path),
            # "keyframe_b64": ...,  # 占位
            # "json_payload": ...,  # 占位
            "audio_energy": self._last_audio_energy,
            "timestamp": time.time(),
        }
        return packaged_data


