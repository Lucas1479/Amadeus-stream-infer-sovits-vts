"""
ASR 管理模块（Google Speech Recognition）
- ASRManager：麦克风初始化、自动设备选择、语音识别
"""
from __future__ import annotations

import logging
import traceback

try:
    import pyaudio
except ImportError as exc:
    pyaudio = None
    _PYAUDIO_IMPORT_ERROR = exc
else:
    _PYAUDIO_IMPORT_ERROR = None

try:
    import speech_recognition as speech_rec
except ImportError as exc:
    speech_rec = None
    _SPEECH_RECOGNITION_IMPORT_ERROR = exc
else:
    _SPEECH_RECOGNITION_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _ensure_asr_dependencies() -> None:
    # 这里不在模块导入阶段直接抛错，是为了让主程序可以先启动并给出
    # 更友好的报错；真正点击麦克风或初始化 ASR 时再提示缺什么。
    if pyaudio is None:
        raise RuntimeError(
            "PyAudio 未安装，无法启用麦克风输入。macOS 需要先安装 portaudio，再重新安装 pyaudio。"
        ) from _PYAUDIO_IMPORT_ERROR
    if speech_rec is None:
        raise RuntimeError(
            "SpeechRecognition 未安装，无法启用麦克风输入。"
        ) from _SPEECH_RECOGNITION_IMPORT_ERROR


class ASRManager:
    # 可在此处强制指定麦克风设备索引（None = 自动选择）
    MICROPHONE_DEVICE_INDEX = 2

    def __init__(self):
        _ensure_asr_dependencies()
        self.logger = logging.getLogger("asr_manager")
        self._init_google_asr()

    def _pick_microphone_index(self):
        """
        用 pyaudio 列举所有输入设备并快速采样测量 RMS，
        选择电平最高的设备（最可能是真实麦克风）。
        """
        if ASRManager.MICROPHONE_DEVICE_INDEX is not None:
            print(f"[ASR] 使用手动指定的麦克风索引: {ASRManager.MICROPHONE_DEVICE_INDEX}")
            return ASRManager.MICROPHONE_DEVICE_INDEX

        pa = pyaudio.PyAudio()
        device_count = pa.get_device_count()
        names = speech_rec.Microphone.list_microphone_names()

        print(f"[ASR] ===== 可用音频输入设备 ({device_count} 个) =====")
        best_index = None
        best_rms = -1

        for i in range(device_count):
            try:
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) <= 0:
                    continue
                dev_name = names[i] if i < len(names) else info.get("name", f"Device {i}")

                try:
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=int(info.get("defaultSampleRate", 16000)),
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024,
                    )
                    import audioop
                    frames = b""
                    for _ in range(4):
                        frames += stream.read(1024, exception_on_overflow=False)
                    stream.stop_stream()
                    stream.close()
                    rms = audioop.rms(frames, 2)
                except Exception:
                    rms = -1

                print(f"[ASR]   [{i}] {dev_name}  RMS={rms}")
                if rms > best_rms:
                    best_rms = rms
                    best_index = i
            except Exception:
                pass

        pa.terminate()

        if best_index is not None:
            name = names[best_index] if best_index < len(names) else best_index
            print(f"[ASR] 自动选择 RMS 最高的设备 [{best_index}]: {name}  (RMS={best_rms})")
        else:
            print("[ASR] 未找到可用输入设备，将使用系统默认")
        return best_index

    def _init_google_asr(self):
        """初始化 Google Speech Recognition。"""
        self.recognizer = speech_rec.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = 30

        mic_index = self._pick_microphone_index()

        for idx in ([mic_index] if mic_index is not None else []) + [None]:
            try:
                mic = speech_rec.Microphone(device_index=idx)
                with mic as source:
                    if source.stream is None:
                        raise OSError("stream is None after __enter__")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.microphone = mic
                label = f"设备[{idx}]" if idx is not None else "系统默认"
                self.logger.info(f"🎤 麦克风初始化成功: {label}")
                break
            except Exception as e:
                label = f"设备[{idx}]" if idx is not None else "系统默认"
                self.logger.warning(f"🎤 麦克风 {label} 打开失败: {e}，尝试下一个…")
        else:
            raise RuntimeError("所有麦克风设备均无法打开，ASR初始化失败")

        self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, 200)
        self.logger.info(f"🎤 初始能量阈值: {self.recognizer.energy_threshold:.0f}")
        self.language = "zh-CN"
        self.logger.info("Google Speech Recognition初始化成功")

    def listen_for_speech(self, max_retries: int = 2) -> str | None:
        """监听语音并返回识别文本，失败则返回 None。"""
        self.logger.info("Listening for speech... (使用: Google ASR)")

        for attempt in range(max_retries + 1):
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, 200)
                    self.recognizer.energy_threshold = min(self.recognizer.energy_threshold, 1200)
                    self.logger.info(f"🎤 能量阈值: {self.recognizer.energy_threshold:.0f}，开始监听...")
                    audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=30)

                self.logger.info(f"Processing speech with Google ASR... (attempt {attempt + 1})")
                text = self.recognizer.recognize_google(
                    audio,
                    language=self.language,
                    show_all=False,
                    with_confidence=False,
                )

                if text and len(text.strip()) > 0:
                    self.logger.info(f"Recognized: {text}")
                    return text.strip()
                else:
                    self.logger.warning(f"Empty recognition result (attempt {attempt + 1})")

            except speech_rec.WaitTimeoutError:
                self.logger.warning(f"No speech detected within timeout period (attempt {attempt + 1})")
                if attempt < max_retries:
                    continue
                return None
            except speech_rec.UnknownValueError:
                self.logger.warning(f"Could not understand audio (attempt {attempt + 1})")
                if attempt < max_retries:
                    continue
                return None
            except speech_rec.RequestError as e:
                self.logger.error(f"Could not request results; {e} (attempt {attempt + 1})")
                if attempt < max_retries:
                    continue
                return None
            except Exception as e:
                self.logger.error(f"Error in speech recognition: {e} (attempt {attempt + 1})")
                self.logger.error(traceback.format_exc())
                if attempt < max_retries:
                    continue
                return None

        self.logger.error("All Google ASR attempts failed")
        return None

    def set_language(self, language_code: str) -> None:
        self.language = language_code
        self.logger.info(f"ASR language set to: {language_code}")

    def set_microphone_index(self, index: int | None) -> None:
        """手动切换麦克风设备索引，传 None 恢复自动选择。"""
        _ensure_asr_dependencies()
        ASRManager.MICROPHONE_DEVICE_INDEX = index
        names = speech_rec.Microphone.list_microphone_names()
        name = names[index] if index is not None and index < len(names) else "系统默认"
        self.microphone = speech_rec.Microphone(device_index=index)
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, 200)
        self.logger.info(f"🎤 切换至麦克风 [{index}]: {name}，阈值: {self.recognizer.energy_threshold:.0f}")
