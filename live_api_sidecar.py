import os
import sys
import json
import asyncio
import argparse
import base64
import io

# Sidecar 仅负责 LIVE（TEXT 输出）。在 .venv_live 环境中运行。

try:
    from google import genai
    from google.genai import types
except Exception as e:
    print(json.dumps({"type": "error", "data": f"IMPORT_ERROR: {e}"}), flush=True)
    sys.exit(1)

# 可选：仅在需要视频时导入
try:
    import cv2
except Exception:
    cv2 = None
try:
    import mss
except Exception:
    mss = None

try:
    import pyaudio
except Exception:
    pyaudio = None


def print_json(obj):
    try:
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    except Exception as e:
        # 最终兜底
        sys.stdout.write(json.dumps({"type": "error", "data": f"PRINT_FAIL: {e}"}) + "\n")
        sys.stdout.flush()


async def read_stdin_commands(session):
    # 使用线程方式读取，避免 Windows Proactor 下的管道问题
    while True:
        try:
            # 用二进制读取并按UTF-8解码，避免 Windows 控制台乱码
            line = await asyncio.to_thread(sys.stdin.buffer.readline)
        except Exception as e:
            print_json({"type": "error", "data": f"STDIN_READ_FAIL: {e}"})
            await asyncio.sleep(0.1)
            continue
        if not line:
            await asyncio.sleep(0.1)
            continue
        try:
            # 优先UTF-8，失败再回退到GBK，最后再用replace
            try:
                text = line.decode('utf-8', errors='strict').strip()
            except UnicodeDecodeError:
                try:
                    text = line.decode('gbk', errors='strict').strip()
                except Exception:
                    text = line.decode('utf-8', errors='replace').strip()
            if not text:
                continue
            cmd = json.loads(text)
            ctype = cmd.get('type')
            data = cmd.get('data')
            if ctype == 'text' and data:
                try:
                    # 为确保模型立即生成文本，这里结束一回合
                    await session.send(input=str(data), end_of_turn=True)
                    print_json({"type": "status", "data": f"TEXT_SENT:{str(data)[:40]}..."})
                except Exception as e:
                    print_json({"type": "error", "data": f"SEND_TEXT_FAIL: {e}"})
            elif ctype == 'persona' and data:
                try:
                    # persona 注入：不结束回合，模型等待用户输入后才响应
                    await session.send(input=str(data), end_of_turn=False)
                    print_json({"type": "status", "data": f"PERSONA_SENT:{str(data)[:40]}..."})
                except Exception as e:
                    print_json({"type": "error", "data": f"SEND_PERSONA_FAIL: {e}"})
            elif ctype == 'mic_on':
                state['mic_enabled'] = True
                print_json({"type": "status", "data": "MIC_ON"})
            elif ctype == 'mic_off':
                state['mic_enabled'] = False
                print_json({"type": "status", "data": "MIC_OFF"})
            elif ctype == 'turn_end':
                # 客户端 VAD 检测到用户停止说话：
                # 发送一段短暂静音 + end_of_turn=True，显式通知模型"用户已说完，请回复"。
                # 这绕过了服务端 VAD（server-side VAD），改用客户端 VAD 触发响应。
                try:
                    silence = bytes(3200)  # 100ms 静音 @16kHz 16-bit PCM
                    await session.send(
                        input={"data": silence, "mime_type": "audio/pcm;rate=16000"},
                        end_of_turn=True,
                    )
                    print_json({"type": "status", "data": "TURN_END_SENT"})
                except Exception as e:
                    print_json({"type": "error", "data": f"TURN_END_FAIL: {e}"})
            elif ctype == 'visual':
                try:
                    # 仅关键帧：data: {keyframe_b64, keyframe_mime}
                    if isinstance(data, dict):
                        k_b64 = data.get('keyframe_b64')
                        k_mime = data.get('keyframe_mime') or 'image/jpeg'
                    else:
                        k_b64, k_mime = None, None
                    if k_b64:
                        k_bytes = base64.b64decode(k_b64)
                        # 直接发送新图片，不附带文本提示（每帧插文本会快速累积 context）
                        await session.send(input={"data": k_bytes, "mime_type": k_mime}, end_of_turn=False)
                        state['has_visual_context'] = True
                        print_json({"type": "status", "data": "VISUAL_SENT"})
                    else:
                        print_json({"type": "error", "data": "VISUAL_EMPTY_INPUTS"})
                except Exception as e:
                    print_json({"type": "error", "data": f"VISUAL_SEND_FAIL: {e}"})
            else:
                print_json({"type": "status", "data": f"CMD_RECEIVED: {ctype}"})
        except Exception as e:
            print_json({"type": "error", "data": f"BAD_CMD: {e}"})


async def receive_text_loop(session):
    """持续接收服务端消息（多轮对话：每轮 turn_complete 后重新调用 receive()）。

    session.receive() 是单轮迭代器：在收到 turn_complete 时停止。
    外层 while True 负责在每轮结束后立即重建迭代器，等待下一轮模型响应。
    """
    print_json({"type": "status", "data": "RECV_LOOP_STARTED"})
    try:
        while True:
            try:
                async for resp in session.receive():
                    # 方式1: 标准 TEXT modality
                    text = getattr(resp, 'text', None)
                    if text:
                        print_json({"type": "text", "data": text})
                        continue

                    # 方式2: native audio 模型输出转录（output_transcription）
                    try:
                        t = resp.server_content.output_transcription.text
                        if t:
                            print_json({"type": "text", "data": t})
                            continue
                    except Exception:
                        pass

                    # 方式3: model_turn.parts 里的文本（跳过思维链 thought=True）
                    try:
                        parts = resp.server_content.model_turn.parts
                        for part in (parts or []):
                            if getattr(part, 'thought', False):
                                continue
                            t = getattr(part, 'text', None)
                            if t:
                                print_json({"type": "text", "data": t})
                    except Exception:
                        pass

                    # turn_complete：模型本轮说完，立即通知主进程 flush
                    try:
                        if resp.server_content.turn_complete:
                            print_json({"type": "turn_complete"})
                    except Exception:
                        pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print_json({"type": "error", "data": f"RECV_TURN_FAIL: {e}"})
                await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print_json({"type": "error", "data": f"RECV_FAIL: {e}"})


def encode_frame_to_part(frame):
    # frame: BGR (cv2)
    import numpy as np
    import cv2 as _cv
    ok, buf = _cv.imencode('.jpg', frame)
    if not ok:
        return None
    jpg_bytes = buf.tobytes()
    return types.Part.from_data(jpg_bytes, mime_type='image/jpeg')


async def run(mode: str, model_name: str, api_key: str):
    if not api_key:
        print_json({"type": "error", "data": "NO_API_KEY"})
        return

    client = genai.Client(http_options={"api_version": "v1beta"}, api_key=api_key)
    system_instruction = os.environ.get("LIVE_SYSTEM_INSTRUCTION", "").strip()
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=system_instruction if system_instruction else None,
        # 禁用思维链：减少首次响应延迟（仅 2.5 系列支持）
        generation_config=types.GenerationConfig(
            thinking_config=types.GenerationConfigThinkingConfig(thinking_budget=0)
        ),
        # 滑动窗口压缩：超过 16k tokens 时自动压缩到 8k，防止多轮对话越来越慢
        context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=8000,
                sliding_window=types.SlidingWindow(target_tokens=4000),
        ),
    )

    # 全局状态（简单字典）
    global state
    state = {
        "mic_enabled": False,
        "has_visual_context": False,  # 跟踪是否已有视觉上下文
    }

    async def mic_loop(session):
        if pyaudio is None:
            print_json({"type": "status", "data": "PYAudio_NOT_AVAILABLE"})
            return
        p = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        # 从环境变量读取麦克风设备索引（与主进程 ASR 使用同一设备）
        mic_device_index = None
        _mic_env = os.environ.get("LIVE_MIC_DEVICE", "").strip()
        if _mic_env.isdigit():
            mic_device_index = int(_mic_env)
        print_json({"type": "status", "data": f"MIC_DEVICE: {mic_device_index}"})
        try:
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            input_device_index=mic_device_index, frames_per_buffer=CHUNK)
        except Exception as e:
            print_json({"type": "error", "data": f"MIC_OPEN_FAIL: {e}"})
            return
        try:
            while True:
                # 未开启时静默（不读取/不发送，避免阻塞）
                if not state.get('mic_enabled', False):
                    await asyncio.sleep(0.02)
                    continue
                try:
                    data = await asyncio.to_thread(stream.read, CHUNK, exception_on_overflow=False)
                except Exception as e:
                    print_json({"type": "error", "data": f"MIC_READ_FAIL: {e}"})
                    await asyncio.sleep(0.02)
                    continue
                try:
                    # 必须指定采样率，否则服务端 VAD 无法正确解析音频
                    await session.send(input={"data": data, "mime_type": "audio/pcm;rate=16000"})  # 不 end_of_turn，持续聆听
                except Exception as e:
                    print_json({"type": "error", "data": f"MIC_SEND_FAIL: {e}"})
                    # 发送失败：关麦等待，由主进程 VAD 重新触发 mic_on，不在这里自动恢复
                    state['mic_enabled'] = False
                    print_json({"type": "status", "data": "MIC_AUTO_OFF_ON_FAIL"})
                    await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                stream.stop_stream(); stream.close(); p.terminate()
            except Exception:
                pass

    try:
        print_json({"type": "status", "data": f"CONNECTING:{model_name}"})
        async with client.aio.live.connect(model=model_name, config=config) as session:
            print_json({"type": "status", "data": "LIVE_CONNECTED"})

            async def stdin_task():
                await read_stdin_commands(session)

            recv_task = asyncio.create_task(receive_text_loop(session))
            cmd_task = asyncio.create_task(stdin_task())
            mic_task = asyncio.create_task(mic_loop(session))

            # 这里不主动推音频/视频，由主进程按需决定；保留关键帧上传接口占位
            await asyncio.gather(recv_task, cmd_task)
            mic_task.cancel()
            print_json({"type": "status", "data": "SESSION_GATHER_DONE"})

    except KeyboardInterrupt:
        print_json({"type": "status", "data": "INTERRUPTED"})
    except Exception as e:
        print_json({"type": "error", "data": f"LIVE_FAIL: {e}"})


def _read_key_from_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return (f.read() or '').strip()
    except Exception:
        return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="screen", choices=["screen", "camera", "none"])  # 预留
    parser.add_argument("--model", type=str, default="models/gemini-live-2.5-flash-preview")
    parser.add_argument("--api_key", type=str, default="", help="override API key (optional)")
    args = parser.parse_args()

    # 解析 API Key 优先级：--api_key > 环境变量 > 本地文件 GEMINI_API_KEY.txt / LIVE_API_KEY.txt
    api_key = (
        args.api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("LIVE_API_KEY")
        or _read_key_from_file(os.path.join(os.getcwd(), "GEMINI_API_KEY.txt"))
        or _read_key_from_file(os.path.join(os.getcwd(), "LIVE_API_KEY.txt"))
        or ""
    )
    if not api_key:
        print_json({"type": "error", "data": "NO_API_KEY"})
        return

    try:
        asyncio.run(run(args.mode, args.model, api_key))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


