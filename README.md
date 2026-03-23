<div align="center">

# amadeus-realtime

**具身实时 AI Agent · Embodied Real-time AI Agent**

[![中文](https://img.shields.io/badge/中文-README-blue)](#中文版) &nbsp;|&nbsp; [![English](https://img.shields.io/badge/English-README-blue)](#english)

![Python](https://img.shields.io/badge/Python-3.10+-blue) &nbsp;
![GPT-SoVITS](https://img.shields.io/badge/Base-GPT--SoVITS_v3-orange) &nbsp;
![asyncio](https://img.shields.io/badge/Async-asyncio_+_qasync-green) &nbsp;
![LLM](https://img.shields.io/badge/LLM-DeepSeek_|_Gemini_|_Bedrock_|_Local-purple)

</div>

---

<a name="中文版"></a>

<div align="right"><a href="#english">English ↓</a></div>

## amadeus-realtime

> 基于 GPT-SoVITS v3 的具身实时 AI Agent 运行时

### 项目定位

**amadeus-realtime** 是一个**具身 AI Agent 运行时**——在 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的推理内核之上，构建了"感知 → 推理 → 行动"的完整 Agent 闭环：

- **感知层**：麦克风语音识别（ASR）、截图 / 摄像头多模态输入、Google Live API 实时音频流
- **推理层**：多路 LLM 后端（DeepSeek / Gemini / Bedrock / 本地）+ 本地 RAG 知识增强
- **行动层**：实时语音合成（流式 TTS）、Live2D 表情 / 口型驱动、OpenClaw 外部工具委托执行

区别于通用聊天机器人，它以**角色具身**为核心——LLM 输出直接编码情绪表情标签和工具委托指令，由 VTS 动作层和 OpenClaw 代理层实时执行，形成"说话 + 表演 + 行动"三位一体的 Agent 行为。

本项目同时将 GPT-SoVITS 的离线批量 TTS 深度改造为流式实时管线，首句延迟 < 1s。

相比原版 GPT-SoVITS，核心改进：

| | 改进点 |
|---|---|
| 🧹 | 移除所有训练代码，只保留推理管线，启动更快、依赖更少 |
| ⚡ | GPT-SoVITS **流式改造**：逐句并发合成 + 边合成边播放，首句延迟 < 1s |
| 🔁 | 全异步架构（asyncio + qasync）：LLM / TTS / 播放三层流水线并行 |
| 🧠 | 集成多路 LLM 后端（DeepSeek / Gemini / AWS Bedrock / 本地） |
| 🎭 | VTuber Studio 深度集成：口型同步 + 表情/动作标签自动驱动 Live2D |
| 🎙️ | Google Live API 双向实时语音对话（client-side VAD） |
| 🖥️ | 多模态输入：截图 / 摄像头视觉上下文注入 |
| 🤖 | OpenClaw 工具调用委托：文件、搜索、代码执行全链路集成 |
| 📚 | 本地 FAISS RAG 知识库增强 |

---

### 核心架构特点

#### 1. GPT-SoVITS 流式改造

原版 GPT-SoVITS 的推理流程是"全文输入 → 完整音频输出 → 整段播放"，延迟极高。本项目对其进行了如下改造：

- **逐句分段推理**：LLM 流式输出按标点分句后，每句独立调用 TTS 推理
- **流水线并行**：句子 N 播放期间，句子 N+1 的 TTS 已在后台并发合成
- **首句冲刺模式**（First-Sentence Sprint）：第一句优先推理，优化感知延迟
- **CUDA Graph 可选加速**：开启后使用静态 KV 缓存，推理吞吐提升至 150–250 it/s
- **StreamPlayerWithBuffer**：音频分块流式送入 PyAudio，无需等待完整音频生成

```
LLM 输出流
   │  ← 分句
   ▼
[句1] → TTS 推理 → 音频块 → 播放 ─────────────────────────────▶
           [句2] → TTS 推理 → 音频块 ──────────────────▶
                      [句3] → TTS 推理 ───────────▶
```

#### 2. 全异步流水线（asyncio 架构）

| 层级 | 实现方式 | 说明 |
|---|---|---|
| LLM 流式接收 | `async for chunk in response` | 逐 token 接收，实时分句 |
| TTS 并发推理 | `asyncio.to_thread` + `ThreadPoolExecutor` | 推理在线程池，不阻塞事件循环 |
| 反压控制 | `asyncio.Queue(maxsize=3)` | 限制堆积句子数，避免内存爆炸 |
| 顺序播放保证 | `PlaybackManager` + 序号锁 | 确保乱序完成的 TTS 按序播放 |
| 预翻译并发 | `translation_executor` 独立线程池 | TTS 合成同时异步预取中文翻译 |
| Qt 事件循环 | `qasync.QEventLoop` | 将 asyncio 与 PyQt5 事件循环统一 |

#### 3. VTS 深度集成

- **口型同步**：播放线程实时计算音频 RMS，20Hz 频率驱动 VTS `MouthOpen` 参数
- **表情标签解析**：LLM 输出内嵌标签，由 `vts/action.py` 的队列消费线程执行
  - `[EMO preset=smile dur=2s]` — 预设情绪组合（PARAM + EXPR 联动）
  - `[EXPR name=Thinking.exp3.json dur=15s fade=0.3s]` — 精确控制单个表情
  - `[PARAM id=EyeOpenL,EyeOpenR value=0.5,0.5 fade=0.2s]` — 直接驱动参数
  - `[HOTKEY name=...]` — 触发 VTS 快捷键动作
- **自动复位**：每轮对话开始时重置所有激活表情，避免表情残留
- **心跳保活**：独立线程每 8 秒发送心跳，防止 VTS WebSocket 超时断开

#### 4. OpenClaw 工具调用集成

LLM 输出中可内嵌 `[DELEGATE task="..."]` 标签触发外部工具执行，全程不中断语音流：

```
LLM 输出: "少し待って。[DELEGATE task="今日の天気を調べて"] 調べてみるわ。"
              │
              ├─ 播放: "少し待って。調べてみるわ。"（TTS 正常合成）
              └─ 异步: OpenClaw 执行任务 → 结果注入对话历史 → 触发第二轮 LLM 回复
```

- **自动截图**：任务描述含视觉关键词时，自动截取屏幕附带给 OpenClaw
- **结果分类**：对 ok / question / partial / error 四种结果类型生成不同的后续提示词
- **非阻塞**：委托任务在 `asyncio.create_task` 中执行，不阻塞当前语音输出

---

### 系统架构

```
用户输入（语音 / 文字）
        │
        ▼
   ASR Manager                    ← asr/manager.py
        │
        ▼
  stream_llm_query                ← main.py（LLM 流式输出 + 分句）
   ├── DeepSeek / Gemini / Bedrock    ← llm/client.py
   ├── 本地 llama-server              ← llm/llama_server.py + llm/local_cli.py
   └── Google Live API                ← live/sidecar.py + live_api_sidecar.py
        │
        ├─ [DELEGATE] ──→ OpenClaw Gateway  ← openclaw/gateway.py
        │
        ▼
   TTS Pipeline                   ← tts/pipeline.py
   ├── GPT-SoVITS 逐句推理            ← local_tts_infer.py
   ├── 流式音频播放                   ← tts/playback.py (StreamPlayerWithBuffer)
   ├── 顺序调度                       ← tts/playback.py (PlaybackManager)
   └── 预翻译并发缓存                 ← tts/sentence_state.py
        │
        ▼
  VTS Action Layer                ← vts/action.py + vts/connection_manager.py
  ├── 口型同步 (RMS → MouthOpen, 20Hz)
  └── 表情队列 ([EMO] / [EXPR] / [PARAM] / [HOTKEY])

  Multimodal Controller           ← multimodal/controller.py
  GUI + 悬浮字幕                  ← chatGui.py + floating_subtitle.py
```

---

### 功能一览

| 功能 | 说明 |
|---|---|
| 流式 TTS | LLM 输出分句后立即并发合成，边说边播 |
| CUDA Graph 加速 | 可选开启，静态 KV 缓存，150–250 it/s |
| 反压队列 | `asyncio.Queue(maxsize=3)` 防止内存爆炸 |
| Live API 模式 | 接入 Google Gemini Live，client-side VAD 多轮对话 |
| 表情驱动 | 标签解析 + 队列执行 + 自动复位 |
| 口型同步 | RMS 实时计算，20Hz 驱动 VTS 参数 |
| OpenClaw 委托 | 非阻塞异步委托，结果自动回注对话 |
| 多模态 | 截图 / 摄像头帧注入 LLM 上下文 |
| 预翻译缓存 | TTS 合成同时异步预取中文翻译 |
| 多 LLM 后端 | DeepSeek / Gemini / Bedrock / llama-server / Ollama / LM Studio |
| 悬浮字幕 | 日语 + 中文翻译双轨，与播放进度同步 |
| 会话管理 | 多会话持久化，支持标题自动生成 |

---

### 环境要求

- Python 3.10+
- NVIDIA GPU（CUDA 11.8+ 或 12.x，推荐 RTX 3060 及以上）
- [VTuber Studio](https://denchisoft.com/)（可选，口型 / 表情同步）
- [OpenClaw](https://github.com/your-openclaw-repo)（可选，工具调用）

### 安装

**1. 安装 PyTorch**

从 [pytorch.org](https://pytorch.org/get-started/locally/) 按 CUDA 版本选择安装命令：

```bash
# CUDA 12.1 示例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**2. 安装项目依赖**

```bash
pip install -r requirements.txt
```

**3. 配置环境变量**

```bash
cp .env.example .env
# 填入 API Key、模型路径等
```

关键配置项：

```env
DEEPSEEK_API_KEY=sk-...          # 或 GEMINI_API_KEY=AIza...
TTS_GPT_MODEL_PATH=GPT_weights_v3/your-model.ckpt
TTS_SOVITS_MODEL_PATH=SoVITS_weights_v3/your-model.pth
VTS_WS_URL=ws://127.0.0.1:8001
```

完整说明见 `.env.example`。

### 使用

```bash
# 标准模式（DeepSeek）
python main.py

# Gemini 后端
python main.py --provider gemini

# 本地 llama-server
python main.py --local --type llama_server

# Ollama
python main.py --local --type ollama --model gemma3:12b
```

在 GUI 中开启 **Live Mode** 开关，即可启动 Google Gemini Live 双向实时语音对话模式。

### 性能调优

| 场景 | 配置 | 速度 |
|---|---|---|
| 高吞吐（单轮长句） | `ENABLE_CUDA_GRAPH=1` + `EXP_TTS_MAX_CONCURRENCY=1` | 150–250 it/s，串行 |
| 低延迟（多句快响） | `ENABLE_CUDA_GRAPH=0` + `EXP_TTS_MAX_CONCURRENCY=2` | 100–140 it/s，并发 |

### 上游项目

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — TTS 推理核心
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — 神经声码器
- [FunASR](https://github.com/modelscope/FunASR) — 语音识别后端

---

<a name="english"></a>

<div align="right"><a href="#中文版">中文 ↑</a></div>

## amadeus-realtime

> Embodied real-time AI Agent runtime built on GPT-SoVITS v3

### About

**amadeus-realtime** is an **embodied AI Agent runtime** — built on top of [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)'s inference core, it implements a full "perceive → reason → act" agent loop:

- **Perception**: microphone ASR, screenshot / camera multimodal input, Google Live API real-time audio stream
- **Reasoning**: multi-backend LLM (DeepSeek / Gemini / Bedrock / Local) + local RAG knowledge augmentation
- **Action**: streaming real-time TTS, Live2D expression / lip-sync control, OpenClaw external tool delegation

Unlike a generic chatbot, its core is **character embodiment** — LLM output directly encodes emotion expression tags and tool delegation commands, which are executed in real time by the VTS action layer and OpenClaw agent layer, forming a unified "speak + perform + act" agent behavior.

It also deeply rewrites GPT-SoVITS's offline batch TTS into a streaming real-time pipeline with first-sentence latency < 1s.

Key improvements over vanilla GPT-SoVITS:

| | Improvement |
|---|---|
| 🧹 | Training code removed; inference pipeline only — faster startup, fewer dependencies |
| ⚡ | GPT-SoVITS **streaming rewrite**: per-sentence concurrent synthesis + play-while-synthesizing, first-sentence latency < 1s |
| 🔁 | Fully async architecture (asyncio + qasync): LLM / TTS / playback as a three-stage parallel pipeline |
| 🧠 | Multi-backend LLM (DeepSeek / Gemini / AWS Bedrock / Local) |
| 🎭 | Deep VTuber Studio integration: lip-sync + expression/action tag → Live2D automation |
| 🎙️ | Google Live API bidirectional voice (client-side VAD, multi-turn) |
| 🖥️ | Multimodal input: screenshot / camera visual context injection |
| 🤖 | OpenClaw tool-use delegation: files, search, code execution — fully integrated |
| 📚 | Local FAISS RAG knowledge base augmentation |

---

### Core Architecture Highlights

#### 1. GPT-SoVITS Streaming Rewrite

The original GPT-SoVITS follows a "full text → full audio → play" flow with high latency. This project rewrites it as:

- **Per-sentence inference**: LLM output is segmented by punctuation; each sentence is inferred independently
- **Pipeline parallelism**: while sentence N is playing, sentence N+1 is already being synthesized in the background
- **First-sentence sprint**: the first sentence is prioritized to minimize perceived latency
- **Optional CUDA Graph acceleration**: static KV cache, 150–250 it/s throughput
- **StreamPlayerWithBuffer**: audio chunks are fed to PyAudio incrementally — no waiting for full audio generation

```
LLM output stream
   │  ← sentence segmentation
   ▼
[S1] → TTS inference → audio chunks → playback ──────────────▶
           [S2] → TTS inference → audio chunks ──────────▶
                      [S3] → TTS inference ─────────▶
```

#### 2. Fully Async Pipeline (asyncio Architecture)

| Layer | Implementation | Description |
|---|---|---|
| LLM streaming | `async for chunk in response` | Token-by-token receive + real-time segmentation |
| Concurrent TTS | `asyncio.to_thread` + `ThreadPoolExecutor` | Inference runs in thread pool, non-blocking |
| Backpressure | `asyncio.Queue(maxsize=3)` | Caps queued sentences to prevent memory bloat |
| Ordered playback | `PlaybackManager` + sequence locks | Ensures out-of-order TTS completions play in sequence |
| Pre-translation | Dedicated `translation_executor` thread pool | Chinese translation fetched concurrently while TTS synthesizes |
| Qt event loop | `qasync.QEventLoop` | Unifies asyncio and PyQt5 event loops |

#### 3. Deep VTuber Studio Integration

- **Lip sync**: playback thread computes audio RMS in real time, drives VTS `MouthOpen` parameter at 20 Hz
- **Expression tag parsing**: LLM output embeds tags consumed by the `vts/action.py` queue worker:
  - `[EMO preset=smile dur=2s]` — preset emotion combos (PARAM + EXPR combined)
  - `[EXPR name=Thinking.exp3.json dur=15s fade=0.3s]` — fine-grained single expression control
  - `[PARAM id=EyeOpenL,EyeOpenR value=0.5,0.5 fade=0.2s]` — direct parameter driving
  - `[HOTKEY name=...]` — triggers VTS hotkey actions
- **Auto reset**: all active expressions are cleared at the start of each turn to prevent residual states
- **Heartbeat keepalive**: dedicated thread sends heartbeat every 8s to prevent VTS WebSocket timeout

#### 4. OpenClaw Tool-Use Integration

LLM output can embed `[DELEGATE task="..."]` tags to trigger external tool execution without interrupting the voice stream:

```
LLM: "少し待って。[DELEGATE task="Look up today's weather"] 調べてみるわ。"
              │
              ├─ TTS plays: "少し待って。調べてみるわ。" (normal synthesis)
              └─ async: OpenClaw executes task → result injected into history → triggers 2nd LLM turn
```

- **Auto screenshot**: when the task description contains visual keywords, the screen is captured and attached
- **Result classification**: generates different follow-up prompts for ok / question / partial / error result types
- **Non-blocking**: delegation runs in `asyncio.create_task`, never blocks the current voice output

---

### Architecture

```
User Input (voice / text)
        │
        ▼
   ASR Manager                    ← asr/manager.py
        │
        ▼
  stream_llm_query                ← main.py (LLM streaming + sentence segmentation)
   ├── DeepSeek / Gemini / Bedrock    ← llm/client.py
   ├── Local llama-server             ← llm/llama_server.py + llm/local_cli.py
   └── Google Live API                ← live/sidecar.py + live_api_sidecar.py
        │
        ├─ [DELEGATE] ──→ OpenClaw Gateway  ← openclaw/gateway.py
        │
        ▼
   TTS Pipeline                   ← tts/pipeline.py
   ├── GPT-SoVITS per-sentence inference  ← local_tts_infer.py
   ├── Streaming audio playback           ← tts/playback.py (StreamPlayerWithBuffer)
   ├── Ordered scheduling                 ← tts/playback.py (PlaybackManager)
   └── Concurrent pre-translation cache   ← tts/sentence_state.py
        │
        ▼
  VTS Action Layer                ← vts/action.py + vts/connection_manager.py
  ├── Lip sync (RMS → MouthOpen, 20 Hz)
  └── Expression queue ([EMO] / [EXPR] / [PARAM] / [HOTKEY])

  Multimodal Controller           ← multimodal/controller.py
  GUI + Floating Subtitle         ← chatGui.py + floating_subtitle.py
```

---

### Features

| Feature | Description |
|---|---|
| Streaming TTS | Per-sentence concurrent synthesis; playback starts immediately |
| CUDA Graph | Optional static KV cache; 150–250 it/s |
| Backpressure queue | `asyncio.Queue(maxsize=3)` prevents memory bloat |
| Live Mode | Google Gemini Live, client-side VAD, multi-turn |
| Expression control | Tag parsing + queue execution + auto reset |
| Lip sync | RMS-based, 20 Hz, drives VTS parameters |
| OpenClaw delegation | Non-blocking async delegation; result auto-injected into conversation |
| Multimodal | Screenshots or camera frames as LLM context |
| Pre-translation cache | Chinese translation fetched concurrently with TTS synthesis |
| Multi-LLM | DeepSeek / Gemini / Bedrock / llama-server / Ollama / LM Studio |
| Floating subtitle | JP + CN dual-track, synchronized with playback |
| Session management | Multi-session persistence with auto title generation |

---

### Requirements

- Python 3.10+
- NVIDIA GPU (CUDA 11.8+ or 12.x, RTX 3060 or above recommended)
- [VTuber Studio](https://denchisoft.com/) (optional — lip-sync & expressions)
- [OpenClaw](https://github.com/your-openclaw-repo) (optional — tool use)

### Installation

**1. Install PyTorch**

Select the command for your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# Example: CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**2. Install project dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment**

```bash
cp .env.example .env
# Fill in API keys, model paths, etc.
```

Key settings:

```env
DEEPSEEK_API_KEY=sk-...          # or GEMINI_API_KEY=AIza...
TTS_GPT_MODEL_PATH=GPT_weights_v3/your-model.ckpt
TTS_SOVITS_MODEL_PATH=SoVITS_weights_v3/your-model.pth
VTS_WS_URL=ws://127.0.0.1:8001
```

See `.env.example` for the full reference.

### Usage

```bash
# Standard mode (DeepSeek)
python main.py

# Gemini backend
python main.py --provider gemini

# Local llama-server
python main.py --local --type llama_server

# Ollama
python main.py --local --type ollama --model gemma3:12b
```

Toggle **Live Mode** in the GUI to start Google Gemini Live bidirectional real-time voice conversation.

### Performance Tuning

| Scenario | Config | Throughput |
|---|---|---|
| High-throughput (long single turn) | `ENABLE_CUDA_GRAPH=1` + `EXP_TTS_MAX_CONCURRENCY=1` | 150–250 it/s, serialized |
| Low-latency (rapid multi-sentence) | `ENABLE_CUDA_GRAPH=0` + `EXP_TTS_MAX_CONCURRENCY=2` | 100–140 it/s, concurrent |

### Upstream Projects

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — TTS inference core
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — Neural vocoder
- [FunASR](https://github.com/modelscope/FunASR) — ASR backend

---

<div align="center">MIT License · Based on GPT-SoVITS · Private deployment only</div>
