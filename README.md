<div align="center">

# amadeus-realtime

**语言 / Language**

[![中文](https://img.shields.io/badge/中文-README-blue)](#中文版) &nbsp;|&nbsp; [![English](https://img.shields.io/badge/English-README-blue)](#english)

</div>

---

<a name="中文版"></a>

<div align="right"><a href="#english">English ↓</a></div>

## amadeus-realtime

> 基于 GPT-SoVITS v3 的实时 AI 伴侣运行时

### 项目定位

**amadeus-realtime** 是在 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 基础上深度定制的**推理专用**系统，核心目标是让 AI 角色（牧瀬紅莉栖 / Kurisu）以接近实时的延迟完成多模态交互。

相比原版 GPT-SoVITS，本项目的核心改进：

| | 改进点 |
|---|---|
| 🧹 | 移除所有训练代码，只保留推理管线，启动更快、依赖更少 |
| 🧠 | 集成多路 LLM 大脑（DeepSeek / Gemini / AWS Bedrock / 本地） |
| ⚡ | 流式 TTS 管线，首句延迟 < 1s（CUDA 下） |
| 🎭 | VTuber Studio 口型 + 表情标签自动驱动 Live2D |
| 🎙️ | Google Live API 双向实时语音对话 |
| 🖥️ | 多模态输入：截图 / 摄像头视觉上下文注入 |
| 🤖 | OpenClaw 工具调用委托（文件、搜索、代码执行） |
| 📚 | 本地 FAISS RAG 知识库增强 |

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
        ▼
   TTS Pipeline                   ← tts/pipeline.py
   ├── GPT-SoVITS 推理                ← local_tts_infer.py
   ├── 音频流播放                     ← tts/playback.py
   └── 预翻译缓存                     ← tts/sentence_state.py
        │
        ▼
  VTS Action Layer                ← vts/action.py + vts/connection_manager.py
  ├── 口型同步（WebSocket → VTS）
  └── 表情/动作标签执行（[EMO] / [EXPR] / [PARAM]）

  Multimodal Controller           ← multimodal/controller.py
  OpenClaw Gateway                ← openclaw/gateway.py + openclaw/client.py
  GUI                             ← chatGui.py + floating_subtitle.py
```

### 功能一览

| 功能 | 说明 |
|---|---|
| 流式 TTS | LLM 输出分句后立即并发合成，边说边播 |
| CUDA Graph 加速 | 可选开启，静态 KV 缓存，高并发推理 |
| Live API 模式 | 直接接入 Google Gemini Live，实时双向语音 |
| 表情驱动 | LLM 输出内嵌 `[EMO]` / `[EXPR]` 标签，自动驱动 Live2D |
| 多模态 | 截图或摄像头帧注入 LLM 上下文 |
| OpenClaw 委托 | LLM 输出 `[DELEGATE]` 触发外部工具执行 |
| RAG 知识库 | 本地 FAISS 向量检索，增强角色专属知识 |
| 多 LLM 后端 | DeepSeek / Gemini / Bedrock / llama-server / Ollama / LM Studio |
| 悬浮字幕 | 实时日语 + 中文翻译双轨字幕窗口 |
| 会话管理 | 多会话持久化，支持标题自动生成 |

### 环境要求

- Python 3.10+
- NVIDIA GPU（CUDA 11.8+ 或 12.x，推荐 RTX 3060 及以上）
- [VTuber Studio](https://denchisoft.com/)（可选，口型 / 表情同步）
- [OpenClaw](https://github.com/your-openclaw-repo)（可选，工具调用）

### 安装

**1. 安装 PyTorch**

先按 CUDA 版本从 [pytorch.org](https://pytorch.org/get-started/locally/) 选择安装命令：

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
# 用编辑器填入 API Key、模型路径等
```

关键配置项：

```env
# LLM（选其一）
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# TTS 模型路径（相对项目根目录或绝对路径）
TTS_GPT_MODEL_PATH=GPT_weights_v3/your-model.ckpt
TTS_SOVITS_MODEL_PATH=SoVITS_weights_v3/your-model.pth

# VTS WebSocket（与 VTuber Studio 端口一致）
VTS_WS_URL=ws://127.0.0.1:8001
```

完整说明见 `.env.example`。

### 使用

```bash
# 标准模式
python main.py

# 指定 Gemini 后端
python main.py --provider gemini

# 使用本地 llama-server
python main.py --local --type llama_server

# 使用 Ollama
python main.py --local --type ollama --model gemma3:12b
```

在 GUI 中开启 **Live Mode** 开关，即可启动 Google Gemini Live 双向实时语音对话模式。

### 模块结构

```
amadeus-realtime/
├── main.py                    # 启动入口 + LLM 流式编排核心
├── live_api_sidecar.py        # Google Live API 独立子进程
├── chatGui.py                 # PyQt5 GUI
├── config/settings.py         # 统一配置（读取 .env）
├── asr/manager.py             # 语音识别管理
├── llm/
│   ├── client.py              # 同步非流式 LLM 查询
│   ├── llama_server.py        # llama-server 进程管理
│   └── local_cli.py           # 本地 CLI 流式查询
├── tts/
│   ├── pipeline.py            # TTS 合成 + 调度管线
│   ├── playback.py            # 音频流播放
│   ├── sentence_state.py      # 句子状态 + 预翻译缓存
│   └── subtitle.py            # 字幕显示
├── vts/
│   ├── connection_manager.py  # VTS WebSocket 连接
│   └── action.py              # 表情 / 动作 / 心跳
├── live/sidecar.py            # Live API sidecar 进程管理
├── multimodal/controller.py   # 多模态输入监控
├── openclaw/
│   ├── client.py              # OpenClaw API 客户端
│   └── gateway.py             # Gateway 进程管理
├── core/session_manager.py    # 多会话持久化
└── tools/
    ├── text_utils.py          # 纯文本工具函数
    └── tts_text_processor.py  # TTS 预处理 + EMO 预设
```

### 性能调优

**CUDA Graph 加速（高吞吐场景）**

```bash
ENABLE_CUDA_GRAPH=1 python main.py
```

启用后首句推理速度可达 150–250 it/s，推理串行化，适合单轮长句场景。

**关闭 CUDA Graph（低延迟优先）**

设置 `EXP_TTS_MAX_CONCURRENCY=2`，并发推理，稳定 100–140 it/s，适合多句快速响应场景。

### 上游项目

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — TTS 推理核心
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — 神经声码器
- [FunASR](https://github.com/modelscope/FunASR) — 语音识别后端

---

<a name="english"></a>

<div align="right"><a href="#中文版">中文 ↑</a></div>

## amadeus-realtime

> Real-time AI companion runtime built on GPT-SoVITS v3

### About

**amadeus-realtime** is a heavily customized **inference-only** runtime built on top of [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). Its goal is to enable an AI character (Makise Kurisu) to engage in multimodal, near-real-time interaction.

Key improvements over vanilla GPT-SoVITS:

| | Improvement |
|---|---|
| 🧹 | Training code removed; inference pipeline only — faster startup, fewer dependencies |
| 🧠 | Multi-backend LLM brain (DeepSeek / Gemini / AWS Bedrock / Local) |
| ⚡ | Streaming TTS pipeline, first-sentence latency < 1s on CUDA |
| 🎭 | VTuber Studio lip-sync + expression tag → Live2D automation |
| 🎙️ | Google Live API bidirectional real-time voice conversation |
| 🖥️ | Multimodal input: screenshot / camera visual context injection |
| 🤖 | OpenClaw tool-use delegation (files, search, code execution) |
| 📚 | Local FAISS RAG knowledge base augmentation |

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
        ▼
   TTS Pipeline                   ← tts/pipeline.py
   ├── GPT-SoVITS Inference           ← local_tts_infer.py
   ├── Streaming Audio Playback       ← tts/playback.py
   └── Pre-translation Cache          ← tts/sentence_state.py
        │
        ▼
  VTS Action Layer                ← vts/action.py + vts/connection_manager.py
  ├── Lip sync (WebSocket → VTS)
  └── Expression / Action tag execution ([EMO] / [EXPR] / [PARAM])

  Multimodal Controller           ← multimodal/controller.py
  OpenClaw Gateway                ← openclaw/gateway.py + openclaw/client.py
  GUI                             ← chatGui.py + floating_subtitle.py
```

### Features

| Feature | Description |
|---|---|
| Streaming TTS | Sentences synthesized concurrently as LLM streams; playback starts immediately |
| CUDA Graph | Optional; enables static KV cache for higher throughput |
| Live Mode | Direct Google Gemini Live bidirectional voice |
| Expression Control | Inline `[EMO]` / `[EXPR]` tags drive Live2D automatically |
| Multimodal | Screenshots or camera frames injected as LLM context |
| Tool Use | `[DELEGATE]` tag triggers external tool execution via OpenClaw |
| RAG | Local FAISS vector search for character-specific knowledge |
| Multi-LLM | DeepSeek / Gemini / Bedrock / llama-server / Ollama / LM Studio |
| Subtitle | Real-time JP + CN dual-track floating subtitle window |
| Sessions | Multi-session persistence with auto title generation |

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
# LLM — choose one
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# TTS model paths (relative to project root or absolute)
TTS_GPT_MODEL_PATH=GPT_weights_v3/your-model.ckpt
TTS_SOVITS_MODEL_PATH=SoVITS_weights_v3/your-model.pth

# VTuber Studio WebSocket (match VTS port setting)
VTS_WS_URL=ws://127.0.0.1:8001
```

See `.env.example` for the full reference.

### Usage

```bash
# Standard mode
python main.py

# Use Gemini backend
python main.py --provider gemini

# Local llama-server
python main.py --local --type llama_server

# Ollama
python main.py --local --type ollama --model gemma3:12b
```

Toggle **Live Mode** in the GUI to start Google Gemini Live bidirectional real-time voice conversation.

### Module Structure

```
amadeus-realtime/
├── main.py                    # Entry point + stream_llm_query orchestration
├── live_api_sidecar.py        # Google Live API subprocess
├── chatGui.py                 # PyQt5 GUI
├── config/settings.py         # Unified config from .env
├── asr/manager.py             # ASR speech recognition
├── llm/
│   ├── client.py              # Sync non-streaming LLM queries
│   ├── llama_server.py        # llama-server process management
│   └── local_cli.py           # Local CLI LLM streaming
├── tts/
│   ├── pipeline.py            # TTS synthesis + scheduling pipeline
│   ├── playback.py            # Streaming audio playback
│   ├── sentence_state.py      # Sentence state + pre-translation cache
│   └── subtitle.py            # Subtitle display
├── vts/
│   ├── connection_manager.py  # VTS WebSocket connection
│   └── action.py              # Expression / action / heartbeat
├── live/sidecar.py            # Live API sidecar process management
├── multimodal/controller.py   # Multimodal input monitoring
├── openclaw/
│   ├── client.py              # OpenClaw API client
│   └── gateway.py             # Gateway process management
├── core/session_manager.py    # Multi-session persistence
└── tools/
    ├── text_utils.py          # Pure text utility functions
    └── tts_text_processor.py  # TTS text preprocessing + EMO presets
```

### Performance Tuning

**CUDA Graph acceleration (high-throughput)**

```bash
ENABLE_CUDA_GRAPH=1 python main.py
```

Enables 150–250 it/s on first sentence; serializes inference, best for single long-turn scenarios.

**Without CUDA Graph (low-latency priority)**

Set `EXP_TTS_MAX_CONCURRENCY=2`. Concurrent inference, stable 100–140 it/s, best for rapid multi-sentence responses.

### Upstream Projects

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — TTS inference core
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — Neural vocoder
- [FunASR](https://github.com/modelscope/FunASR) — ASR backend

---

<div align="center">MIT License · Based on GPT-SoVITS · Private deployment only</div>
