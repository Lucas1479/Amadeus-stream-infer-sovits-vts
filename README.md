# amadeus-realtime

> 基于 GPT-SoVITS v3 的实时 AI 伴侣运行时  
> Real-time AI companion runtime built on GPT-SoVITS v3

---

## 项目定位 / About

**amadeus-realtime** 是在 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 基础上深度定制的**推理专用**系统，核心目标是让 AI 角色（牧瀬紅莉栖 / Kurisu）以接近实时的延迟完成多模态交互。

**amadeus-realtime** is a heavily customized **inference-only** runtime built on top of [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). Its goal is to enable an AI character (Makise Kurisu) to engage in multimodal, near-real-time interaction.

相比原版 GPT-SoVITS，本项目的核心改进：  
Key improvements over vanilla GPT-SoVITS:

| | 中文 | English |
|---|---|---|
| 🧹 | 移除所有训练代码，只保留推理管线 | Training code removed; inference pipeline only |
| 🧠 | 集成多路 LLM 大脑（DeepSeek / Gemini / Bedrock / 本地） | Multi-backend LLM brain (DeepSeek / Gemini / Bedrock / Local) |
| ⚡ | 流式 TTS 管线，首句延迟 < 1s（CUDA 下） | Streaming TTS pipeline, first-sentence latency < 1s on CUDA |
| 🎭 | VTuber Studio 口型 + 表情标签自动驱动 Live2D | VTS lip-sync + expression tag → Live2D automation |
| 🎙️ | Google Live API 双向实时语音对话 | Google Live API bidirectional real-time voice conversation |
| 🖥️ | 多模态输入：截图 / 摄像头视觉上下文注入 | Multimodal: screenshot / camera visual context injection |
| 🤖 | OpenClaw 工具调用委托（文件、搜索、代码执行） | OpenClaw tool-use delegation (files, search, code execution) |
| 📚 | 本地 FAISS RAG 知识库增强 | Local FAISS RAG knowledge base augmentation |

---

## 系统架构 / Architecture

```
User Input / 用户输入（语音 / 文字）
        │
        ▼
   ASR Manager                    ← asr/manager.py
        │
        ▼
  stream_llm_query                ← main.py（LLM streaming + sentence segmentation）
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

---

## 功能一览 / Features

| 功能 / Feature | 说明 / Description |
|---|---|
| 流式 TTS / Streaming TTS | LLM 输出分句后立即并发合成，边说边播 / Sentences synthesized concurrently as LLM streams |
| CUDA Graph 加速 / CUDA Graph | 可选开启，静态 KV 缓存 / Optional; enables static KV cache for higher throughput |
| Live API 模式 / Live Mode | 直接接入 Google Gemini Live，实时双向语音 / Direct Google Gemini Live bidirectional voice |
| 表情驱动 / Expression Control | LLM 输出内嵌 `[EMO]` / `[EXPR]` 标签，自动驱动 Live2D / Inline emotion tags drive Live2D automatically |
| 多模态 / Multimodal | 截图或摄像头帧注入 LLM 上下文 / Screenshots or camera frames injected as LLM context |
| OpenClaw 委托 / Tool Use | LLM 输出 `[DELEGATE]` 触发外部工具执行 / `[DELEGATE]` tag triggers external tool execution |
| RAG 知识库 / RAG | 本地 FAISS 向量检索，增强角色专属知识 / Local FAISS vector search for character-specific knowledge |
| 多 LLM 后端 / Multi-LLM | DeepSeek / Gemini / Bedrock / llama-server / Ollama / LM Studio | |
| 悬浮字幕 / Subtitle | 实时日语 + 中文翻译双轨字幕 / Real-time JP + CN dual-track floating subtitle |
| 会话管理 / Session | 多会话持久化，支持标题自动生成 / Multi-session persistence with auto title generation |

---

## 环境要求 / Requirements

- Python 3.10+
- NVIDIA GPU（CUDA 11.8+ 或 12.x，推荐 RTX 3060 及以上 / RTX 3060 or above recommended）
- [VTuber Studio](https://denchisoft.com/)（可选 / optional — lip-sync & expressions）
- [OpenClaw](https://github.com/your-openclaw-repo)（可选 / optional — tool use）

---

## 安装 / Installation

### 1. 安装 PyTorch / Install PyTorch

先按 CUDA 版本从官网选择安装命令：  
Select the command for your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# Example: CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装项目依赖 / Install project dependencies

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量 / Configure environment

```bash
cp .env.example .env
# 用编辑器填入 API Key、模型路径等
# Fill in API keys, model paths, etc.
```

关键配置项 / Key settings：

```env
# LLM — choose one / 选择一个
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# TTS model paths (relative to project root or absolute)
# TTS 模型路径（相对项目根目录或绝对路径）
TTS_GPT_MODEL_PATH=GPT_weights_v3/your-model.ckpt
TTS_SOVITS_MODEL_PATH=SoVITS_weights_v3/your-model.pth

# VTuber Studio WebSocket (match VTS port setting)
# VTS WebSocket（与 VTuber Studio 端口设置一致）
VTS_WS_URL=ws://127.0.0.1:8001
```

完整说明见 `.env.example` / See `.env.example` for full reference.

---

## 使用 / Usage

### 标准模式 / Standard mode（GUI + text / voice input）

```bash
python main.py
```

### 指定 LLM 后端 / Select LLM backend

```bash
# Gemini
python main.py --provider gemini

# Local llama-server
python main.py --local --type llama_server

# Ollama
python main.py --local --type ollama --model gemma3:12b
```

### Live API 模式 / Live API mode

在 GUI 中开启 **Live Mode** 开关。系统将启动 `live_api_sidecar.py` 子进程，直接接入 Google Gemini Live 进行双向实时语音对话。

Toggle **Live Mode** in the GUI. The system launches `live_api_sidecar.py` as a subprocess to connect directly to Google Gemini Live for bidirectional real-time voice conversation.

---

## 模块结构 / Module Structure

```
amadeus-realtime/
├── main.py                    # Entry point + stream_llm_query orchestration
│                              # 启动入口 + LLM 流式编排核心
├── live_api_sidecar.py        # Google Live API subprocess
│                              # Google Live API 独立子进程
├── chatGui.py                 # PyQt5 GUI
├── floating_subtitle.py       # Floating subtitle window / 悬浮字幕窗
├── local_tts_infer.py         # GPT-SoVITS inference wrapper / 推理封装
├── rag_system.py              # RAG knowledge base / RAG 知识库
│
├── config/
│   └── settings.py            # Unified config from .env / 统一配置
│
├── asr/
│   └── manager.py             # ASR speech recognition / 语音识别管理
│
├── llm/
│   ├── client.py              # Sync non-streaming LLM queries / 同步非流式 LLM 查询
│   ├── llama_server.py        # llama-server process management / 进程管理
│   └── local_cli.py           # Local CLI LLM streaming / 本地 CLI 流式查询
│
├── tts/
│   ├── pipeline.py            # TTS synthesis + scheduling / 合成 + 调度管线
│   ├── playback.py            # Streaming audio playback / 音频流播放
│   ├── sentence_state.py      # Sentence state + pre-translation cache / 句子状态 + 预翻译缓存
│   └── subtitle.py            # Subtitle display / 字幕显示
│
├── vts/
│   ├── connection_manager.py  # VTS WebSocket connection / VTS 连接管理
│   └── action.py              # Expression / action / heartbeat / 表情、动作、心跳
│
├── live/
│   └── sidecar.py             # Live API sidecar process management / sidecar 进程管理
│
├── multimodal/
│   └── controller.py          # Multimodal input monitoring / 多模态输入监控
│
├── openclaw/
│   ├── client.py              # OpenClaw API client / 客户端
│   └── gateway.py             # OpenClaw Gateway process / Gateway 进程管理
│
├── core/
│   └── session_manager.py     # Multi-session persistence / 多会话持久化
│
└── tools/
    ├── text_utils.py          # Pure text utility functions / 纯文本工具函数
    └── tts_text_processor.py  # TTS text preprocessing + EMO presets / TTS 预处理 + EMO 预设
```

---

## 性能调优 / Performance Tuning

### CUDA Graph 加速 / CUDA Graph acceleration（high-throughput scenarios）

```env
# .env
EXP_TTS_MAX_CONCURRENCY=1
```

```bash
ENABLE_CUDA_GRAPH=1 python main.py
```

启用后首句推理速度可达 150–250 it/s，推理串行化，适合单轮长句场景。  
Enables 150–250 it/s on first sentence; serializes inference, best for single long-turn scenarios.

### 关闭 CUDA Graph / Without CUDA Graph（low-latency priority）

```env
EXP_TTS_MAX_CONCURRENCY=2
```

并发推理，稳定 100–140 it/s，适合多句快速响应场景。  
Concurrent inference, stable 100–140 it/s, best for rapid multi-sentence responses.

---

## 上游项目 / Upstream Projects

本项目基于以下开源项目构建 / Built upon:

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — TTS inference core / TTS 推理核心
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — Neural vocoder / 神经声码器
- [FunASR](https://github.com/modelscope/FunASR) — ASR backend / 语音识别后端

---

## License

本项目继承 GPT-SoVITS 的 MIT License。  
This project inherits the MIT License from GPT-SoVITS.

模型权重和语音数据不随代码分发，仅用于个人私有部署。  
Model weights and voice data are not distributed with the code and are for private personal use only.
