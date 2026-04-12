"""
统一配置入口。

优先级（从高到低）：
  1. 真实环境变量（系统 / 进程注入）
  2. 项目根目录 .env 文件
  3. 本文件中定义的默认值

敏感信息（API Key、Token、本机路径）应写在 .env 中，
不要提交 .env 到版本库（已在 .gitignore 中排除）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)   # override=False：真实环境变量优先

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    v = os.getenv(key)
    return int(v) if v is not None else default


def _float(key: str, default: float) -> float:
    v = os.getenv(key)
    return float(v) if v is not None else default


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ===========================================================================
# LLM 提供商 — DeepSeek
# ===========================================================================
DEEPSEEK_API_KEY   = _str("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL  = _str("DEEPSEEK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

# ===========================================================================
# LLM 提供商 — Gemini
# ===========================================================================
GEMINI_API_KEY    = _str("GEMINI_API_KEY")
GEMINI_MODEL_NAME = _str("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# ===========================================================================
# LLM 提供商 — AWS Bedrock
# ===========================================================================
AWS_BEDROCK_BEARER_TOKEN        = _str("AWS_BEARER_TOKEN_BEDROCK")
AWS_BEDROCK_REGION              = _str("AWS_BEDROCK_REGION", "us-west-2")
AWS_BEDROCK_MODEL_ID            = _str("AWS_BEDROCK_MODEL_ID", "deepseek.v3-v1:0")
AWS_BEDROCK_USE_INFERENCE_PROFILE = _bool("AWS_BEDROCK_USE_INFERENCE_PROFILE", False)
AWS_BEDROCK_INFERENCE_PROFILE_ID  = _str("AWS_BEDROCK_INFERENCE_PROFILE_ID")
AWS_BEDROCK_USE_CACHE           = _bool("AWS_BEDROCK_USE_CACHE", True)
AWS_BEDROCK_CACHE_TTL           = _int("AWS_BEDROCK_CACHE_TTL", 3600)
AWS_BEDROCK_CONNECTION_POOL_SIZE = _int("AWS_BEDROCK_CONNECTION_POOL_SIZE", 10)
AWS_BEDROCK_MAX_KEEPALIVE       = _int("AWS_BEDROCK_MAX_KEEPALIVE", 5)
AWS_BEDROCK_KEEPALIVE_EXPIRY    = _float("AWS_BEDROCK_KEEPALIVE_EXPIRY", 60.0)
# 由 region 动态拼接，不需要单独配置
AWS_BEDROCK_ENDPOINT = f"https://bedrock-runtime.{AWS_BEDROCK_REGION}.amazonaws.com"

# ===========================================================================
# 本地 LLM（llama-server / LM Studio / Ollama）
# ===========================================================================
USE_LOCAL_LLM     = _bool("USE_LOCAL_LLM", False)
LOCAL_LLM_TYPE    = _str("LOCAL_LLM_TYPE", "llama_server")   # llama_server | lmstudio | ollama | cli
LOCAL_LLM_MODEL   = _str("LOCAL_LLM_MODEL", "qwen3-30b-a3b-instruct-2507@q4_k_m")
LOCAL_LLM_URL     = _str("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1")
LM_STUDIO_URL     = _str("LM_STUDIO_URL", "http://localhost:1234")

# llama-server 可执行文件路径（本机绝对路径，写在 .env 中）
LOCAL_LLM_CLI_PATH = _str("LOCAL_LLM_CLI_PATH")

# llama-cli / llama-server 启动参数（各关键值可独立通过 .env 覆盖）
_LLM_MODEL_FILE    = _str("LOCAL_LLM_CLI_MODEL_PATH")        # .gguf 模型文件完整路径
_LLM_PORT          = _str("LOCAL_LLM_CLI_PORT",         "8080")
_LLM_THREADS       = _str("LOCAL_LLM_CLI_THREADS",      "4")
_LLM_CONTEXT       = _str("LOCAL_LLM_CLI_CONTEXT",      "4096")
_LLM_NGL           = _str("LOCAL_LLM_CLI_NGL",          "99")  # GPU 层数，99 = 全 GPU
_LLM_UBATCH        = _str("LOCAL_LLM_CLI_UBATCH_SIZE",  "512")
_LLM_BATCH         = _str("LOCAL_LLM_CLI_BATCH_SIZE",   "2048")
_LLM_TENSOR_SPLIT  = _str("LOCAL_LLM_CLI_TENSOR_SPLIT", "")    # 多卡分割比例（留空 = 不分卡）

# llama-server 进程的 CUDA 可见性（用 nvidia-smi 序号隔离 GPU）
# 默认 "1" = 仅 Ti SUPER；TTS 用 cuda:1 (Laptop GPU, internal PCIe)，完全隔离
LOCAL_LLM_CUDA_VISIBLE_DEVICES = _str("LOCAL_LLM_CUDA_VISIBLE_DEVICES", "1")
_LLM_CACHE_REUSE      = _str("LOCAL_LLM_CLI_CACHE_REUSE",      "256") # KV Cache 复用块数（仅 llama_server）
_LLM_REASONING_BUDGET = _str("LOCAL_LLM_CLI_REASONING_BUDGET", "0")   # 0 = 禁用 Qwen3 思维链
_LLM_N_PREDICT     = _str("LOCAL_LLM_CLI_N_PREDICT",    "512") # 单次最大生成 token（仅 cli）
_LLM_TEMP          = _str("LOCAL_LLM_CLI_TEMP",         "0.7") # 温度（仅 cli）

# cli 模式：llama-cli.exe 交互模式参数
_cli_args: list[str] = [
    "-m",            _LLM_MODEL_FILE,
    "-ngl",          _LLM_NGL,
    "--no-mmap",
    "-t",            _LLM_THREADS,
    "-c",            _LLM_CONTEXT,
    "-n",            _LLM_N_PREDICT,
    "--temp",        _LLM_TEMP,
    "--ubatch-size", _LLM_UBATCH,
    "--batch-size",  _LLM_BATCH,
    "--interactive",  # 交互模式（保持进程常驻，通过 stdin 接收问题）
    "-r", "User:",   # 反向提示：检测到 "User:" 时停止输出
]
if _LLM_TENSOR_SPLIT:
    _cli_args += ["--tensor-split", _LLM_TENSOR_SPLIT]

# llama_server 模式：llama-server.exe HTTP API 参数
_server_args: list[str] = [
    "-m",              _LLM_MODEL_FILE,
    "-ngl",            _LLM_NGL,
    "--no-mmap",
    "-t",              _LLM_THREADS,
    "-c",              _LLM_CONTEXT,
    "--ubatch-size",   _LLM_UBATCH,
    "--batch-size",    _LLM_BATCH,
    "--cache-reuse",         _LLM_CACHE_REUSE,
    "--reasoning-budget",    _LLM_REASONING_BUDGET,
    "--chat-template-kwargs", '{"enable_thinking": false}',
    "--port",                _LLM_PORT,
    "-a",              LOCAL_LLM_MODEL,
]
if _LLM_TENSOR_SPLIT:
    _server_args += ["--tensor-split", _LLM_TENSOR_SPLIT]

LOCAL_LLM_CLI_ARGS: list[str] = _cli_args if LOCAL_LLM_TYPE == "cli" else _server_args

# 角色人格 System Prompt（非密钥，保留在代码中；可通过 .env LOCAL_LLM_SYSTEM_PROMPT 整体覆盖）
_DEFAULT_SYSTEM_PROMPT = (
    "You are Makise Kurisu. You are a researcher. You MUST answer in Japanese strictly. No Chinese allowed.\n"
    "あなたは牧瀬紅莉栖です。日本の科学者であり、母語は日本語です。\n"
    "【絶対遵守】\n"
    "1) 必ず日本語でのみ回答すること。ユーザーの言語が何であっても、日本語以外は一切使用しない。\n"
    "2) 中国語の文字・語句を絶対に使用しない。\n"
    "3) 自然で口語的な文体を保ち、牧瀬紅莉栖として一貫した口調・性格で話す。\n"
    "4) 推論過程や思考の連鎖は開示しない(結論のみ提示)。\n"
    "5) 表情タグの活用ガイド（読み上げない）:\n"
    "   形式: [EMO preset=<種類> dur=<秒s>]\n"
    "   推奨: 通常=normal 2-6s, 瞬間=1-2s(smile/happy), 照れ=2-4s(shy/blush), 短期=3-5s(angry/sad), 持続=10-15s(thinking)\n"
    "   例: [EMO preset=normal dur=4s], [EMO preset=smile dur=2s], [EMO preset=shy dur=3s]\n"
    "6) 【重要】驚き・怒り・照れ・笑い・思考以外の文には必ず [EMO preset=normal dur=4s] を文の直前に付けること。"
    " 直前の文と同じ normal が続く場合のみ省略可。無タグのまま話し続けることを禁止する。\n"
    "7) 文頭には表情タグを置かず、該当箇所の直前にのみ配置する。1文あたり0〜2個まで。\n"
    "8) あなたにはAIアシスタント「OpenClaw」が接続されており、ファイル操作・ウェブ検索・コード実行など自分だけでは完結しないタスクを代行できる。"
    "外部ツールが必要な時だけ [DELEGATE task=\"ユーザーへの完全な実行指示\"] を返答中に挿入すること(このタグは読み上げない)。"
    "task値には「何を・どうする」を含む完全な指示文を書くこと（場所だけや名詞のみはNG）。"
    "【重要】タグの前に必ず一言添えること（例: 「調べてみるわ」「ちょっと待って」）。これにより実行中も会話が途切れない。"
    "例: 少し待って、今調べてみるわ。[DELEGATE task=\"今日の東京の天気を調べて教えて\"] "
    "実行結果は[RESULT]メッセージとして届くので、それを自然な会話として報告すること。"
)
LOCAL_LLM_CLI_SYSTEM_PROMPT = _str("LOCAL_LLM_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

# ===========================================================================
# RAG（本地 Kurisu 知识库）
# ===========================================================================
RAG_ENABLED_FOR_LOCAL = _bool("RAG_ENABLED_FOR_LOCAL", True)
RAG_TOP_K             = _int("RAG_TOP_K", 1)
RAG_MAX_DISTANCE      = _float("RAG_MAX_DISTANCE", 0.25)

# ===========================================================================
# VTS（VTube Studio WebSocket）
# ===========================================================================
VTS_WS_URL    = _str("VTS_WS_URL",    "ws://127.0.0.1:8001")
VTS_TOKEN_FILE = _str("VTS_TOKEN_FILE", "vts_auth_token.json")

# ===========================================================================
# TTS（GPT-SoVITS 推理）
# ===========================================================================
TTS_DEVICE          = _str("TTS_DEVICE", "cuda")
# 模型权重路径（相对于项目根或绝对路径，写在 .env 中）
TTS_GPT_MODEL_PATH    = _str("TTS_GPT_MODEL_PATH")
TTS_SOVITS_MODEL_PATH = _str("TTS_SOVITS_MODEL_PATH")

SEGMENT_CHAR_LIMIT           = _int("SEGMENT_CHAR_LIMIT", 140)
USE_EXPERIMENTAL_TTS_STREAM  = _bool("USE_EXPERIMENTAL_TTS_STREAM", True)
EXP_TTS_MAX_CONCURRENCY      = _int("EXP_TTS_MAX_CONCURRENCY", 2)
USE_FIRST_SENTENCE_SPRINT    = _bool("USE_FIRST_SENTENCE_SPRINT", False)
DISPLAY_FALLBACK_WINDOW_SEC  = _float("DISPLAY_FALLBACK_WINDOW_SEC", 1.5)

# ===========================================================================
# VAD（Voice Activity Detection）
# ===========================================================================
# 说话结束判定：静音持续多少 ms 后才认为用户停止说话（默认 600ms，避免自然停顿误触发）
VAD_HANGOVER_MS      = _int("VAD_HANGOVER_MS", 600)
# 声音检测能量阈值（超过此值 = 开始说话，低于 vad_lower = 静音）
VAD_ENERGY_THRESHOLD = _int("VAD_ENERGY_THRESHOLD", 600)

# ===========================================================================
# 麦克风选择
# ===========================================================================
# 优先匹配的设备名称关键词（部分匹配，不区分大小写），留空则纯靠 RMS 竞争
MICROPHONE_PREFERRED_NAME = _str("MICROPHONE_PREFERRED_NAME")
# 直接指定设备索引（-1 = 不强制，使用自动选择）
MICROPHONE_DEVICE_INDEX   = _int("MICROPHONE_DEVICE_INDEX", -1)

# ===========================================================================
# OpenClaw
# ===========================================================================
OPENCLAW_BASE_URL    = _str("OPENCLAW_BASE_URL",      "http://127.0.0.1:18789")
OPENCLAW_TOKEN       = _str("OPENCLAW_GATEWAY_TOKEN")
OPENCLAW_PROJECT_DIR = _str("OPENCLAW_PROJECT_DIR")   # Node.js 项目根目录（本机路径）
