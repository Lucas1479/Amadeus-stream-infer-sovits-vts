"""llm/client.py — LLM 客户端层（同步非流式）

负责：
  - 客户端初始化（init_llm_client）
  - 远程 API 查询（remote_llm_query：DeepSeek / Gemini / AWS Bedrock）
  - 本地模型查询（local_llm_query：Ollama / LM Studio / llama-server / CLI）

依赖注入（configure()）：
  - llm_provider : str，覆盖默认 LLM_PROVIDER
"""

import json
import logging
import traceback

import requests
from openai import OpenAI
import google.generativeai as google_genai

from config.settings import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    GEMINI_API_KEY, GEMINI_MODEL_NAME,
    AWS_BEDROCK_BEARER_TOKEN, AWS_BEDROCK_REGION,
    AWS_BEDROCK_MODEL_ID, AWS_BEDROCK_USE_INFERENCE_PROFILE,
    AWS_BEDROCK_INFERENCE_PROFILE_ID, AWS_BEDROCK_ENDPOINT,
    AWS_BEDROCK_USE_CACHE, AWS_BEDROCK_CACHE_TTL,
    AWS_BEDROCK_CONNECTION_POOL_SIZE, AWS_BEDROCK_MAX_KEEPALIVE,
    AWS_BEDROCK_KEEPALIVE_EXPIRY,
    LOCAL_LLM_TYPE, LOCAL_LLM_MODEL,
    LOCAL_LLM_URL, LM_STUDIO_URL,
)
from llm.local_cli import local_llm_query_cli

logger = logging.getLogger(__name__)

# ===== 运行时状态 =====
LLM_PROVIDER: str = "deepseek"
llm_client = None
gemini_model = None
bedrock_http_client = None
bedrock_runtime_client = None


def configure(llm_provider: str = None):
    """设置运行时 LLM_PROVIDER。在程序启动（解析完 CLI args）后调用。"""
    global LLM_PROVIDER
    if llm_provider is not None:
        LLM_PROVIDER = llm_provider


# =============================================================================
# 客户端初始化
# =============================================================================

def init_llm_client():
    """Initialize the appropriate LLM client based on LLM_PROVIDER."""
    global llm_client, gemini_model, bedrock_http_client, bedrock_runtime_client

    if LLM_PROVIDER == "deepseek":
        logger.info("🚀 Initializing DeepSeek LLM client with connection pool")
        import httpx
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(30.0),
            http2=False,
        )
        llm_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            http_client=http_client,
        )
        logger.info("🚀 DeepSeek客户端已配置连接池，SSL握手延迟将显著减少")
        return llm_client

    elif LLM_PROVIDER == "gemini":
        logger.info("Initializing Gemini LLM client")
        google_genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = google_genai.GenerativeModel(GEMINI_MODEL_NAME)
        return gemini_model

    elif LLM_PROVIDER == "bedrock":
        logger.info("Initializing AWS Bedrock client")

        if not AWS_BEDROCK_BEARER_TOKEN:
            logger.error("❌ AWS Bedrock Bearer Token未设置")
            return None

        if bedrock_http_client is None:
            try:
                import httpx
                bedrock_http_client = httpx.Client(
                    limits=httpx.Limits(
                        max_connections=AWS_BEDROCK_CONNECTION_POOL_SIZE,
                        max_keepalive_connections=AWS_BEDROCK_MAX_KEEPALIVE,
                        keepalive_expiry=AWS_BEDROCK_KEEPALIVE_EXPIRY,
                    ),
                    timeout=httpx.Timeout(30.0),
                    http2=False,
                )
                logger.info(
                    f"✅ AWS Bedrock连接池已初始化: "
                    f"最大连接数={AWS_BEDROCK_CONNECTION_POOL_SIZE}, "
                    f"保持连接数={AWS_BEDROCK_MAX_KEEPALIVE}"
                )
            except Exception as e:
                logger.warning(f"⚠️ 初始化Bedrock连接池失败，将使用默认HTTP客户端: {e}")
                bedrock_http_client = None

        if bedrock_runtime_client is None:
            try:
                import boto3
                bedrock_runtime_client = boto3.client(
                    "bedrock-runtime", region_name=AWS_BEDROCK_REGION
                )
                logger.info("✅ AWS Bedrock boto3客户端已初始化（将复用HTTP连接）")
            except Exception as e:
                logger.warning(f"⚠️ 初始化Bedrock boto3客户端失败，将在调用时退回临时客户端: {e}")
                bedrock_runtime_client = None

        if AWS_BEDROCK_USE_INFERENCE_PROFILE and AWS_BEDROCK_INFERENCE_PROFILE_ID:
            model_id = AWS_BEDROCK_INFERENCE_PROFILE_ID
            logger.info(f"✅ AWS Bedrock配置完成（使用Inference Profile）")
            logger.info(f"   区域: {AWS_BEDROCK_REGION}")
            logger.info(f"   Inference Profile ID: {model_id}")
        else:
            model_id = AWS_BEDROCK_MODEL_ID
            logger.info(f"✅ AWS Bedrock配置完成（按需使用模式）")
            logger.info(f"   区域: {AWS_BEDROCK_REGION}")
            logger.info(f"   模型ID: {model_id}")
            if AWS_BEDROCK_USE_INFERENCE_PROFILE:
                logger.warning("   ⚠️ 已启用Inference Profile模式，但未设置Profile ID，将尝试使用模型ID")

        if AWS_BEDROCK_USE_CACHE:
            logger.info(f"   系统prompt缓存: 已启用 (TTL: {AWS_BEDROCK_CACHE_TTL}秒)")
        else:
            logger.info("   系统prompt缓存: 已禁用")

        return "bedrock_client"

    else:
        logger.error(f"Unknown LLM provider: {LLM_PROVIDER}")
        return None


# =============================================================================
# 远程 API 查询（同步，非流式）
# =============================================================================

_SYSTEM_PROMPT_BASE = (
    "あなたは牧瀬紅莉栖.日本の科学者であり,母語は日本語です.\n\n"
    "【絶対遵守】\n"
    "1) 必ず日本語でのみ回答すること.ユーザーの言語が何であっても,日本語以外は一切使用しない.\n"
    "2) 中国語の文字・語句を絶対に使用しない.\n"
    "3) 自然で口語的な文体を保ち,牧瀬紅莉栖として一貫した口調・性格で話す.\n"
    "4) 推論過程や思考の連鎖は開示しない(結論のみ提示).\n"
    "5) 表情タグの活用ガイド（自然な感情表現のため、適切に使用）:\n"
    "   形式: [EMO preset=<種類> dur=<秒s>] / [EXPR name=<表情> weight=<0..1> dur=<秒s> fade=<秒> active=true|false]\n"
    "   推奨: 通常=normal 2-6s, 瞬間=1-2s(smile/happy), 照れ=2-4s(shy/blush), 短期=3-5s(angry/sad), 持続=10-15s(thinking)\n"
    "   例: [EMO preset=normal dur=4s], [EMO preset=thinking dur=12s], [EMO preset=shy dur=3s], [EMO preset=smile dur=2s]\n"
    "6) 【重要】特定の強い感情（驚き・怒り・照れ・笑い・思考）がない限り, 必ず [EMO preset=normal dur=4s] を文の直前に付けること."
    " EMO タグを省略してよいのは, 直前の文と同じ normal が連続する場合のみ. 無タグのまま話し続けることを禁止する.\n"
    "7) 恥ずかしい・照れ・赤面・視線回避の文脈では shy を優先し, blush は軽度な照れのみで使う. shy / blush を通常会話で濫用しない.\n"
    "8) 1文あたり0〜2個, 文頭には置かず, 該当箇所の直前にのみ配置する。"
)

_SYSTEM_PROMPT_WITH_DELEGATE = (
    _SYSTEM_PROMPT_BASE
    + "\n9) あなたにはAIアシスタント「OpenClaw」が接続されており、ファイル操作・ウェブ検索・コード実行など"
    "自分だけでは完結しないタスクを代行できる。外部ツールが必要な時だけ "
    "[DELEGATE task=\"ユーザーへの完全な実行指示\"] を返答中に挿入すること(このタグは読み上げない)。"
    "task値には「何を・どうする」を含む完全な指示文を書くこと（場所だけや名詞のみはNG）。"
    "【重要】タグの前に必ず一言添えること（例:「調べてみるわ」「ちょっと待って」）。"
    "これにより実行中も会話が途切れない。"
    "例: 少し待って、今調べてみるわ。[DELEGATE task=\"今日の東京の天気を調べて教えて\"] "
    "実行結果は[RESULT]メッセージとして届くので、それを自然な会話として報告すること。"
)


def remote_llm_query(question: str) -> str:
    """Call online API (DeepSeek, Gemini, or AWS Bedrock), with enhanced error handling."""
    global llm_client, gemini_model

    try:
        if LLM_PROVIDER == "deepseek" and llm_client is None:
            llm_client = init_llm_client()
        elif LLM_PROVIDER == "gemini" and gemini_model is None:
            gemini_model = init_llm_client()
        elif LLM_PROVIDER == "bedrock":
            init_llm_client()

        logger.info(f"Sending API request to {LLM_PROVIDER}...")

        # ── DeepSeek ──────────────────────────────────────────────────────────
        if LLM_PROVIDER == "deepseek":
            response = llm_client.chat.completions.create(
                model="deepseek-v3-250324",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT_BASE},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=500,
                stream=False,
                timeout=5,
            )
            if not response or not hasattr(response, "choices") or not response.choices:
                logger.warning("⚠️ DeepSeek API returned invalid response")
                return "APIからの応答が無効です."
            reply = response.choices[0].message.content

        # ── Gemini ────────────────────────────────────────────────────────────
        elif LLM_PROVIDER == "gemini":
            if gemini_model is None:
                logger.info("Initializing Gemini LLM client")
                google_genai.configure(api_key=GEMINI_API_KEY)
                gemini_model = google_genai.GenerativeModel(GEMINI_MODEL_NAME)
            full_prompt = f"{_SYSTEM_PROMPT_WITH_DELEGATE}\n\n質問:{question}"
            generation_config = {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 1000,
            }
            try:
                response = gemini_model.generate_content(
                    full_prompt, generation_config=generation_config
                )
                if not response or not hasattr(response, "text"):
                    logger.warning("⚠️ Gemini API returned invalid response")
                    return "Gemini APIからの応答が無効です."
                reply = response.text
                logger.info(f"✓ Gemini API response successful, length: {len(reply)}")
                return reply
            except Exception as e:
                logger.error(f"❌ Gemini API error: {str(e)}")
                return f"Gemini APIエラー:{str(e)}"

        # ── AWS Bedrock ───────────────────────────────────────────────────────
        elif LLM_PROVIDER == "bedrock":
            system_prompt = (
                _SYSTEM_PROMPT_WITH_DELEGATE
                + "\n10) 通常の応答は,ユーザーの質問に直接答えることを優先し,不要な自己紹介・挨拶・雑談を追加しない.\n"
                "11) 解説が必要な科学的定義や技術的内容では,必要な範囲で段階的に説明してよいが,同じ内容を言い換えて何度も繰り返さない.\n"
                "12) 1ターンの発話は,原則として簡潔なまとまり(目安として日本語で数文程度)に収めること.\n"
                "    ユーザーが特別に『もっと詳しく』と依頼した場合のみ,例や詳細説明を追加してよい.\n"
                "13) 会話の最後に,ユーザーが求めていない新しい質問を投げて会話を引き延ばさない.\n"
                "【EMO再確認】感情タグのない文が連続することを厳禁する."
                " 驚き・怒り・照れ・笑い・思考以外の文には必ず [EMO preset=normal dur=4s] を付けること.\n"
            )
            if AWS_BEDROCK_USE_INFERENCE_PROFILE and AWS_BEDROCK_INFERENCE_PROFILE_ID:
                model_id = AWS_BEDROCK_INFERENCE_PROFILE_ID
            else:
                model_id = AWS_BEDROCK_MODEL_ID
            try:
                try:
                    import boto3
                    global bedrock_runtime_client
                    if bedrock_runtime_client is None:
                        bedrock_runtime_client = boto3.client(
                            "bedrock-runtime", region_name=AWS_BEDROCK_REGION,
                            aws_access_key_id=None, aws_secret_access_key=None,
                        )
                    payload = {
                        "max_tokens": 500,
                        "temperature": 0.7,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                    }
                    response = bedrock_runtime_client.invoke_model(
                        modelId=model_id, body=json.dumps(payload)
                    )
                    result = json.loads(response["body"].read())
                    if "content" in result and len(result["content"]) > 0:
                        reply = result["content"][0]["text"]
                    else:
                        logger.warning(f"⚠️ Bedrock API returned invalid response: {result}")
                        return "Bedrock APIからの応答が無効です."
                    logger.info(f"✓ Bedrock API response successful (boto3), reply length: {len(reply)}")
                    return reply
                except ImportError:
                    logger.warning("⚠️ boto3未安装，使用HTTP方式调用Bedrock API")
                except Exception as boto_error:
                    logger.warning(f"⚠️ boto3调用失败: {boto_error}，尝试HTTP方式")

                # HTTP 降级
                url = f"{AWS_BEDROCK_ENDPOINT}/model/{model_id}/invoke"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {AWS_BEDROCK_BEARER_TOKEN}",
                }
                payload = {
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                }
                global bedrock_http_client
                if bedrock_http_client is not None:
                    response = bedrock_http_client.post(url, headers=headers, json=payload, timeout=30)
                else:
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"❌ Bedrock API错误 {response.status_code}: {error_detail}")
                    return f"Bedrock APIエラー: {response.status_code} - {error_detail[:200]}"
                result = response.json()
                if "content" in result and len(result["content"]) > 0:
                    reply = result["content"][0]["text"]
                else:
                    logger.warning(f"⚠️ Bedrock API returned invalid response: {result}")
                    return "Bedrock APIからの応答が無効です."
                logger.info(f"✓ Bedrock API response successful (HTTP), reply length: {len(reply)}")
                return reply
            except Exception as e:
                logger.error(f"❌ Bedrock API error: {str(e)}")
                logger.error(f"   错误详情: {traceback.format_exc()}")
                return f"Bedrock APIエラー:{str(e)}"

        logger.info(f"✓ {LLM_PROVIDER} API response successful, reply length: {len(reply)}")
        return reply

    except Exception as e:
        logger.error(f"❌ Failed to call online LLM ({LLM_PROVIDER}): {str(e)}")
        return "すみません,今ちょっと調子が悪いです……."


# =============================================================================
# 本地模型查询（同步，非流式）
# =============================================================================

def local_llm_query(question: str) -> str:
    """调用本地模型(Ollama / LM Studio / llama-server / CLI) - 非流式版本"""
    try:
        _system = (
            "あなたは牧瀬紅莉栖で,優秀で理知的な性格です."
            "少しツンデレで,でも根は優しい.日本語で自然に答えてください."
        )

        if LOCAL_LLM_TYPE == "ollama":
            payload = {
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": _system},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "temperature": 0.7,
            }
            response = requests.post(f"{LOCAL_LLM_URL}/api/chat", json=payload, timeout=20)
            response.raise_for_status()
            reply = response.json()["message"]["content"]

        elif LOCAL_LLM_TYPE == "lmstudio":
            payload = {
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": _system},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "temperature": 0.7,
            }
            response = requests.post(f"{LM_STUDIO_URL}/v1/chat/completions", json=payload, timeout=20)
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"]

        elif LOCAL_LLM_TYPE == "cli":
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            reply = loop.run_until_complete(local_llm_query_cli(question, stream=False))

        elif LOCAL_LLM_TYPE == "llama_server":
            base_url = LOCAL_LLM_URL.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            payload = {
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": _system},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "temperature": 0.7,
                "cache_prompt": True,
            }
            response = requests.post(f"{base_url}/chat/completions", json=payload, timeout=20)
            response.raise_for_status()
            raw_reply = response.json()["choices"][0]["message"]["content"]
            import re as _re
            reply = _re.sub(r"<think>.*?</think>", "", raw_reply, flags=_re.DOTALL).strip()

        else:
            raise ValueError(f"未知的本地LLM类型: {LOCAL_LLM_TYPE}")

        logger.info(f"💬 本地LLM回答: {reply}")
        return reply

    except Exception as e:
        logger.error(f"❌ 调用本地LLM失败: {e}")
        return "(ローカルモデルの応答に失敗しました……)"
