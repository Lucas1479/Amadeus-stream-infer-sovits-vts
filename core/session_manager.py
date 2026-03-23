"""
会话管理模块
- ConversationHistory：对话历史维护（滚动窗口 + token 估算 + 摘要触发）
- 会话持久化 CRUD（JSON 文件存储）

注意：ENABLE_CONVERSATION 运行时开关保留在 main.py，通过参数传入 save_session / load_session，
避免与 chatGui.py 对 main_module.ENABLE_CONVERSATION 的直接属性访问产生冲突。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------
class ConversationHistory:
    def __init__(self, max_rounds: int = 10, summary_token_threshold: int = 3000):
        self.dialog = []  # {"role": "user"|"assistant", "content": str}
        self.max_rounds = max_rounds
        self.summary_token_threshold = summary_token_threshold
        self._summary_requested_flag = False
        self.last_summary = ""

    def reset(self):
        self.dialog.clear()
        self._summary_requested_flag = False

    def _estimate_tokens(self, text: str) -> int:
        return len(text or "")

    def total_tokens(self) -> int:
        return sum(self._estimate_tokens(m.get("content", "")) for m in self.dialog)

    def add_user(self, content: str):
        if not content:
            return
        self.dialog.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str):
        if not content:
            return
        self.dialog.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self):
        max_items = max(2, self.max_rounds * 2)
        if len(self.dialog) > max_items:
            self.dialog = self.dialog[-max_items:]

    def should_request_summary(self) -> bool:
        return (self.total_tokens() >= self.summary_token_threshold) and (not self._summary_requested_flag)

    def mark_summary_requested(self):
        self._summary_requested_flag = True

    def build_deepseek_messages(self, system_prompt: str, latest_user: str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for m in self.dialog:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": latest_user})
        if self.should_request_summary():
            messages.append({
                "role": "user",
                "content": "ここまでの会話を100~150字で要約してください。要約部分は必ず[SUMMARY]と[/SUMMARY]で厳密に囲んでください。その後に通常の応答を続けてください。"
            })
            self.mark_summary_requested()
        return messages

    def build_gemini_full_prompt(self, system_prompt: str, latest_user: str) -> str:
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        for m in self.dialog:
            prefix = "ユーザー:" if m["role"] == "user" else "アシスタント:"
            parts.append(f"{prefix}{m['content']}")
        parts.append(f"質問:{latest_user}")
        if self.should_request_summary():
            parts.append("補足指示: ここまでの会話を100~150字で要約し、その要約は必ず[SUMMARY]と[/SUMMARY]で囲んでください。続けて通常の応答を返してください。")
            self.mark_summary_requested()
        return "\n\n".join(parts)


# 全局单例
conversation_history = ConversationHistory(max_rounds=10, summary_token_threshold=3000)

# ---------------------------------------------------------------------------
# 会话持久化
# ---------------------------------------------------------------------------
_SESSION_DIR = os.path.join(os.getcwd(), "sessions")
_CURRENT_SESSION_ID: str | None = None


def _ensure_session_dir():
    try:
        os.makedirs(_SESSION_DIR, exist_ok=True)
    except Exception:
        pass


def list_sessions() -> list[str]:
    _ensure_session_dir()
    try:
        files = [f for f in os.listdir(_SESSION_DIR) if f.endswith(".json")]
        return sorted([os.path.splitext(f)[0] for f in files])
    except Exception:
        return []


def _session_path(session_id: str) -> str:
    _ensure_session_dir()
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id or "default")
    return os.path.join(_SESSION_DIR, f"{safe}.json")


def create_session(session_id: str) -> str:
    global _CURRENT_SESSION_ID
    if not session_id:
        session_id = time.strftime("%Y%m%d-%H%M%S")
    _CURRENT_SESSION_ID = session_id
    conversation_history.reset()
    save_session(session_id)
    return session_id


def save_session(session_id: str = None, *, enable_conversation: bool = False):
    """
    持久化当前会话到 JSON 文件。

    参数：
        enable_conversation: 当前 ENABLE_CONVERSATION 运行时状态，由调用方传入。
    """
    sid = session_id or _CURRENT_SESSION_ID
    if not sid:
        return
    existing_title = None
    path = _session_path(sid)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing_title = json.load(f).get("title")
        except Exception:
            pass
    data = {
        "session_id": sid,
        "dialog": conversation_history.dialog,
        "last_summary": getattr(conversation_history, "last_summary", ""),
        "max_rounds": conversation_history.max_rounds,
        "summary_token_threshold": conversation_history.summary_token_threshold,
        "enable_conversation": enable_conversation,
        "timestamp": time.time(),
    }
    if existing_title:
        data["title"] = existing_title
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存会话失败: {e}")


def load_session(session_id: str) -> tuple[bool, bool]:
    """
    从 JSON 文件加载会话，返回 (success, enable_conversation)。
    调用方负责将 enable_conversation 写回自己的全局变量。
    """
    global _CURRENT_SESSION_ID
    path = _session_path(session_id)
    if not os.path.exists(path):
        return False, False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        conversation_history.dialog = data.get("dialog", [])
        conversation_history.last_summary = data.get("last_summary", "")
        conversation_history.max_rounds = int(data.get("max_rounds", conversation_history.max_rounds))
        conversation_history.summary_token_threshold = int(
            data.get("summary_token_threshold", conversation_history.summary_token_threshold)
        )
        _CURRENT_SESSION_ID = data.get("session_id", session_id)
        return True, bool(data.get("enable_conversation", False))
    except Exception as e:
        logger.error(f"加载会话失败: {e}")
        return False, False


def delete_session(session_id: str) -> bool:
    try:
        path = _session_path(session_id)
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
    return False


def rename_session(old_id: str, new_id: str) -> bool:
    try:
        old_path = _session_path(old_id)
        new_path = _session_path(new_id)
        if os.path.exists(old_path):
            os.replace(old_path, new_path)
            return True
    except Exception as e:
        logger.error(f"重命名会话失败: {e}")
    return False


def get_current_session_id() -> str | None:
    return _CURRENT_SESSION_ID


def set_current_session_id(session_id: str) -> None:
    global _CURRENT_SESSION_ID
    _CURRENT_SESSION_ID = session_id


def get_session_title(session_id: str) -> str:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return session_id
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("title") or session_id
    except Exception:
        return session_id


def set_session_title(session_id: str, title: str) -> bool:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["title"] = title
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"更新会话标题失败: {e}")
        return False


async def generate_session_title(first_user_message: str) -> str:
    """调用 DeepSeek API 为当前会话生成简短标题。"""
    try:
        from openai import AsyncOpenAI
        from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = await client.chat.completions.create(
            model="deepseek-v3-250324",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个对话标题生成助手。根据用户的第一条消息，"
                        "用简洁的短语（中文不超过12个字，英文不超过6个单词）概括对话主题。"
                        "只返回标题本身，不加引号、不加解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"为以下消息生成对话标题：{first_user_message[:300]}",
                },
            ],
            max_tokens=40,
            temperature=0.3,
        )
        title = resp.choices[0].message.content.strip().strip('"\'「」《》')
        return title[:30]
    except Exception as e:
        logger.error(f"生成会话标题失败: {e}")
        return ""
