"""
VTube Studio WebSocket 连接管理
- VTSConnectionManager：连接/鉴权/重连/口型/表情/热键指令封装
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from threading import Lock, Thread

import websocket as ws

logger = logging.getLogger(__name__)


class VTSConnectionManager:
    def __init__(self, ws_url: str, token_file: str = None):
        self.ws_url = ws_url
        self.ws = None
        self.ws_lock = Lock()
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.message_buffer = []
        self.auth_token = None
        self.ws_app = None

        # token 文件路径，默认从 config.settings 读取
        if token_file is None:
            try:
                from config.settings import VTS_TOKEN_FILE
                token_file = VTS_TOKEN_FILE
            except Exception:
                token_file = "vts_token.json"
        self._token_file = token_file

    # ------------------------------------------------------------------
    # Token 持久化
    # ------------------------------------------------------------------
    def _load_auth_token(self) -> str | None:
        if os.path.exists(self._token_file):
            try:
                with open(self._token_file, "r") as f:
                    token_data = json.load(f)
                    self.auth_token = token_data.get("authenticationToken")
                    return self.auth_token
            except Exception as e:
                logger.error(f"读取Token文件失败: {e}")
        return None

    def _save_auth_token(self, token: str) -> None:
        try:
            with open(self._token_file, "w") as f:
                json.dump({"authenticationToken": token}, f)
            logger.info("✅ Token已保存到文件")
        except Exception as e:
            logger.error(f"保存Token失败: {e}")

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        if self.connected and self.ws and self.ws.connected:
            return True
        self.disconnect()
        try:
            self.ws = ws.WebSocket()
            self.ws.settimeout(10)
            self.ws.connect(
                self.ws_url,
                skip_utf8_validation=True,
                suppress_origin=True,
                enable_multithread=True,
            )
            logger.info("✅ VTS WebSocket连接成功")
            self.connected = True
            self.reconnect_attempts = 0

            if not self.auth_token:
                self.auth_token = self._load_auth_token()

            if self.auth_token:
                self.authenticate()
            else:
                self.request_auth_token()

            self._send_buffered_messages()
            return True
        except Exception as e:
            self._handle_connection_error(e)
            return False

    def disconnect(self) -> None:
        with self.ws_lock:
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass
                self.ws = None
            self.connected = False

    def _handle_connection_error(self, error) -> None:
        self.connected = False
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"❌ 达到最大重连次数({self.max_reconnect_attempts})，放弃连接")
            return
        backoff = min(0.5 * (2 ** self.reconnect_attempts) + random.uniform(0, 1), 30)
        self.reconnect_attempts += 1
        logger.warning(
            f"⚠️ VTS连接失败: {error}. 将在{backoff:.2f}秒后尝试重连 "
            f"(尝试 {self.reconnect_attempts}/{self.max_reconnect_attempts})"
        )
        time.sleep(backoff)
        Thread(target=self.connect, daemon=True).start()

    # ------------------------------------------------------------------
    # 消息收发
    # ------------------------------------------------------------------
    def send_message(self, payload: dict) -> bool:
        if not self.connected or not self.ws:
            logger.info("🔄 WebSocket未连接，尝试连接...")
            if not self.connect():
                logger.info("📩 连接失败，消息已加入缓冲区")
                self.message_buffer.append(payload)
                return False
        try:
            with self.ws_lock:
                self.ws.send(json.dumps(payload))
            return True
        except Exception as e:
            logger.warning(f"📤 消息发送失败: {e}")
            self.message_buffer.append(payload)
            self._handle_connection_error(e)
            return False

    def _send_buffered_messages(self) -> None:
        if not self.message_buffer:
            return
        logger.info(f"📤 发送缓冲区中的{len(self.message_buffer)}条消息")
        messages_to_send = self.message_buffer.copy()
        self.message_buffer.clear()
        for msg in messages_to_send:
            self.send_message(msg)

    def receive_message(self, timeout: float = 1) -> dict | None:
        if not self.connected or not self.ws:
            return None
        try:
            with self.ws_lock:
                self.ws.settimeout(timeout)
                return json.loads(self.ws.recv())
        except ws.WebSocketTimeoutException:
            return None
        except Exception as e:
            logger.warning(f"📥 接收消息失败: {e}")
            self._handle_connection_error(e)
            return None

    # ------------------------------------------------------------------
    # 认证
    # ------------------------------------------------------------------
    def request_auth_token(self) -> str | None:
        logger.info("🔑 请求新的VTS认证Token...")
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "get_token",
            "messageType": "AuthenticationTokenRequest",
            "data": {"pluginName": "Lucas-Plugin", "pluginDeveloper": "Lucas"},
        }
        if self.send_message(payload):
            for _ in range(10):
                response = self.receive_message(timeout=0.5)
                if response and response.get("messageType") == "AuthenticationTokenResponse":
                    token = response.get("data", {}).get("authenticationToken")
                    if token:
                        self.auth_token = token
                        self._save_auth_token(token)
                        logger.info("✅ VTS Token请求成功!")
                        return token
            logger.error("❌ VTS Token响应超时或无效")
        return None

    def authenticate(self) -> bool:
        if not self.auth_token:
            logger.warning("⚠️ 没有可用的认证Token")
            return False
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth_lucas",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": "Lucas-Plugin",
                "pluginDeveloper": "Lucas",
                "authenticationToken": self.auth_token,
            },
        }
        if self.send_message(payload):
            logger.info("🔐 VTS认证请求已发送")
            for _ in range(5):
                response = self.receive_message(timeout=0.5)
                if response and response.get("messageType") == "AuthenticationResponse":
                    if response.get("data", {}).get("authenticated", False):
                        logger.info("✅ VTS认证成功!")
                        return True
                    else:
                        logger.warning("❌ VTS认证失败，将尝试请求新Token")
                        self.auth_token = None
                        return self.request_auth_token() is not None
            logger.warning("⚠️ VTS认证响应超时")
        return False

    # ------------------------------------------------------------------
    # 控制指令
    # ------------------------------------------------------------------
    def send_mouth_data(self, mouth_value: float) -> bool:
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "mouth_sync",
            "messageType": "InjectParameterDataRequest",
            "data": {"parameterValues": [{"id": "MouthOpen", "value": mouth_value}]},
        }
        return self.send_message(payload)

    def send_parameters(self, param_dict: dict) -> bool:
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "param_batch",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "parameterValues": [
                    {"id": pid, "value": float(val)} for pid, val in param_dict.items()
                ]
            },
        }
        return self.send_message(payload)

    def trigger_hotkey(self, name_or_id: str) -> bool:
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "hotkey_trigger",
            "messageType": "HotkeyTriggerRequest",
            "data": {"hotkeyID": name_or_id, "hotkeyName": name_or_id},
        }
        return self.send_message(payload)

    def activate_expression(
        self,
        name_or_file: str,
        active: bool = True,
        fade_time: float = 0.0,
        weight: float = None,
    ) -> bool:
        data: dict = {"active": bool(active), "fadeTime": float(fade_time)}
        if name_or_file.lower().endswith(".exp3.json"):
            data["expressionFile"] = name_or_file
        else:
            data["expressionName"] = name_or_file
        if weight is not None:
            try:
                data["weight"] = float(weight)
            except Exception:
                pass
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "expr_activate",
            "messageType": "ExpressionActivationRequest",
            "data": data,
        }
        logger.info(f"发送表情激活请求: {payload}")
        return self.send_message(payload)

    def send_heartbeat(self) -> bool:
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "heartbeat",
            "messageType": "APIStateRequest",
            "data": {},
        }
        return self.send_message(payload)
