"""render/engine.py — PixiJS 渲染引擎封装

支持两种后端（可同时运行）：
  - "sprite"  : 静态帧动画，类似 Kur1oR3/AMDS-RE 的 kurisu_*.png 方案
  - "live2d"  : pixi-live2d-display 驱动的 Live2D Cubism 模型

主进程通过 set_emotion() / set_speaking() 等方法驱动渲染器；
内部通过 QWebEngineView.runJavaScript() 将命令传入 JS。

QWebEngineView 需要 PyQtWebEngine 包：
    pip install PyQtWebEngine
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 项目根目录（render/ 的父目录）
_PROJECT_ROOT = Path(__file__).parent.parent

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
    from PyQt5.QtCore import QUrl, pyqtSlot
    from PyQt5.QtWidgets import QWidget, QVBoxLayout
    _WEBENGINE_OK = True
except Exception as _e:
    _WEBENGINE_OK = False
    logger.warning(f"[RenderEngine] PyQtWebEngine 不可用: {type(_e).__name__}: {_e}")


class RenderEngine:
    """PixiJS 渲染引擎。

    用法：
        engine = RenderEngine()
        engine.start()                        # 启动 HTTP 服务器 + 创建 WebView
        parent_layout.addWidget(engine.widget)
        engine.load_kur1or3_sprites(images_dir)  # 加载全套 kurisu_*.png
        engine.set_emotion("smile")
        engine.set_speaking(True)
    """

    def __init__(self):
        if not _WEBENGINE_OK:
            raise RuntimeError("PyQtWebEngine 未安装，无法创建 RenderEngine")

        from render.server import AssetServer

        self._server = AssetServer(_PROJECT_ROOT)
        self._port: int = -1
        self._view: Optional[QWebEngineView] = None
        self._widget: Optional[QWidget] = None
        self._pending: list[str] = []   # 页面加载前缓存的 JS 调用
        self._ready = False
        self.on_ready: Optional[callable] = None  # 页面加载完成后调用一次

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------

    def start(self) -> QWidget:
        """启动服务器并返回可嵌入主窗口的 QWidget。"""
        self._port = self._server.start()
        logger.info(f"[RenderEngine] AssetServer 监听 http://127.0.0.1:{self._port}/")

        self._view = QWebEngineView()
        # WA_NativeWindow：给 WebEngine 独立的 native window handle，
        # 防止其 OpenGL 上下文污染父窗口的 QPainter（导致 FluentWindow navbar 变黑）
        from PyQt5.QtCore import Qt as _Qt
        self._view.setAttribute(_Qt.WA_NativeWindow, True)
        settings = self._view.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.AllowRunningInsecureContent, True)

        # 页面加载完成后刷新 pending 队列
        self._view.loadFinished.connect(self._on_load_finished)

        url = f"http://127.0.0.1:{self._port}/render/web/index.html"
        self._view.load(QUrl(url))

        # 包裹进 QWidget，方便 addWidget
        self._widget = QWidget()
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        return self._widget

    def stop(self) -> None:
        self._server.stop()

    @property
    def widget(self) -> Optional[QWidget]:
        return self._widget

    # ------------------------------------------------------------------
    # 页面就绪回调
    # ------------------------------------------------------------------

    def _on_load_finished(self, ok: bool) -> None:
        if not ok:
            logger.error("[RenderEngine] 页面加载失败")
            return
        self._ready = True
        logger.info("[RenderEngine] 页面加载完成，刷新 pending 队列")
        for js in self._pending:
            self._view.page().runJavaScript(js)
        self._pending.clear()
        if self.on_ready is not None:
            try:
                self.on_ready()
            except Exception as e:
                logger.error(f"[RenderEngine] on_ready 回调失败: {e}")

    # ------------------------------------------------------------------
    # JS 调用工具
    # ------------------------------------------------------------------

    def _js(self, code: str) -> None:
        if self._ready and self._view:
            self._view.page().runJavaScript(code)
        else:
            self._pending.append(code)

    def _call(self, method: str, *args) -> None:
        """调用 window.renderApp.{method}(...args)"""
        arg_str = ", ".join(json.dumps(a, ensure_ascii=False) for a in args)
        self._js(f"window.renderApp && window.renderApp.{method}({arg_str});")

    # ------------------------------------------------------------------
    # preset key → Kur1oR3 sprite name 映射
    # ExpressionController 传来的是 emotion_presets.json 的 key（经过 _ALIASES 归一化）
    # Kur1oR3 图片按自己的命名规则，两者不完全一致，在此转换。
    # ------------------------------------------------------------------
    _PRESET_TO_SPRITE: dict[str, str] = {
        # preset key   → kurisu_*.png emotion name
        "smile":        "happy",          # kurisu_happy*.png
        "thinking":     "sided_thinking", # kurisu_sided_thinking*.png
        "surprised":    "sided_surprised",# kurisu_sided_surprised*.png
        # 其余 key 与 Kur1oR3 名称相同，不需要映射
    }

    # ------------------------------------------------------------------
    # 公共 API（与 VTSConnectionManager 接口对称）
    # ------------------------------------------------------------------

    def set_emotion(self, emotion: str) -> None:
        """切换表情（sprite 后端切换帧组，live2d 后端切换 expression）。"""
        sprite_name = self._PRESET_TO_SPRITE.get(emotion, emotion)
        self._call("setEmotion", sprite_name)

    def set_speaking(self, speaking: bool) -> None:
        """控制说话动画（sprite 后端循环嘴部帧，live2d 后端驱动 MouthOpenY）。"""
        self._call("setSpeaking", speaking)

    def set_mouth_value(self, value: float) -> None:
        """实时口型值（0.0–1.0）。"""
        self._call("setMouth", value)

    def set_subtitle(self, text: str) -> None:
        """更新字幕文本。"""
        self._call("setSubtitle", text)

    # ------------------------------------------------------------------
    # 资产加载
    # ------------------------------------------------------------------

    def load_live2d_model(self, model_url: str) -> None:
        """加载 Live2D 模型（.model3.json URL 或相对于项目根的路径）。"""
        if not model_url.startswith("http"):
            model_url = f"http://127.0.0.1:{self._port}/{model_url.lstrip('/')}"
        self._call("loadLive2DModel", model_url)

    def load_sprite_frames(self, emotion: str, frame_urls: list[str]) -> None:
        """向 sprite 后端注册某个情绪的帧 URL 列表（1 帧 = 静止，3 帧 = 说话循环）。"""
        urls = [
            f"http://127.0.0.1:{self._port}/{u.lstrip('/')}" if not u.startswith("http") else u
            for u in frame_urls
        ]
        self._call("loadSpriteFrames", emotion, urls)

    def load_kur1or3_sprites(self, images_dir: Path | str) -> None:
        """自动扫描 images_dir 中的 kurisu_*.png 并注册所有情绪帧。

        文件命名规则：kurisu_{emotion}{frame}.png
        例：kurisu_normal1.png, kurisu_smile1.png, kurisu_smile2.png, ...
        """
        import re
        images_dir = Path(images_dir)
        pattern = re.compile(r"kurisu_([a-z_]+?)(\d+)\.png$", re.IGNORECASE)

        emotion_frames: dict[str, list[tuple[int, str]]] = {}
        for p in sorted(images_dir.glob("kurisu_*.png")):
            m = pattern.match(p.name)
            if not m:
                continue
            emotion, frame_idx = m.group(1), int(m.group(2))
            rel = p.relative_to(_PROJECT_ROOT).as_posix()
            emotion_frames.setdefault(emotion, []).append((frame_idx, rel))

        for emotion, frames in emotion_frames.items():
            frames.sort(key=lambda x: x[0])
            urls = [rel for _, rel in frames]
            self.load_sprite_frames(emotion, urls)
            logger.info(f"[RenderEngine] 注册 sprite: {emotion} ({len(urls)} 帧)")

    def set_mode(self, mode: str) -> None:
        """切换渲染模式: 'sprite' | 'live2d' | 'both'"""
        self._call("setMode", mode)
