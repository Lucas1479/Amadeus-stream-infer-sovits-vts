"""render/engine.py — PixiJS 渲染引擎封装

支持四种渲染模式：
  - "sprite"  : 静态帧动画，类似 Kur1oR3/AMDS-RE 的 kurisu_*.png 方案
  - "live2d"  : pixi-live2d-display 驱动的 Live2D Cubism 模型
  - "both"    : sprite + Live2D 同时显示（叠加）
  - "hybrid"  : 智能混合模式 —— 待机时显示 Live2D，说话时自动交叉淡入帧
                动画（200ms ease-out 渐变），说话结束后淡回 Live2D。
                调用 set_hybrid_mode() 启用，无需手动管理 setMode。

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
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings, QWebEnginePage
    from PyQt5.QtCore import QUrl, pyqtSlot, QObject, pyqtSignal
    from PyQt5.QtWidgets import QWidget, QVBoxLayout
    _WEBENGINE_OK = True

    class _ConsolePage(QWebEnginePage):
        """将 JS console.log/warn/error 转发到 Python logger。"""
        _LEVEL_MAP = {
            QWebEnginePage.InfoMessageLevel:    "info",
            QWebEnginePage.WarningMessageLevel: "warning",
            QWebEnginePage.ErrorMessageLevel:   "error",
        }

        def javaScriptConsoleMessage(self, level, message, line, source_id):
            src = source_id.split("/")[-1] if source_id else "?"
            log_fn = getattr(logger, self._LEVEL_MAP.get(level, "debug"))
            log_fn(f"[JS:{src}:{line}] {message}")

    class _MouthRelay(QObject):
        """跨线程传递口型振幅值（音频线程 → Qt 主线程）。
        pyqtSignal 连接默认为 AutoConnection，跨线程时自动变 QueuedConnection，
        比 QMetaObject.invokeMethod 更可靠。
        """
        mouth_value = pyqtSignal(float)

except Exception as _e:
    _WEBENGINE_OK = False
    _ConsolePage = None
    _MouthRelay = None
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

        # 跨线程口型同步：音频线程 emit → Qt 主线程 slot 调用 runJavaScript
        self._mouth_relay = _MouthRelay()
        self._mouth_relay.mouth_value.connect(self._on_mouth_value_main_thread)

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

        # 子类化 QWebEnginePage 以转发 JS console 到 Python logger
        if _ConsolePage is not None:
            page = _ConsolePage(self._view)
            self._view.setPage(page)

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
        # 页面就绪后应用已保存的 Live2D 布局（在模型加载之前设置，_fitToStage 会读取）
        self._apply_saved_live2d_layout()
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
        "shy":          "blush",          # kurisu_blush_in/loop/out（目录化资产）
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
        """口型振幅驱动（0.0–1.0），由音频播放线程高频调用。
        通过 pyqtSignal（AutoConnection）将值投递到 Qt 主线程，
        比 QMetaObject.invokeMethod 更可靠，完全线程安全。
        """
        v = float(max(0.0, min(1.0, value)))
        # emit 是线程安全的；若页面未就绪 slot 内会静默跳过
        self._mouth_relay.mouth_value.emit(v)

    def _on_mouth_value_main_thread(self, v: float) -> None:
        """Qt 主线程中执行，将口型值注入 JS。由 _mouth_relay 信号触发。"""
        if self._ready and self._view:
            self._view.page().runJavaScript(
                f"window.renderApp && window.renderApp.setMouth({v});"
            )

    def set_subtitle(self, text: str) -> None:
        """更新字幕文本。"""
        self._call("setSubtitle", text)

    def load_mouth_config(self, sprite_name: str, config: dict) -> None:
        """推送单个 sprite 的口型配置到 JS 渲染层。
        sprite_name 为素材目录名（blush/normal/angry 等），
        config 来自 config/mouth_masks.json 中 expressions[sprite_name]。
        """
        self._call("loadMouthConfig", str(sprite_name), config)

    def load_all_mouth_configs(self, configs: dict) -> None:
        """批量推送口型配置，configs 为 {sprite_name: config} 字典。"""
        for sprite_name, cfg in configs.items():
            self.load_mouth_config(sprite_name, cfg)

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

    def load_sprite_clip_frames(
        self,
        emotion: str,
        in_frame_urls: list[str],
        loop_frame_urls: list[str],
        out_frame_urls: list[str],
    ) -> None:
        """注册三段式说话动画帧（in/loop/out）。

        约定：
          - in   : 进入说话过渡（可选）
          - loop : 说话循环主体（至少 1 帧）
          - out  : 退出返回 idle（可选）
        """
        to_abs = lambda urls: [
            f"http://127.0.0.1:{self._port}/{u.lstrip('/')}" if not u.startswith("http") else u
            for u in urls
        ]
        self._call(
            "loadSpriteClipFrames",
            emotion,
            to_abs(in_frame_urls),
            to_abs(loop_frame_urls),
            to_abs(out_frame_urls),
        )

    def set_sprite_clip_config(self, emotion: str, config: dict) -> None:
        """设置某个情绪的三段式 clip 播放参数。"""
        # 与 set_emotion 一致做一次映射，确保例如 blush->shy 时也能命中正确 clip 配置。
        sprite_name = self._PRESET_TO_SPRITE.get(emotion, emotion)
        self._call("setSpriteClipConfig", sprite_name, config or {})

    def play_sprite_clip_preview(self, duration_sec: float, follow_live2d: bool = False) -> None:
        """按目标总时长播放一次 in/loop/out 预览（测试界面用）。"""
        try:
            d = max(0.2, float(duration_sec))
        except Exception:
            d = 1.2
        self._call("playSpriteClipPreview", d, {"followLive2D": bool(follow_live2d)})

    def set_sprite_align_transform(
        self,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale_mul: float = 1.0,
        baseline_offset_px: float = 0.0,
        show_baseline: bool = False,
        emotion: str | None = None,
        texture_cx: float | None = None,
        texture_feet_y: float | None = None,
    ) -> None:
        """设置 sprite 表情层对齐参数（位置/缩放/底线标记）。

        texture_cx / texture_feet_y 由后处理脚本计算（align_meta.json），
        设置后该情绪进入「锚点感知模式」，自动跟随 Live2D 位置，无需 lock。
        """
        payload = {
            "offsetX": float(offset_x),
            "offsetY": float(offset_y),
            "scaleMul": float(scale_mul),
            "baselineOffsetPx": float(baseline_offset_px),
            "showBaseline": bool(show_baseline),
        }
        if texture_cx is not None:
            payload["textureCx"] = float(texture_cx)
        if texture_feet_y is not None:
            payload["textureFeetY"] = float(texture_feet_y)
        sprite_name = None
        if emotion:
            sprite_name = self._PRESET_TO_SPRITE.get(str(emotion), str(emotion))
        if sprite_name:
            self._call("setSpriteAlignTransform", payload, sprite_name)
        else:
            self._call("setSpriteAlignTransform", payload)

    def get_sprite_align_transform(self, callback, emotion: str | None = None) -> None:
        """异步读取当前 sprite 表情层对齐参数。"""
        if self._ready and self._view:
            sprite_name = None
            if emotion:
                sprite_name = self._PRESET_TO_SPRITE.get(str(emotion), str(emotion))
            if sprite_name:
                js = f"window.renderApp ? window.renderApp.getSpriteAlignTransform({json.dumps(sprite_name, ensure_ascii=False)}) : null"
            else:
                js = "window.renderApp ? window.renderApp.getSpriteAlignTransform() : null"
            self._view.page().runJavaScript(
                js,
                callback,
            )

    def set_sprite_align_lock_to_live2d(self, locked: bool, emotion: str | None = None) -> None:
        """开启/关闭 sprite 表情层对齐对 Live2D 变换的实时锁定。"""
        sprite_name = None
        if emotion:
            sprite_name = self._PRESET_TO_SPRITE.get(str(emotion), str(emotion))
        if sprite_name:
            self._call("setSpriteAlignLockToLive2D", bool(locked), sprite_name)
        else:
            self._call("setSpriteAlignLockToLive2D", bool(locked))

    def set_sprite_align_lock_delta(
        self,
        dx: float,
        dy: float,
        ds: float,
        lock_s: float,
        emotion: str | None = None,
    ) -> None:
        """直接注入已持久化的 delta，绕过 isReady() 检查，用于重启后恢复绑定。"""
        sprite_name = None
        if emotion:
            sprite_name = self._PRESET_TO_SPRITE.get(str(emotion), str(emotion))
        if sprite_name:
            self._call("setSpriteAlignLockDelta", float(dx), float(dy), float(ds), float(lock_s), sprite_name)
        else:
            self._call("setSpriteAlignLockDelta", float(dx), float(dy), float(ds), float(lock_s))

    def load_kur1or3_sprites(self, images_dir: Path | str) -> None:
        """自动扫描 images_dir 中的 kurisu_*.png 并注册所有情绪帧。

        文件命名规则：kurisu_{emotion}{frame}.png
        例：kurisu_normal1.png, kurisu_smile1.png, kurisu_smile2.png, ...
        也支持三段式说话动画：
          kurisu_{emotion}_in0001.png
          kurisu_{emotion}_loop0001.png
          kurisu_{emotion}_out0001.png

        支持目录化组织（递归扫描）：
          images/{emotion}/kurisu_{emotion}1.png
          images/{emotion}/{in|loop|out}/kurisu_{emotion}_{phase}0001.png
        """
        import re
        import os
        images_dir = Path(images_dir)
        # 旧式单序列：kurisu_smile1.png
        pattern = re.compile(r"kurisu_([a-z_]+?)(\d+)\.png$", re.IGNORECASE)
        # 新式分段：kurisu_blush_in0001.png
        clip_pattern = re.compile(r"kurisu_([a-z_]+)_(in|loop|out)(\d+)\.png$", re.IGNORECASE)

        emotion_frames: dict[str, list[tuple[int, str]]] = {}
        clip_frames: dict[str, dict[str, list[tuple[int, str]]]] = {}
        # 过滤备份目录，避免把调参过程中的历史帧误当成正式素材加载。
        # 默认排除包含 "backup" 的任意目录层级，可通过环境变量显式关闭：
        #   SPRITE_SCAN_INCLUDE_BACKUPS=1
        include_backups = os.environ.get("SPRITE_SCAN_INCLUDE_BACKUPS", "").strip().lower() in ("1", "true", "yes", "on")
        include_manual = os.environ.get("SPRITE_SCAN_INCLUDE_MANUALS", "").strip().lower() in ("1", "true", "yes", "on")
        for p in sorted(images_dir.rglob("kurisu_*.png")):
            if not include_backups and any("backup" in part.lower() for part in p.parts):
                continue
            # 默认跳过人工实验目录（如 shy_seedance_manual*），避免同名情绪被重复并入导致节奏异常。
            if not include_manual and any("seedance_manual" in part.lower() for part in p.parts):
                continue
            cm = clip_pattern.match(p.name)
            if cm:
                emotion = cm.group(1).lower()
                phase = cm.group(2).lower()
                frame_idx = int(cm.group(3))
                rel = p.relative_to(_PROJECT_ROOT).as_posix()
                clip_frames.setdefault(emotion, {"in": [], "loop": [], "out": []})
                clip_frames[emotion][phase].append((frame_idx, rel))
                continue
            m = pattern.match(p.name)
            if not m:
                continue
            emotion, frame_idx = m.group(1), int(m.group(2))
            emotion = emotion.lower()
            rel = p.relative_to(_PROJECT_ROOT).as_posix()
            emotion_frames.setdefault(emotion, []).append((frame_idx, rel))

        for emotion, frames in emotion_frames.items():
            frames.sort(key=lambda x: x[0])
            urls = [rel for _, rel in frames]
            self.load_sprite_frames(emotion, urls)
            logger.info(f"[RenderEngine] 注册 sprite: {emotion} ({len(urls)} 帧)")

        # clip 纹理较重（尤其高分辨率长序列），这里做安全限帧与去重，避免 WebEngine/GPU 崩溃。
        max_in = int(os.environ.get("SPRITE_CLIP_MAX_IN_FRAMES", "24"))
        max_loop = int(os.environ.get("SPRITE_CLIP_MAX_LOOP_FRAMES", "240"))
        max_out = int(os.environ.get("SPRITE_CLIP_MAX_OUT_FRAMES", "24"))

        def _sample_urls(urls: list[str], max_frames: int) -> list[str]:
            if max_frames <= 0 or len(urls) <= max_frames:
                return urls
            if max_frames == 1:
                return [urls[0]]
            n = len(urls)
            idxs = [round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
            return [urls[i] for i in idxs]

        for emotion, phases in clip_frames.items():
            in_urls = [rel for _, rel in sorted(phases["in"], key=lambda x: x[0])]
            loop_urls = [rel for _, rel in sorted(phases["loop"], key=lambda x: x[0])]
            out_urls = [rel for _, rel in sorted(phases["out"], key=lambda x: x[0])]
            # blush/shy 为全帧视频驱动，必须保留原始帧数以匹配真实时长（24fps -> frameInterval=42ms）。
            if emotion not in ("shy", "blush"):
                in_urls = _sample_urls(in_urls, max_in)
                loop_urls = _sample_urls(loop_urls, max_loop)
                out_urls = _sample_urls(out_urls, max_out)
            # loop 为空时无法播放，说话阶段至少需要一个主循环帧
            if not loop_urls:
                logger.warning(
                    f"[RenderEngine] 跳过 clip 注册: {emotion}（缺少 loop 帧）"
                )
                continue
            self.load_sprite_clip_frames(emotion, in_urls, loop_urls, out_urls)
            logger.info(
                f"[RenderEngine] 注册 clip: {emotion} (in={len(in_urls)}, "
                f"loop={len(loop_urls)}, out={len(out_urls)})"
            )

    def set_mode(self, mode: str) -> None:
        """切换渲染模式: 'sprite' | 'live2d' | 'both' | 'hybrid'

        hybrid 模式下：
          - 待机时显示 Live2D 模型
          - set_speaking(True)  → Live2D 淡出，帧动画淡入（200ms）
          - set_speaking(False) → 帧动画淡出，Live2D 淡入（200ms）
        """
        self._call("setMode", mode)

    def load_kurisu_model(self) -> None:
        """加载项目内置的 Kurisu Live2D 模型（render/assets/models/Kurisu2_vts）。"""
        model_path = "render/assets/models/Kurisu2_vts/kurisu.model3.json"
        self.load_live2d_model(model_path)
        logger.info("[RenderEngine] 已加载 Kurisu Live2D 模型")

    # ------------------------------------------------------------------
    # Live2D 变换：位置 / 缩放
    # ------------------------------------------------------------------

    _LAYOUT_FILE = _PROJECT_ROOT / "config" / "live2d_layout.json"

    def set_live2d_transform(self, offset_x: float, offset_y: float, scale_factor: float) -> None:
        """设置 Live2D 模型的位置偏移（逻辑像素）与缩放倍率（1.0 = 默认）。"""
        self._call("setLive2DTransform", float(offset_x), float(offset_y), float(scale_factor))

    def get_live2d_transform(self, callback) -> None:
        """异步获取当前 Live2D 变换参数，结果 dict 通过 callback({offsetX, offsetY, scaleFactor}) 返回。"""
        if self._ready and self._view:
            self._view.page().runJavaScript(
                "window.renderApp ? window.renderApp.getLive2DTransform() : null",
                callback,
            )

    def reset_live2d_transform(self) -> None:
        """重置 Live2D 变换到默认（居中、无偏移、1.0 缩放）。"""
        self._call("resetLive2DTransform")

    def save_live2d_layout(self, offset_x: float, offset_y: float, scale_factor: float) -> None:
        """将当前变换参数保存到 config/live2d_layout.json。"""
        layout = {"offset_x": float(offset_x), "offset_y": float(offset_y),
                  "scale_factor": float(scale_factor)}
        try:
            self._LAYOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._LAYOUT_FILE.write_text(json.dumps(layout, indent=2, ensure_ascii=False))
            logger.info(f"[RenderEngine] Live2D 布局已保存: {layout}")
        except Exception as e:
            logger.warning(f"[RenderEngine] Live2D 布局保存失败: {e}")

    def load_saved_live2d_layout(self) -> dict | None:
        """读取已保存的布局参数，不存在时返回 None。"""
        if not self._LAYOUT_FILE.exists():
            return None
        try:
            return json.loads(self._LAYOUT_FILE.read_text())
        except Exception as e:
            logger.warning(f"[RenderEngine] Live2D 布局读取失败: {e}")
            return None

    def _apply_saved_live2d_layout(self) -> None:
        """页面加载完成后自动应用已保存的布局（内部调用）。"""
        layout = self.load_saved_live2d_layout()
        if layout:
            self.set_live2d_transform(
                layout.get("offset_x", 0),
                layout.get("offset_y", 0),
                layout.get("scale_factor", 1.0),
            )
            logger.info(f"[RenderEngine] 已应用保存的 Live2D 布局: {layout}")

    def set_hybrid_mode(self) -> None:
        """启用 hybrid 混合模式（Live2D 待机 + 帧动画说话）的便捷方法。

        需要同时满足以下条件才能正常显示：
          1. 已加载 Live2D 模型：engine.load_kurisu_model() 或 engine.load_live2d_model(url)
          2. 已加载至少 normal 情绪的 sprite 帧：engine.load_kur1or3_sprites(dir)
          3. index.html 中已取消注释 pixi-live2d-display CDN 脚本

        一键启动示例::

            engine.load_kurisu_model()
            engine.load_kur1or3_sprites("render/assets/images")
            engine.set_hybrid_mode()
            # 之后正常调用 set_speaking() / set_emotion() 即可
        """
        self.set_mode("hybrid")
