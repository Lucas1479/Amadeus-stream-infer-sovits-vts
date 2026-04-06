import sys
import os
import asyncio
import threading
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLabel, QScrollArea, QFrame, QSizePolicy,
                             QListWidget, QListWidgetItem, QMenu, QAction, QInputDialog,
                             QPushButton, QDesktopWidget, QDoubleSpinBox, QCheckBox, QSlider)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QObject, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor, QCursor

from qfluentwidgets import (FluentWindow, NavigationItemPosition, SubtitleLabel, setFont,
                            TextEdit, PushButton, PrimaryPushButton, ScrollArea,
                            SettingCardGroup, ComboBoxSettingCard, SwitchSettingCard,
                            OptionsSettingCard, FluentIcon as FIF, InfoBar, InfoBarPosition,
                            setTheme, Theme, ToolButton, TransparentToolButton,
                            LineEdit, Action, RoundMenu, PushSettingCard,
                            OptionsConfigItem, OptionsValidator, ConfigItem, BoolValidator,
                            setThemeColor, ComboBox)

def get_main_module():
    # 直接运行 python main.py 时，主模块注册为 '__main__' 而非 'main'
    m = sys.modules.get('__main__')
    if m and hasattr(m, 'ENABLE_CONVERSATION'):
        return m
    return sys.modules.get('main')


def _get_live_mod():
    """返回 live.sidecar 模块（重构后 LIVE_API_ENABLED 移到此处）。"""
    return sys.modules.get('live.sidecar')


def _get_mm_mod():
    """返回 multimodal.controller 模块（重构后 MULTIMODAL_ENABLED 移到此处）。"""
    return sys.modules.get('multimodal.controller')

class SignalBridge(QObject):
    response_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    user_input_signal = pyqtSignal(str)
    # event_type: "start"|"done"|"error"  task_id: str  data: str(task text or result)
    openclaw_signal = pyqtSignal(str, str, str)
    # session_id, title
    title_signal = pyqtSignal(str, str)
    # ASR 识别结果，由后台线程发回主线程走正常发送流程
    asr_recognized_signal = pyqtSignal(str)

class ChatBubble(QFrame):
    def __init__(self, role, text="", parent=None):
        super().__init__(parent)
        self.role = role
        self.init_ui()
        self.setText(text)

    def init_ui(self):
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 5, 0, 5)
        
        self.avatar = QLabel(self)
        self.avatar.setFixedSize(36, 36)
        self.avatar.setAlignment(Qt.AlignCenter)
        self.avatar.setStyleSheet("border-radius: 18px; color: white; font-weight: bold; font-size: 16px;")
        
        self.bubble = QLabel(self)
        self.bubble.setWordWrap(True)
        self.bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        if self.role == 'user':
            self.avatar.setText("U")
            self.avatar.setStyleSheet(self.avatar.styleSheet() + "background-color: #0078D4;")
            self.bubble.setStyleSheet("""
                background-color: #F3F2F1;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            """)
            self.layout.addStretch(1)
            self.layout.addWidget(self.bubble, 8)
            self.layout.addWidget(self.avatar, 0, Qt.AlignTop)
        else:
            self.avatar.setText("K")
            self.avatar.setStyleSheet(self.avatar.styleSheet() + "background-color: #107C10;")
            self.bubble.setStyleSheet("""
                background-color: transparent;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                border: 1px solid #E0E0E0;
            """)
            self.layout.addWidget(self.avatar, 0, Qt.AlignTop)
            self.layout.addWidget(self.bubble, 8)
            self.layout.addStretch(1)

    def setText(self, text):
        self.bubble.setText(text)

class ChatInputBar(QFrame):
    def __init__(self, send_cb, asr_cb, parent=None):
        super().__init__(parent)
        self.send_cb = send_cb
        self.asr_cb = asr_cb
        self.init_ui()

    def init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 10)
        outer.setSpacing(4)
        self.setStyleSheet("background-color: transparent;")

        # Live API 状态提示条（默认隐藏）
        self.live_status_bar = QLabel("🔴  Live API 运行中 — 麦克风已由 Live API 占用，请直接说话", self)
        self.live_status_bar.setAlignment(Qt.AlignCenter)
        self.live_status_bar.setStyleSheet(
            "background-color: #FFF3CD; color: #856404; border-radius: 4px;"
            "padding: 4px 8px; font-size: 12px;"
        )
        self.live_status_bar.hide()
        outer.addWidget(self.live_status_bar)

        # 输入行
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        self.text_input = TextEdit(self)
        self.text_input.setPlaceholderText("Type a message or press mic to speak...")
        self.text_input.setFixedHeight(60)

        self.send_button = PrimaryPushButton(FIF.SEND, "Send", self)
        self.send_button.setFixedSize(100, 40)
        self.send_button.clicked.connect(self._on_send)

        self.mic_button = ToolButton(FIF.MICROPHONE, self)
        self.mic_button.setFixedSize(40, 40)
        if self.asr_cb:
            self.mic_button.clicked.connect(self.asr_cb)
        else:
            self.mic_button.setEnabled(False)

        row.addWidget(self.text_input, 1)
        row.addWidget(self.mic_button)
        row.addWidget(self.send_button)
        outer.addLayout(row)

    def set_live_api_active(self, active: bool):
        """Live API 开启/关闭时调用，禁用麦克风按钮并显示状态条。"""
        self.live_status_bar.setVisible(active)
        # Live API 占用麦克风时禁用 ASR 按钮；关闭时只有原本有回调才恢复
        self.mic_button.setEnabled(not active and self.asr_cb is not None)
        if active:
            self.mic_button.setToolTip("Live API 运行中，麦克风已被占用")
        else:
            self.mic_button.setToolTip("")

    def _on_send(self):
        text = self.text_input.toPlainText().strip()
        if text:
            self.send_cb(text)
            self.text_input.clear()

class ChatInterface(QWidget):
    _PANEL_W = 420  # 聊天面板宽度

    def __init__(self, send_callback, asr_callback, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("ChatInterface")
        self.send_callback = send_callback
        self.asr_callback = asr_callback
        self.last_assistant_bubble = None
        self._render_engine = None
        self._panel_expanded = True
        self._render_mode = "vts"   # "vts" | "pixi"
        self.init_ui()

    def init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 左侧：角色背景区 ───────────────────────────────────────────────────
        self._char_container = QWidget(self)
        self._char_container.setStyleSheet("background: #1e1e1e;")
        self._char_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        char_layout = QVBoxLayout(self._char_container)
        char_layout.setContentsMargins(0, 0, 0, 0)
        self._char_container.hide()   # 默认 VTS 模式，角色区折叠
        root.addWidget(self._char_container, 1)

        # ── 右侧：聊天面板 ────────────────────────────────────────────────────
        self._chat_panel = QFrame(self)
        # VTS 模式下不限制宽度，Pixi 模式下固定为 _PANEL_W
        self._chat_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._chat_panel.setStyleSheet(
            "QFrame#chatPanel { background-color: rgba(248,248,248,230); "
            "border-left: 1px solid #ddd; }"
        )
        self._chat_panel.setObjectName("chatPanel")
        panel_layout = QVBoxLayout(self._chat_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        # ── 面板头部（model bar + 折叠按钮，始终可见）──────────────────────────
        self.model_bar = QFrame(self._chat_panel)
        self.model_bar.setFixedHeight(44)
        self.model_bar.setStyleSheet(
            "QFrame { background-color: #F5F5F5; border-bottom: 1px solid #E0E0E0; }"
        )
        mb = QHBoxLayout(self.model_bar)
        mb.setContentsMargins(8, 0, 12, 0)
        mb.setSpacing(8)

        self._toggle_btn = TransparentToolButton(FIF.LEFT_ARROW, self.model_bar)
        self._toggle_btn.setToolTip("折叠历史记录")
        self._toggle_btn.clicked.connect(self._toggle_panel)
        mb.addWidget(self._toggle_btn)

        sep0 = QFrame(self.model_bar)
        sep0.setFrameShape(QFrame.VLine)
        sep0.setStyleSheet("background:#D0D0D0; border:none;")
        sep0.setFixedWidth(1)
        mb.addWidget(sep0)

        lbl_provider = QLabel("Model:", self.model_bar)
        lbl_provider.setStyleSheet("font-size:12px; color:#444; background:transparent; border:none;")
        self.provider_combo = ComboBox(self.model_bar)
        self.provider_combo.addItems(["deepseek", "gemini", "bedrock", "local"])
        self.provider_combo.setFixedWidth(110)
        self.provider_combo.currentTextChanged.connect(self._on_bar_provider_changed)

        sep1 = QFrame(self.model_bar)
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("background:#D0D0D0; border:none;")
        sep1.setFixedWidth(1)

        lbl_tts = QLabel("TTS:", self.model_bar)
        lbl_tts.setStyleSheet("font-size:12px; color:#444; background:transparent; border:none;")
        self.tts_mode_combo = ComboBox(self.model_bar)
        self.tts_mode_combo.addItems(["Parallel ×2", "CUDA Graph ×1"])
        self.tts_mode_combo.setFixedWidth(140)
        self.tts_mode_combo.currentTextChanged.connect(self._on_bar_tts_mode_changed)

        mb.addWidget(lbl_provider)
        mb.addWidget(self.provider_combo)
        mb.addWidget(sep1)
        mb.addWidget(lbl_tts)
        mb.addWidget(self.tts_mode_combo)
        mb.addStretch(1)

        sep2 = QFrame(self.model_bar)
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("background:#D0D0D0; border:none;")
        sep2.setFixedWidth(1)
        mb.addWidget(sep2)

        self._render_mode_btn = TransparentToolButton(FIF.VIDEO, self.model_bar)
        self._render_mode_btn.setToolTip("切换到角色渲染视图 (PixiJS)")
        self._render_mode_btn.clicked.connect(self._toggle_render_mode)
        mb.addWidget(self._render_mode_btn)

        panel_layout.addWidget(self.model_bar)

        # ── 历史区（可折叠）──────────────────────────────────────────────────
        self._history_area = QWidget(self._chat_panel)
        hist_layout = QVBoxLayout(self._history_area)
        hist_layout.setContentsMargins(0, 0, 0, 0)
        hist_layout.setSpacing(0)

        # Session bar
        sess_bar = QWidget(self._history_area)
        sess_bar.setStyleSheet("background:#FAFAFA; border-bottom:1px solid #E8E8E8;")
        sb_layout = QHBoxLayout(sess_bar)
        sb_layout.setContentsMargins(8, 4, 8, 4)
        sb_layout.setSpacing(6)
        self.new_chat_btn = PrimaryPushButton(FIF.ADD, "New Chat", sess_bar)
        self.new_chat_btn.clicked.connect(self._on_new_chat)
        sb_layout.addWidget(self.new_chat_btn)
        sb_layout.addStretch(1)
        hist_layout.addWidget(sess_bar)

        self.session_list = QListWidget(self._history_area)
        self.session_list.setMaximumHeight(140)
        self.session_list.setStyleSheet(
            "QListWidget { border:none; background:#FAFAFA; }"
            "QListWidget::item { padding:7px 10px; border-radius:4px; }"
            "QListWidget::item:selected { background:#EAEAEA; }"
        )
        self.session_list.itemClicked.connect(self._on_session_selected)
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._on_session_context_menu)
        hist_layout.addWidget(self.session_list)

        self.scroll_area = ScrollArea(self._history_area)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("background:transparent;")
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setContentsMargins(8, 8, 8, 8)
        self.scroll_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_widget)
        hist_layout.addWidget(self.scroll_area, 1)

        panel_layout.addWidget(self._history_area, 1)

        # ── 输入栏（始终可见）────────────────────────────────────────────────
        self.input_bar = ChatInputBar(self._on_send_text, self._on_mic_clicked, self._chat_panel)
        panel_layout.addWidget(self.input_bar)

        root.addWidget(self._chat_panel, 0)

        self.reload_sessions()
        self._sync_model_bar()

    def _init_render_engine(self):
        """在 ChatInterface 内部启动渲染引擎并嵌入背景区（首次切换到 pixi 模式时懒加载）。"""
        import os
        try:
            from render.engine import RenderEngine
            engine = RenderEngine()
            # WebEngine 加载完成后强制 navbar repaint（OpenGL 初始化会使 navbar 变黑）
            engine.on_ready = lambda: QTimer.singleShot(80, self._force_navbar_repaint)
            widget = engine.start()
            self._char_container.layout().addWidget(widget)
            self._render_engine = engine
            images_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "render", "assets", "images"
            )
            if os.path.isdir(images_dir):
                engine.load_kur1or3_sprites(images_dir)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[ChatInterface] 渲染引擎初始化失败: {e}")

    def engine(self):
        return self._render_engine

    def _toggle_panel(self):
        self._panel_expanded = not self._panel_expanded
        self._history_area.setVisible(self._panel_expanded)
        if self._panel_expanded:
            self._toggle_btn.setIcon(FIF.LEFT_ARROW)
            self._toggle_btn.setToolTip("折叠历史记录")
        else:
            self._toggle_btn.setIcon(FIF.RIGHT_ARROW)
            self._toggle_btn.setToolTip("展开历史记录")

    def _toggle_render_mode(self):
        """在 VTS 渲染 和 PixiJS 角色渲染 之间切换。"""
        import logging as _logging
        _log = _logging.getLogger(__name__)
        from vts.expression_controller import get_controller
        ctrl = get_controller()

        if self._render_mode == "vts":
            self._render_mode = "pixi"
            self._char_container.show()
            self._chat_panel.setFixedWidth(self._PANEL_W)
            self._render_mode_btn.setIcon(FIF.MOVIE)
            self._render_mode_btn.setToolTip("切换到 VTS 渲染")
            # 清除 VTS 当前表情状态（避免 VTS 持续渲染旧表情）
            ctrl.on_turn_end()
            # 首次激活时懒加载渲染引擎
            if self._render_engine is None:
                self._init_render_engine()
            # 切换 backend → pixi
            if self._render_engine is not None:
                ctrl.set_render_engine(self._render_engine, backend="pixi")
                _log.info("[ChatInterface] 切换到 PixiJS 渲染模式")
        else:
            self._render_mode = "vts"
            self._char_container.hide()
            self._chat_panel.setMaximumWidth(16777215)  # 解除固定宽度
            self._chat_panel.setMinimumWidth(0)
            self._chat_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._render_mode_btn.setIcon(FIF.VIDEO)
            self._render_mode_btn.setToolTip("切换到角色渲染视图 (PixiJS)")
            # 停止口型动画
            if self._render_engine is not None:
                try:
                    self._render_engine.set_speaking(False)
                except Exception as _e:
                    _log.error(f"[ChatInterface] set_speaking(False) 失败: {_e}")
            # 切换 backend → vts
            if self._render_engine is not None:
                ctrl.set_render_engine(self._render_engine, backend="vts")
                _log.info("[ChatInterface] 切换到 VTS 渲染模式")

    def set_live_api_active(self, active: bool):
        """由 SettingInterface 调用，同步 Live API 状态到输入栏。"""
        self.input_bar.set_live_api_active(active)

    def _on_send_text(self, text):
        self.append_user_bubble(text)
        self.create_assistant_bubble()
        self.send_callback(text)

    def _on_mic_clicked(self):
        if self.asr_callback:
            self.asr_callback()

    def _sync_model_bar(self):
        """从 main_module / 环境变量同步 model bar 初始状态。"""
        import os
        main_module = get_main_module()
        if main_module:
            provider = getattr(main_module, 'LLM_PROVIDER', 'deepseek')
            idx = self.provider_combo.findText(provider)
            if idx >= 0:
                self.provider_combo.blockSignals(True)
                self.provider_combo.setCurrentIndex(idx)
                self.provider_combo.blockSignals(False)
        cuda_graph_on = os.environ.get('ENABLE_CUDA_GRAPH', '0') == '1'
        tts_text = "CUDA Graph ×1" if cuda_graph_on else "Parallel ×2"
        idx = self.tts_mode_combo.findText(tts_text)
        if idx >= 0:
            self.tts_mode_combo.blockSignals(True)
            self.tts_mode_combo.setCurrentIndex(idx)
            self.tts_mode_combo.blockSignals(False)

    def _on_bar_provider_changed(self, text):
        main_module = get_main_module()
        if main_module:
            main_module.LLM_PROVIDER = text
            main_module.USE_LOCAL_LLM = (text == "local")
        # 同步到 Settings 页
        main_win = self._get_main_window()
        if main_win and hasattr(main_win, 'settingInterface'):
            si = main_win.settingInterface
            if hasattr(si, 'providerCard'):
                si.providerCard.setValue(text)

    def _on_bar_tts_mode_changed(self, text):
        try:
            import tts.pipeline as _tts_pipeline
            if text == "CUDA Graph ×1":
                _tts_pipeline.reconfigure_tts_mode(cuda_graph=True, concurrency=1)
            else:
                _tts_pipeline.reconfigure_tts_mode(cuda_graph=False, concurrency=2)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"TTS mode switch failed: {e}")
        # 同步到 Settings 页
        main_win = self._get_main_window()
        if main_win and hasattr(main_win, 'settingInterface'):
            si = main_win.settingInterface
            if hasattr(si, 'ttsModeCard'):
                si.ttsModeCard.setValue(text)

    def append_user_bubble(self, text):
        bubble = ChatBubble('user', text)
        self.scroll_layout.addWidget(bubble)
        self._scroll_to_bottom()

    def create_assistant_bubble(self):
        self.last_assistant_bubble = ChatBubble('assistant', "")
        self.scroll_layout.addWidget(self.last_assistant_bubble)
        self._scroll_to_bottom()

    def update_assistant_bubble(self, text):
        if not self.last_assistant_bubble:
            self.create_assistant_bubble()
        self.last_assistant_bubble.setText(text)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum()))

    def clear_chat(self):
        for i in reversed(range(self.scroll_layout.count())):
            item = self.scroll_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.last_assistant_bubble = None

    def _get_main_window(self):
        """向上遍历父链，找到持有 settingInterface 的主窗口"""
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'settingInterface'):
                return parent
            parent = parent.parent()
        return None

    def reload_sessions(self):
        main_module = get_main_module()
        if main_module:
            self.session_list.clear()
            for sid in main_module.list_sessions():
                title = main_module.get_session_title(sid) if hasattr(main_module, 'get_session_title') else sid
                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, sid)
                self.session_list.addItem(item)

    def update_session_title_in_list(self, session_id: str, title: str):
        """在列表中更新指定会话的显示标题"""
        for i in range(self.session_list.count()):
            item = self.session_list.item(i)
            if item.data(Qt.UserRole) == session_id:
                item.setText(title)
                break

    def _on_session_context_menu(self, pos):
        """右键菜单：重命名 / 删除"""
        item = self.session_list.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        rename_action = menu.addAction("✏️  重命名")
        menu.addSeparator()
        delete_action = menu.addAction("🗑️  删除")
        action = menu.exec_(self.session_list.mapToGlobal(pos))
        if action == rename_action:
            self._rename_session(item)
        elif action == delete_action:
            self._delete_session(item)

    def _rename_session(self, item: QListWidgetItem):
        """弹出输入框修改会话显示标题（只改 JSON 内部 title 字段，文件名不变）"""
        current_title = item.text()
        new_title, ok = QInputDialog.getText(
            self, "重命名对话", "输入新标题：", text=current_title
        )
        if ok and new_title.strip() and new_title.strip() != current_title:
            new_title = new_title.strip()
            session_id = item.data(Qt.UserRole) or current_title
            main_module = get_main_module()
            if main_module and hasattr(main_module, 'set_session_title'):
                main_module.set_session_title(session_id, new_title)
            item.setText(new_title)

    def _delete_session(self, item: QListWidgetItem):
        """删除会话文件并从列表移除"""
        from PyQt5.QtWidgets import QMessageBox
        title = item.text()
        session_id = item.data(Qt.UserRole) or title
        reply = QMessageBox.question(
            self, "删除对话",
            f"确认删除「{title}」？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        main_module = get_main_module()
        if main_module and hasattr(main_module, 'delete_session'):
            main_module.delete_session(session_id)
            # 若删除的是当前活跃会话，重置状态
            if (hasattr(main_module, 'get_current_session_id') and
                    main_module.get_current_session_id() == session_id):
                main_module.set_current_session_id(None)
                self.clear_chat()
        row = self.session_list.row(item)
        self.session_list.takeItem(row)

    def _select_session_in_list(self, session_id):
        """在列表中高亮选中指定会话"""
        for i in range(self.session_list.count()):
            item = self.session_list.item(i)
            if item.data(Qt.UserRole) == session_id or item.text() == session_id:
                self.session_list.setCurrentItem(item)
                break

    def _on_session_selected(self, item):
        main_module = get_main_module()
        if main_module:
            sid = item.data(Qt.UserRole) or item.text()
            if main_module.load_session(sid):
                main_module.set_current_session_id(sid)
                main_module.ENABLE_CONVERSATION = True
                main_win = self._get_main_window()
                if main_win:
                    main_win.settingInterface.convCard.setChecked(True)
                self._render_history()

    def _on_new_chat(self):
        main_module = get_main_module()
        if main_module:
            import datetime
            new_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            main_module.create_session(new_id)
            main_module.ENABLE_CONVERSATION = True
            main_win = self._get_main_window()
            if main_win:
                main_win.settingInterface.convCard.setChecked(True)
            self.clear_chat()
            self.reload_sessions()
            self._select_session_in_list(new_id)

    def _render_history(self):
        self.clear_chat()
        main_module = get_main_module()
        if main_module and hasattr(main_module, 'conversation_history'):
            for msg in main_module.conversation_history.dialog:
                role = msg.get("role")
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_content = next((item["text"] for item in content if item.get("type") == "text"), "")
                else:
                    text_content = content
                
                if role == "user":
                    self.append_user_bubble(text_content)
                elif role == "assistant":
                    bubble = ChatBubble('assistant', text_content)
                    self.scroll_layout.addWidget(bubble)
            self._scroll_to_bottom()

class OpenClawTaskCard(QFrame):
    """单条 OpenClaw 任务记录卡片"""
    _STATUS_COLORS = {
        "running": "#FF8C00",
        "done":    "#107C10",
        "error":   "#C42B1C",
    }
    _STATUS_LABELS = {
        "running": "Running...",
        "done":    "Done",
        "error":   "Error",
    }

    def __init__(self, task_id: str, task_text: str, timestamp: str, parent=None):
        super().__init__(parent)
        self.task_id = task_id
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            OpenClawTaskCard {
                background: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header row: status dot + status label + timestamp
        header = QHBoxLayout()
        self._dot = QLabel(self)
        self._dot.setFixedSize(12, 12)
        self._dot.setStyleSheet(
            f"border-radius:6px; background-color:{self._STATUS_COLORS['running']};"
        )
        self._status_lbl = QLabel("Running...", self)
        self._status_lbl.setStyleSheet("font-weight:bold; font-size:12px; color:#FF8C00;")

        ts_lbl = QLabel(timestamp, self)
        ts_lbl.setStyleSheet("font-size:11px; color:#888;")

        header.addWidget(self._dot)
        header.addWidget(self._status_lbl)
        header.addStretch(1)
        header.addWidget(ts_lbl)
        layout.addLayout(header)

        # Task text
        task_lbl = QLabel(self)
        task_lbl.setText(f"<b>Task:</b> {task_text}")
        task_lbl.setWordWrap(True)
        task_lbl.setStyleSheet("font-size:12px; color:#333;")
        task_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(task_lbl)

        # Result area (initially hidden)
        self._result_lbl = QLabel(self)
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._result_lbl.setStyleSheet("""
            font-size:12px;
            color:#222;
            background:#F0F0F0;
            border-radius:4px;
            padding:6px;
        """)
        self._result_lbl.hide()
        layout.addWidget(self._result_lbl)

    def set_done(self, result: str):
        self._dot.setStyleSheet(
            f"border-radius:6px; background-color:{self._STATUS_COLORS['done']};"
        )
        self._status_lbl.setText(self._STATUS_LABELS['done'])
        self._status_lbl.setStyleSheet("font-weight:bold; font-size:12px; color:#107C10;")
        self._result_lbl.setText(f"<b>Result:</b> {result}")
        self._result_lbl.show()

    def set_error(self, result: str):
        self._dot.setStyleSheet(
            f"border-radius:6px; background-color:{self._STATUS_COLORS['error']};"
        )
        self._status_lbl.setText(self._STATUS_LABELS['error'])
        self._status_lbl.setStyleSheet("font-weight:bold; font-size:12px; color:#C42B1C;")
        self._result_lbl.setText(f"<b>Error:</b> {result}")
        self._result_lbl.show()


class OpenClawInterface(QWidget):
    """OpenClaw 任务日志面板"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("OpenClawInterface")
        self._cards: dict[str, OpenClawTaskCard] = {}
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(10)

        # Title + clear button
        title_row = QHBoxLayout()
        title = SubtitleLabel("OpenClaw Tasks", self)
        setFont(title, 20)
        clear_btn = PushButton(FIF.DELETE, "Clear", self)
        clear_btn.setFixedSize(90, 32)
        clear_btn.clicked.connect(self.clear_tasks)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(clear_btn)
        root.addLayout(title_row)

        # Scroll area for cards
        self._scroll = ScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._card_layout = QVBoxLayout(self._container)
        self._card_layout.setAlignment(Qt.AlignTop)
        self._card_layout.setSpacing(10)
        self._card_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll.setWidget(self._container)
        root.addWidget(self._scroll, 1)

        # Empty hint
        self._empty_lbl = QLabel("No OpenClaw tasks yet.", self._container)
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet("color:#AAA; font-size:13px;")
        self._card_layout.addWidget(self._empty_lbl)

    def _scroll_to_bottom(self):
        QTimer.singleShot(60, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))

    def add_task(self, task_id: str, task_text: str):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        # Hide empty hint on first real card
        self._empty_lbl.setVisible(self._card_layout.count() == 1)
        card = OpenClawTaskCard(task_id, task_text, ts, self._container)
        self._cards[task_id] = card
        self._card_layout.addWidget(card)
        self._scroll_to_bottom()

    def update_task(self, task_id: str, status: str, result: str):
        card = self._cards.get(task_id)
        if card is None:
            return
        if status == "done":
            card.set_done(result)
        else:
            card.set_error(result)
        self._scroll_to_bottom()

    def clear_tasks(self):
        for card in list(self._cards.values()):
            card.deleteLater()
        self._cards.clear()
        self._empty_lbl.setVisible(True)


class SettingInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("SettingInterface")
        
        self.scrollWidget = QWidget()
        self.vBoxLayout = QVBoxLayout(self.scrollWidget)
        self.vBoxLayout.setContentsMargins(20, 20, 20, 20)
        self.vBoxLayout.setSpacing(15)
        
        self.titleLabel = SubtitleLabel("Settings", self.scrollWidget)
        setFont(self.titleLabel, 24)
        self.vBoxLayout.addWidget(self.titleLabel)
        
        self.init_cards()
        self.vBoxLayout.addStretch(1)
        
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

    def init_cards(self):
        # Model Settings
        self.modelGroup = SettingCardGroup("Model Configuration", self.scrollWidget)
        
        self.providerCard = ComboBoxSettingCard(
            OptionsConfigItem("Model", "Provider", "deepseek", OptionsValidator(["deepseek", "gemini", "bedrock", "local"])),
            FIF.ROBOT,
            "LLM Provider",
            "Select the language model provider",
            texts=["deepseek", "gemini", "bedrock", "local"],
            parent=self.modelGroup
        )
        self.providerCard.comboBox.currentTextChanged.connect(self._on_provider_changed)
        self.modelGroup.addSettingCard(self.providerCard)
        
        self.localTypeCard = ComboBoxSettingCard(
            OptionsConfigItem("Model", "LocalType", "llama_server", OptionsValidator(["llama_server", "lmstudio", "cli"])),
            FIF.COMMAND_PROMPT,
            "Local LLM Type",
            "Select the local LLM backend",
            texts=["llama_server", "lmstudio", "cli"],
            parent=self.modelGroup
        )
        self.localTypeCard.comboBox.currentTextChanged.connect(self._on_local_type_changed)
        self.modelGroup.addSettingCard(self.localTypeCard)
        
        self.vBoxLayout.addWidget(self.modelGroup)
        
        # Multimodal Settings
        self.mmGroup = SettingCardGroup("Multimodal & Vision", self.scrollWidget)
        
        self.liveApiCard = SwitchSettingCard(
            FIF.CAMERA,
            "LIVE API",
            "Enable vision capabilities",
            configItem=ConfigItem("Multimodal", "LiveAPI", False, BoolValidator()),
            parent=self.mmGroup
        )
        self.liveApiCard.switchButton.checkedChanged.connect(self._on_live_api_changed)
        self.mmGroup.addSettingCard(self.liveApiCard)
        
        self.mmSourceCard = ComboBoxSettingCard(
            OptionsConfigItem("Multimodal", "Source", "Screen", OptionsValidator(["Screen", "Webcam"])),
            FIF.VIDEO,
            "Vision Source",
            "Select input source for vision",
            texts=["Screen", "Webcam"],
            parent=self.mmGroup
        )
        self.mmSourceCard.comboBox.currentTextChanged.connect(self._on_mm_source_changed)
        self.mmGroup.addSettingCard(self.mmSourceCard)
        
        self.mmRegionCard = PushSettingCard(
            "Select Region",
            FIF.CUT,
            "Screen Region",
            "Select a specific region of the screen for vision",
            parent=self.mmGroup
        )
        self.mmRegionCard.clicked.connect(self._on_mm_region_click)
        self.mmGroup.addSettingCard(self.mmRegionCard)
        
        self.vBoxLayout.addWidget(self.mmGroup)
        
        # Chat & Voice Settings
        self.chatGroup = SettingCardGroup("Chat & Voice", self.scrollWidget)
        
        self.convCard = SwitchSettingCard(
            FIF.CHAT,
            "Continuous Conversation",
            "Enable multi-turn conversation memory",
            configItem=ConfigItem("Chat", "Continuous", True, BoolValidator()),
            parent=self.chatGroup
        )
        self.convCard.switchButton.checkedChanged.connect(self._on_conv_changed)
        self.chatGroup.addSettingCard(self.convCard)
        
        self.sprintCard = SwitchSettingCard(
            FIF.SPEED_HIGH,
            "First Sentence Sprint",
            "Optimize latency for the first sentence",
            configItem=ConfigItem("Chat", "Sprint", True, BoolValidator()),
            parent=self.chatGroup
        )
        self.sprintCard.switchButton.checkedChanged.connect(self._on_sprint_changed)
        self.chatGroup.addSettingCard(self.sprintCard)
        
        self.subtitleCard = SwitchSettingCard(
            FIF.FONT,
            "Floating Subtitle Window",
            "Show floating subtitles",
            configItem=ConfigItem("Chat", "Subtitle", False, BoolValidator()),
            parent=self.chatGroup
        )
        self.subtitleCard.switchButton.checkedChanged.connect(self._on_subtitle_changed)
        self.chatGroup.addSettingCard(self.subtitleCard)

        self.ttsModeCard = ComboBoxSettingCard(
            OptionsConfigItem("TTS", "Mode", "Parallel ×2", OptionsValidator(["Parallel ×2", "CUDA Graph ×1"])),
            FIF.TILES,
            "TTS Inference Mode",
            "Parallel ×2: 2-slot concurrent | CUDA Graph ×1: single, faster first sentence",
            texts=["Parallel ×2", "CUDA Graph ×1"],
            parent=self.chatGroup
        )
        self.ttsModeCard.comboBox.currentTextChanged.connect(self._on_tts_mode_changed)
        self.chatGroup.addSettingCard(self.ttsModeCard)

        self.vBoxLayout.addWidget(self.chatGroup)
        
        # Sync initial state
        self.sync_from_main()

    def sync_from_main(self):
        main_module = get_main_module()
        if main_module:
            self.providerCard.setValue(getattr(main_module, 'LLM_PROVIDER', 'deepseek'))
            self.localTypeCard.setValue(getattr(main_module, 'LOCAL_LLM_TYPE', 'llama_server'))
            _live = _get_live_mod()
            self.liveApiCard.setChecked(
                getattr(_live, 'LIVE_API_ENABLED', False) if _live
                else getattr(main_module, 'LIVE_API_ENABLED', False)
            )
            _mm = _get_mm_mod()
            raw_src = (
                getattr(_mm, 'MULTIMODAL_INPUT_SOURCE', 'screen') if _mm
                else getattr(main_module, 'MULTIMODAL_SOURCE', 'screen')
            )
            self.mmSourceCard.setValue('Screen' if raw_src == 'screen' else 'Webcam')
            self.convCard.setChecked(getattr(main_module, 'ENABLE_CONVERSATION', True))
            self.sprintCard.setChecked(getattr(main_module, 'FIRST_SENTENCE_SPRINT', True))
            self.subtitleCard.setChecked(getattr(main_module, 'SHOW_SUBTITLE_WINDOW', False))
            import os
            cuda_graph_on = os.environ.get('ENABLE_CUDA_GRAPH', '0') == '1'
            self.ttsModeCard.setValue("CUDA Graph ×1" if cuda_graph_on else "Parallel ×2")

            self._update_visibility()

    def _update_visibility(self):
        is_local = self.providerCard.comboBox.currentText() == "local"
        self.localTypeCard.setVisible(is_local)

    def _on_provider_changed(self, text):
        main_module = get_main_module()
        if main_module:
            main_module.LLM_PROVIDER = text
            main_module.USE_LOCAL_LLM = (text == "local")
        self._update_visibility()
        # 反向同步到 Chat model bar
        main_win = self._get_main_window()
        if main_win and hasattr(main_win, 'chatInterface'):
            ci = main_win.chatInterface
            if hasattr(ci, 'provider_combo'):
                ci.provider_combo.blockSignals(True)
                idx = ci.provider_combo.findText(text)
                if idx >= 0:
                    ci.provider_combo.setCurrentIndex(idx)
                ci.provider_combo.blockSignals(False)

    def _on_local_type_changed(self, text):
        main_module = get_main_module()
        if main_module:
            main_module.LOCAL_LLM_TYPE = text

    def _on_live_api_changed(self, checked):
        main_module = get_main_module()
        if not main_module:
            return
        # 优先写入重构后的子模块；若尚未导入则回退到 main 模块属性
        _live = _get_live_mod()
        _mm = _get_mm_mod()
        if _live:
            _live.LIVE_API_ENABLED = checked
        else:
            main_module.LIVE_API_ENABLED = checked
        if _mm:
            _mm.MULTIMODAL_ENABLED = checked
        else:
            main_module.MULTIMODAL_ENABLED = checked
        if checked:
            if _mm and hasattr(_mm, '_start_multimodal_if_enabled'):
                asyncio.create_task(_mm._start_multimodal_if_enabled())
            elif hasattr(main_module, '_start_multimodal_if_enabled'):
                asyncio.create_task(main_module._start_multimodal_if_enabled())
        else:
            if _mm and hasattr(_mm, '_stop_multimodal'):
                asyncio.create_task(_mm._stop_multimodal())
            elif hasattr(main_module, '_stop_multimodal'):
                asyncio.create_task(main_module._stop_multimodal())
        # 通知聊天界面更新麦克风状态，并同步悬浮按钮
        main_win = self._get_main_window()
        if main_win and hasattr(main_win, 'chatInterface'):
            main_win.chatInterface.set_live_api_active(checked)
        if main_win and hasattr(main_win, 'sync_live_float_btn'):
            main_win.sync_live_float_btn(checked)

    def _get_main_window(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'chatInterface'):
                return parent
            parent = parent.parent()
        return None

    def _on_mm_source_changed(self, text):
        # "Screen" → "screen", "Webcam" → "webcam"
        source_value = text.lower() if text else "screen"
        _mm = _get_mm_mod()
        if _mm:
            _mm.MULTIMODAL_INPUT_SOURCE = source_value
        else:
            main_module = get_main_module()
            if main_module:
                main_module.MULTIMODAL_SOURCE = source_value

    def _on_mm_region_click(self):
        main_module = get_main_module()
        if main_module:
            dialog = QInputDialog(self)
            dialog.setWindowTitle("Set Region")
            dialog.setLabelText("Enter region (x,y,w,h) or 'None':")
            current = getattr(main_module, 'MULTIMODAL_REGION', None)
            dialog.setTextValue(str(current) if current else "None")
            if dialog.exec_():
                val = dialog.textValue()
                if val.lower() == "none":
                    main_module.MULTIMODAL_REGION = None
                else:
                    try:
                        main_module.MULTIMODAL_REGION = eval(val)
                    except:
                        pass

    def _on_conv_changed(self, checked):
        main_module = get_main_module()
        if main_module:
            main_module.ENABLE_CONVERSATION = checked
            # If enabled, ensure we have a session
            if checked and not main_module.get_current_session_id():
                import datetime
                new_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                main_module.create_session(new_id)
                main_module.set_current_session_id(new_id)
            elif main_module.get_current_session_id():
                # If we have a session, save the new state immediately
                main_module.save_session()

    def _on_sprint_changed(self, checked):
        main_module = get_main_module()
        if main_module:
            main_module.FIRST_SENTENCE_SPRINT = checked

    def _on_subtitle_changed(self, checked):
        main_module = get_main_module()
        if main_module:
            main_module.SHOW_SUBTITLE_WINDOW = checked
            if checked:
                main_module.start_floating_subtitle()
            else:
                main_module.stop_floating_subtitle()

    def _on_tts_mode_changed(self, text):
        try:
            import tts.pipeline as _tts_pipeline
            if text == "CUDA Graph ×1":
                _tts_pipeline.reconfigure_tts_mode(cuda_graph=True, concurrency=1)
                msg = "CUDA Graph serial — faster first sentence"
            else:
                _tts_pipeline.reconfigure_tts_mode(cuda_graph=False, concurrency=2)
                msg = "Parallel ×2 — higher throughput"
            main_win = self._get_main_window()
            anchor = main_win if main_win else self
            InfoBar.success(
                title="TTS Mode Updated",
                content=msg,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=anchor,
            )
            # 反向同步到 Chat model bar
            if main_win and hasattr(main_win, 'chatInterface'):
                ci = main_win.chatInterface
                if hasattr(ci, 'tts_mode_combo'):
                    ci.tts_mode_combo.blockSignals(True)
                    idx = ci.tts_mode_combo.findText(text)
                    if idx >= 0:
                        ci.tts_mode_combo.setCurrentIndex(idx)
                    ci.tts_mode_combo.blockSignals(False)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"TTS mode switch failed: {e}")

class LiveApiFloatButton(QWidget):
    """始终置顶的 Live API 快捷开关悬浮窗，不在任务栏显示，可拖动。"""

    def __init__(self, toggle_callback, parent=None):
        super().__init__(
            parent,
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool,
        )
        self._toggle_callback = toggle_callback
        self._active = False
        self._drag_pos = None

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(110, 38)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._btn = QPushButton("⚪  LIVE", self)
        self._btn.setFixedSize(110, 38)
        self._btn.clicked.connect(self._on_click)
        self._apply_style(False)
        layout.addWidget(self._btn)

        # 默认放在屏幕右下角
        try:
            geo = QDesktopWidget().availableGeometry()
            self.move(geo.right() - 130, geo.bottom() - 80)
        except Exception:
            self.move(1200, 800)

    # ── 样式 ──────────────────────────────────────────────────────────
    def _apply_style(self, active: bool):
        if active:
            self._btn.setStyleSheet("""
                QPushButton {
                    background-color: #C42B1C;
                    color: white;
                    font-weight: bold;
                    font-size: 13px;
                    border-radius: 8px;
                    border: none;
                }
                QPushButton:hover { background-color: #E53935; }
                QPushButton:pressed { background-color: #B71C1C; }
            """)
            self._btn.setText("🔴  LIVE ON")
        else:
            self._btn.setStyleSheet("""
                QPushButton {
                    background-color: #3C3C3C;
                    color: #CCCCCC;
                    font-weight: bold;
                    font-size: 13px;
                    border-radius: 8px;
                    border: none;
                }
                QPushButton:hover { background-color: #555555; }
                QPushButton:pressed { background-color: #222222; }
            """)
            self._btn.setText("⚪  LIVE OFF")

    # ── 外部状态同步 ──────────────────────────────────────────────────
    def set_active(self, active: bool):
        self._active = active
        self._apply_style(active)

    # ── 点击事件 ─────────────────────────────────────────────────────
    def _on_click(self):
        new_state = not self._active
        self.set_active(new_state)
        self._toggle_callback(new_state)

    # ── 拖动支持 ─────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None


class VadMonitorWidget(QWidget):
    """悬浮 VAD 能量监控，仅在 Live API 激活时显示。"""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(160, 28)
        self._label = QLabel("🎤 --", self)
        self._label.setFixedSize(160, 28)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            "background:#222; color:#aaa; font-size:11px; border-radius:6px; padding:0 6px;"
        )
        self._drag_pos = None
        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._refresh)

        try:
            geo = QDesktopWidget().availableGeometry()
            self.move(geo.right() - 180, geo.bottom() - 120)
        except Exception:
            self.move(1100, 750)

    def _refresh(self):
        try:
            mm = sys.modules.get('multimodal.controller')
            mgr = getattr(mm, '_mm_manager', None) if mm else None
            energy = getattr(mgr, '_last_audio_energy', 0.0)
            state = getattr(mgr, 'state', 'IDLE')
            speaking = state == 'LISTENING_TO_USER'
            icon = "🟢" if speaking else "🎤"
            color = "#4CAF50" if speaking else "#aaa"
            self._label.setStyleSheet(
                f"background:#222; color:{color}; font-size:11px; border-radius:6px; padding:0 6px;"
            )
            self._label.setText(f"{icon} {int(energy):5d}  {state[:12]}")
        except Exception:
            pass

    def set_active(self, active: bool):
        if active:
            self.show()
            self._timer.start()
        else:
            self._timer.stop()
            self.hide()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(e.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None


class CharacterInterface(QWidget):
    """角色渲染界面 — 嵌入 PixiJS 渲染引擎（支持 Live2D 和静态帧两种后端）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("characterInterface")
        self._engine = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            from render.engine import RenderEngine
            self._engine = RenderEngine()
            widget = self._engine.start()
            layout.addWidget(widget)
            self._auto_load_kur1or3()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[CharacterInterface] 渲染引擎初始化失败: {e}")
            label = QLabel(f"渲染引擎不可用\n{e}", self)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

    def _auto_load_kur1or3(self):
        """自动加载 render/assets/images/ 中的 kurisu_*.png。"""
        import os
        images_dir = os.path.join(os.path.dirname(__file__), "render", "assets", "images")
        if os.path.isdir(images_dir):
            try:
                self._engine.load_kur1or3_sprites(images_dir)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"[CharacterInterface] 加载 sprite 失败: {e}")

    def engine(self):
        return self._engine


class EmotionPresetsInterface(QWidget):
    """Emotion preset editor — live read/write emotion_presets.json + hot-reload."""

    # (field_key, display_label, min_float, max_float, step_float)
    _FIELDS = [
        ("fade_in_sec",            "Fade In",           0.05, 5.0,   0.05),
        ("fade_out_sec",           "Fade Out",          0.05, 5.0,   0.05),
        ("transition_overlap_sec", "Transition Overlap", 0.0,  3.0,   0.05),
        ("idle_return_delay_sec",  "Idle Return Delay", 0.5,  120.0, 0.5),
    ]
    _SLIDER_STYLE = """
        QSlider::groove:horizontal {
            height: 4px; background: #E0E0E0; border-radius: 2px;
        }
        QSlider::sub-page:horizontal {
            background: #0078D4; border-radius: 2px;
        }
        QSlider::handle:horizontal {
            width: 16px; height: 16px; margin: -6px 0;
            background: white; border: 2px solid #0078D4; border-radius: 8px;
        }
        QSlider::handle:horizontal:hover { background: #E3F2FD; }
        QSlider::groove:horizontal:disabled { background: #EBEBEB; }
        QSlider::sub-page:horizontal:disabled { background: #C8C8C8; }
        QSlider::handle:horizontal:disabled { border-color: #C8C8C8; }
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("EmotionPresetsInterface")
        self._current_preset: str | None = None
        self._data: dict = {}
        # field → (QSlider, QLabel_value)
        self._sliders: dict[str, tuple] = {}
        self._auto_return_chk: QCheckBox | None = None
        self._init_ui()
        self._load_presets()

    # ------------------------------------------------------------------
    # Path & I/O
    # ------------------------------------------------------------------
    def _preset_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_presets.json")

    def _load_presets(self):
        import json
        try:
            with open(self._preset_path(), encoding="utf-8") as f:
                raw = json.load(f)
            self._data = {k: v for k, v in raw.items() if not k.startswith("_")}
        except Exception:
            self._data = {}
        self._preset_list.clear()
        for name in self._data:
            self._preset_list.addItem(name)
        if self._preset_list.count():
            self._preset_list.setCurrentRow(0)
            self._on_preset_selected(self._preset_list.item(0))

    def _save_and_reload(self):
        import json
        try:
            with open(self._preset_path(), encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
        for k, v in self._data.items():
            raw[k] = v
        try:
            with open(self._preset_path(), "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[EmotionGUI] save failed: {e}")
            return
        try:
            from vts.expression_controller import get_controller
            get_controller().load_registry(self._preset_path())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers: float ↔ slider int
    # ------------------------------------------------------------------
    @staticmethod
    def _to_int(value: float, step: float) -> int:
        return round(value / step)

    @staticmethod
    def _to_float(tick: int, step: float) -> float:
        return round(tick * step, 3)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: preset list ──────────────────────────────────────────
        left = QFrame(self)
        left.setFixedWidth(190)
        left.setStyleSheet("background:#FAFAFA; border-right:1px solid #E0E0E0;")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 16, 0, 16)
        left_lay.setSpacing(4)

        list_title = QLabel("Presets", left)
        list_title.setStyleSheet(
            "font-size:13px; font-weight:bold; color:#555; padding:0 14px 8px 14px;"
        )
        left_lay.addWidget(list_title)

        self._preset_list = QListWidget(left)
        self._preset_list.setStyleSheet("""
            QListWidget { border:none; background:transparent; }
            QListWidget::item { padding:10px 16px; font-size:13px; border-radius:4px; }
            QListWidget::item:selected { background:#E3F2FD; color:#0078D4; }
            QListWidget::item:hover:!selected { background:#F5F5F5; }
        """)
        self._preset_list.itemClicked.connect(self._on_preset_selected)
        left_lay.addWidget(self._preset_list, 1)
        root.addWidget(left)

        # ── Right: detail panel ────────────────────────────────────────
        right = QWidget(self)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(36, 28, 36, 28)
        right_lay.setSpacing(0)

        # Title row + Test button
        title_row = QHBoxLayout()
        self._detail_title = SubtitleLabel("Select a preset", right)
        setFont(self._detail_title, 18)
        self._test_btn = PushButton(FIF.PLAY, "Test", right)
        self._test_btn.setFixedSize(80, 32)
        self._test_btn.setEnabled(False)
        self._test_btn.clicked.connect(self._on_test)
        title_row.addWidget(self._detail_title)
        title_row.addStretch(1)
        title_row.addWidget(self._test_btn)
        right_lay.addLayout(title_row)

        sep = QFrame(right)
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#E8E8E8; margin:12px 0;")
        right_lay.addWidget(sep)
        right_lay.addSpacing(8)

        # Slider form
        form = QFrame(right)
        form_lay = QVBoxLayout(form)
        form_lay.setContentsMargins(0, 0, 0, 0)
        form_lay.setSpacing(20)

        for field, label, fmin, fmax, step in self._FIELDS:
            block = QVBoxLayout()
            block.setSpacing(6)

            # Label row: name on left, current value on right
            label_row = QHBoxLayout()
            lbl = QLabel(label, form)
            lbl.setStyleSheet("font-size:13px; color:#444; font-weight:500;")
            val_lbl = QLabel("—", form)
            val_lbl.setFixedWidth(70)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet("font-size:13px; color:#0078D4; font-weight:bold;")
            label_row.addWidget(lbl)
            label_row.addStretch(1)
            label_row.addWidget(val_lbl)
            block.addLayout(label_row)

            # Slider row: min label, slider, max label
            slider_row = QHBoxLayout()
            slider_row.setSpacing(8)
            min_lbl = QLabel(f"{fmin:.1f}s", form)
            min_lbl.setStyleSheet("font-size:11px; color:#AAA;")
            max_lbl = QLabel(f"{fmax:.0f}s", form)
            max_lbl.setStyleSheet("font-size:11px; color:#AAA;")

            slider = QSlider(Qt.Horizontal, form)
            int_min = self._to_int(fmin, step) if fmin > 0 else 0
            int_max = self._to_int(fmax, step)
            slider.setRange(int_min, int_max)
            slider.setSingleStep(1)
            slider.setPageStep(max(1, int_max // 20))
            slider.setStyleSheet(self._SLIDER_STYLE)
            slider.setProperty("emo_field", field)
            slider.setProperty("emo_step", step)
            # store val_lbl reference on slider for update
            slider.setProperty("val_lbl_id", id(val_lbl))
            slider.valueChanged.connect(self._on_slider_changed)

            self._sliders[field] = (slider, val_lbl)

            slider_row.addWidget(min_lbl)
            slider_row.addWidget(slider, 1)
            slider_row.addWidget(max_lbl)
            block.addLayout(slider_row)

            form_lay.addLayout(block)

        # Auto-return-to-idle checkbox
        chk_block = QVBoxLayout()
        chk_block.setSpacing(4)
        chk_header = QLabel("Auto Return to Idle", form)
        chk_header.setStyleSheet("font-size:13px; color:#444; font-weight:500;")
        chk_block.addWidget(chk_header)

        chk_row = QHBoxLayout()
        chk_row.setSpacing(10)
        self._auto_return_chk = QCheckBox("Enable", form)
        self._auto_return_chk.setStyleSheet("QCheckBox { font-size:13px; color:#333; }")
        self._auto_return_chk.stateChanged.connect(self._on_auto_return_changed)
        chk_desc = QLabel("Fade out after delay — otherwise stays until next turn", form)
        chk_desc.setStyleSheet("font-size:11px; color:#999;")
        chk_row.addWidget(self._auto_return_chk)
        chk_row.addWidget(chk_desc)
        chk_row.addStretch(1)
        chk_block.addLayout(chk_row)
        form_lay.addLayout(chk_block)

        right_lay.addWidget(form)
        right_lay.addStretch(1)

        note = QLabel("Changes are saved instantly to emotion_presets.json and hot-reloaded.", right)
        note.setStyleSheet("font-size:11px; color:#BBBBBB;")
        right_lay.addWidget(note)

        root.addWidget(right, 1)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_preset_selected(self, item: QListWidgetItem):
        name = item.text()
        self._current_preset = name
        cfg = self._data.get(name, {})

        self._detail_title.setText(f"Preset: {name}")
        self._test_btn.setEnabled(True)

        for field, label, fmin, fmax, step in self._FIELDS:
            slider, val_lbl = self._sliders[field]
            raw_val = float(cfg.get(field, step))
            tick = self._to_int(raw_val, step)
            slider.blockSignals(True)
            slider.setValue(tick)
            slider.blockSignals(False)
            val_lbl.setText(f"{raw_val:.2f} s")

        self._auto_return_chk.blockSignals(True)
        self._auto_return_chk.setChecked(bool(cfg.get("auto_return_to_idle", False)))
        self._auto_return_chk.blockSignals(False)
        self._refresh_delay_enabled()

    def _refresh_delay_enabled(self):
        enabled = self._auto_return_chk.isChecked() if self._auto_return_chk else False
        pair = self._sliders.get("idle_return_delay_sec")
        if pair:
            pair[0].setEnabled(enabled)

    def _on_slider_changed(self, tick: int):
        if not self._current_preset:
            return
        slider = self.sender()
        field = slider.property("emo_field")
        step = slider.property("emo_step")
        val = self._to_float(tick, step)

        # Update value label
        pair = self._sliders.get(field)
        if pair:
            pair[1].setText(f"{val:.2f} s")

        if field and self._current_preset in self._data:
            self._data[self._current_preset][field] = val
            self._save_and_reload()

    def _on_auto_return_changed(self, state: int):
        if not self._current_preset:
            return
        checked = (state == Qt.Checked)
        if self._current_preset in self._data:
            self._data[self._current_preset]["auto_return_to_idle"] = checked
            self._save_and_reload()
        self._refresh_delay_enabled()

    def _on_test(self):
        if not self._current_preset:
            return
        try:
            from vts.expression_controller import get_controller
            get_controller().transition_to(self._current_preset)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[EmotionGUI] test failed: {e}")


class SubtitleWindow(FluentWindow):
    def __init__(self, callback_func, asr_callback_func=None):
        super().__init__()
        self.callback_func = callback_func
        self.asr_callback_func = asr_callback_func
        
        self.setWindowTitle("VTS Assistant")
        self.resize(1000, 750)
        setTheme(Theme.LIGHT)
        setThemeColor('#000000') # Set primary color to black
        
        self.signals = SignalBridge()
        self.signals.response_signal.connect(self._update_response)
        self.signals.status_signal.connect(self._update_status)
        self.signals.user_input_signal.connect(self._update_user_input)
        
        self.chatInterface = ChatInterface(self._on_send, self._on_asr, self)
        self.settingInterface = SettingInterface(self)
        self.openclawInterface = OpenClawInterface(self)
        self.emotionPresetsInterface = EmotionPresetsInterface(self)
        # characterInterface 已合并到 chatInterface，此处保留别名供外部访问 engine()
        self.characterInterface = self.chatInterface

        self.signals.openclaw_signal.connect(self._on_openclaw_event)
        self.signals.title_signal.connect(self._on_title_updated)
        self.signals.asr_recognized_signal.connect(self._on_asr_recognized)
        self.initNavigation()

        # 悬浮 Live API 快捷按钮（始终置顶，独立于主窗口）
        self._live_float_btn = LiveApiFloatButton(self._on_live_api_float_toggle)
        self._live_float_btn.show()
        # 悬浮 VAD 能量监控（Live API 激活时显示）
        self._vad_monitor = VadMonitorWidget()
        
    def initNavigation(self):
        self.addSubInterface(self.chatInterface, FIF.CHAT, 'Chat')
        self.addSubInterface(self.openclawInterface, FIF.COMMAND_PROMPT, 'OpenClaw')
        self.addSubInterface(self.emotionPresetsInterface, FIF.PALETTE, '表情预设')
        self.addSubInterface(self.settingInterface, FIF.SETTING, 'Settings', NavigationItemPosition.BOTTOM)
        # 等 UI 完全就绪后再自动恢复会话（避免初始化顺序问题）
        QTimer.singleShot(300, self._auto_restore_session)
        # 同步 Live API 初始状态到悬浮按钮
        QTimer.singleShot(400, self._sync_live_btn_initial)
        # 注：QWebEngineView 已改为懒加载（首次点击渲染模式时才创建），
        # 所以此处不再需要提前 repaint。repaint 在 engine.on_ready 回调中触发。

    def _force_navbar_repaint(self):
        """QWebEngineView 启动时会导致 FluentWindow 导航栏不刷新变黑，强制 repaint 修复。"""
        try:
            self.update()
            self.repaint()
            # 遍历子控件找到 navigationInterface 并强制刷新
            for child in self.findChildren(QWidget):
                name = child.objectName()
                if "navigation" in name.lower() or "nav" in name.lower():
                    child.update()
                    child.repaint()
                    break
        except Exception:
            pass

    def _sync_live_btn_initial(self):
        """启动后同步一次 Live API 初始状态到悬浮按钮。"""
        main_module = get_main_module()
        if main_module and hasattr(self, '_live_float_btn'):
            _live = _get_live_mod()
            active = (
                getattr(_live, 'LIVE_API_ENABLED', False) if _live
                else getattr(main_module, 'LIVE_API_ENABLED', False)
            )
            self._live_float_btn.set_active(active)
            if hasattr(self, '_vad_monitor'):
                self._vad_monitor.set_active(active)

    def _auto_restore_session(self):
        """
        启动时自动恢复上次会话：
        - 若有已保存的会话，加载最近一条并渲染历史
        - 若没有，自动新建一条并启用连续对话
        """
        main_module = get_main_module()
        if not main_module:
            return
        sessions = main_module.list_sessions()
        if sessions:
            last_sid = sessions[-1]          # 按字母序排列，时间戳命名时即最新
            if main_module.load_session(last_sid):
                # load_session 已从文件恢复 ENABLE_CONVERSATION，强制置 True
                main_module.ENABLE_CONVERSATION = True
                self.settingInterface.convCard.setChecked(True)
                self.chatInterface.reload_sessions()
                self.chatInterface._select_session_in_list(last_sid)
                self.chatInterface._render_history()
        else:
            import datetime
            new_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            main_module.create_session(new_id)
            main_module.ENABLE_CONVERSATION = True
            self.settingInterface.convCard.setChecked(True)
            self.chatInterface.reload_sessions()
            self.chatInterface._select_session_in_list(new_id)
        
    def _on_send(self, text):
        self.handle_status("Processing...")
        asyncio.create_task(self._process_message_async(text))
        
    async def _process_message_async(self, text):
        try:
            await self.callback_func(text, self.handle_response)
            main_module = get_main_module()
            if main_module and main_module.ENABLE_CONVERSATION:
                main_module.save_session()
                # 首次完整对话后（仅 1 条 user + 1 条 assistant）异步生成标题
                dialog = getattr(getattr(main_module, 'conversation_history', None), 'dialog', [])
                if (len(dialog) == 2
                        and hasattr(main_module, 'generate_session_title')
                        and hasattr(main_module, 'set_session_title')):
                    session_id = main_module.get_current_session_id()
                    asyncio.create_task(self._generate_title_async(session_id, text))
        except Exception as e:
            self.handle_status(f"Error: {e}")
        finally:
            self.handle_status("Ready")

    async def _generate_title_async(self, session_id: str, first_message: str):
        """后台调用 DeepSeek 生成会话标题，完成后通过信号更新 UI。"""
        main_module = get_main_module()
        if not main_module:
            return
        try:
            title = await main_module.generate_session_title(first_message)
            if title:
                main_module.set_session_title(session_id, title)
                self.signals.title_signal.emit(session_id, title)
        except Exception:
            pass

    def _on_asr(self):
        # Live API 开启时 sidecar 已占用麦克风，跳过 ASR
        main_module = get_main_module()
        _live = _get_live_mod()
        _live_enabled = (
            getattr(_live, 'LIVE_API_ENABLED', False) if _live
            else getattr(main_module, 'LIVE_API_ENABLED', False)
        )
        if main_module and _live_enabled:
            InfoBar.warning(
                title='Live API 运行中',
                content='麦克风已由 Live API 占用，请直接说话即可。',
                orient=Qt.Horizontal,
                position=InfoBarPosition.TOP,
                duration=2500,
                parent=self,
            )
            return
        if self.asr_callback_func:
            self.handle_status("Listening...")
            self.chatInterface.input_bar.mic_button.setEnabled(False)
            threading.Thread(target=self._process_asr_thread, daemon=True).start()

    def _process_asr_thread(self):
        """后台线程：只做阻塞式语音采集，识别结果通过信号发回主线程走正常发送流程。"""
        try:
            main_module = get_main_module()
            if main_module and hasattr(main_module, 'asr_listen_sync'):
                text = main_module.asr_listen_sync()
            else:
                # 降级：直接跑旧的完整协程（可能出现事件循环冲突，仅作兜底）
                import asyncio as _asyncio
                loop = _asyncio.new_event_loop()
                _asyncio.set_event_loop(loop)
                loop.run_until_complete(self.asr_callback_func())
                loop.close()
                return
            if text:
                self.signals.asr_recognized_signal.emit(text)
            else:
                self.handle_status("Ready")
        except Exception as e:
            self.handle_status(f"ASR Error: {e}")
        finally:
            self.chatInterface.input_bar.mic_button.setEnabled(True)

    # Methods called by main.py or signals
    def handle_response(self, text):
        self.signals.response_signal.emit(text)
        
    def _update_response(self, text):
        self.chatInterface.update_assistant_bubble(text)
        
    def handle_status(self, status_text):
        self.signals.status_signal.emit(status_text)
        
    def _update_status(self, text):
        # We can use InfoBar or a custom status label. For now, let's use InfoBar for errors, or just ignore normal status
        if "Error" in text or "エラー" in text:
            InfoBar.error(title='Status', content=text, orient=Qt.Horizontal, position=InfoBarPosition.TOP, duration=3000, parent=self)
            
    def handle_user_input(self, text):
        self.signals.user_input_signal.emit(text)
        
    def _update_user_input(self, text):
        self.chatInterface.append_user_bubble(text)
        self.chatInterface.create_assistant_bubble()

    # ── OpenClaw 事件接口 ────────────────────────────────────────────
    def handle_openclaw_event(self, event_type: str, task_id: str, data: str):
        """线程安全：由 main.py 的后台协程调用，通过信号转发到 Qt 主线程"""
        self.signals.openclaw_signal.emit(event_type, task_id, data)

    def _on_openclaw_event(self, event_type: str, task_id: str, data: str):
        """Qt 主线程处理 OpenClaw 事件，更新面板"""
        if event_type == "start":
            self.openclawInterface.add_task(task_id, data)
        elif event_type == "done":
            self.openclawInterface.update_task(task_id, "done", data)
        elif event_type == "error":
            self.openclawInterface.update_task(task_id, "error", data)

    def _on_asr_recognized(self, text: str):
        """ASR 识别完成后在主线程执行，走与文字输入完全相同的处理路径。"""
        self.chatInterface.append_user_bubble(text)
        self.chatInterface.create_assistant_bubble()
        self._on_send(text)

    def _on_title_updated(self, session_id: str, title: str):
        """收到 DeepSeek 生成的标题后，刷新侧边栏对应条目。"""
        self.chatInterface.update_session_title_in_list(session_id, title)

    # ── 悬浮按钮回调 ─────────────────────────────────────────────────
    def _on_live_api_float_toggle(self, new_state: bool):
        """悬浮按钮点击 → 同步到 Settings 开关 → 触发实际的 Live API 开关逻辑。"""
        # 更新 Settings 面板里的开关（会触发 _on_live_api_changed）
        self.settingInterface.liveApiCard.setChecked(new_state)
        # 同步聊天输入栏状态
        self.chatInterface.set_live_api_active(new_state)

    def sync_live_float_btn(self, active: bool):
        """当 Settings 里的开关被手动拨动时，同步悬浮按钮显示。"""
        if hasattr(self, '_live_float_btn'):
            self._live_float_btn.set_active(active)
        if hasattr(self, '_vad_monitor'):
            self._vad_monitor.set_active(active)

def launch_subtitle_gui(app, callback_func, asr_callback_func=None):
    window = SubtitleWindow(callback_func, asr_callback_func)
    window.show()
    window.raise_()
    window.activateWindow()
    return window
