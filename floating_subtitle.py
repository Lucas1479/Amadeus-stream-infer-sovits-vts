import tkinter as tk
from tkinter import ttk
import threading
import time
import requests
import json
from typing import Optional, Callable

class FloatingSubtitleWindow:
    def __init__(self, gemini_api_key: str, gemini_model: str = "gemini-2.5-flash-preview-05-20"):
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏主窗口
        
        # 创建悬浮窗
        self.subtitle_window = tk.Toplevel(self.root)
        self.subtitle_window.overrideredirect(True)  # 无边框
        self.subtitle_window.attributes('-topmost', True)  # 始终置顶
        self.subtitle_window.attributes('-alpha', 0.95)  # 半透明
        
        # Gemini翻译配置
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.translate_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        # 字幕模式：bilingual, japanese_only, chinese_only
        self.display_mode = "bilingual"
        
        # 拖拽相关
        self.drag_data = {"x": 0, "y": 0}
        self.is_dragging = False
        
        # 窗口状态
        self.is_minimized = False
        self.original_height = 180  # 增加最小高度，适合4K屏幕显示完整内容
        
        self.setup_ui()
        self.setup_drag_handlers()
        
    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        self.main_frame = tk.Frame(self.subtitle_window, bg='#2C2C2C', relief='raised', bd=2)
        self.main_frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        # 🎯 字幕内容区域（移到上方）
        self.content_frame = tk.Frame(self.main_frame, bg='#2C2C2C')
        self.content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 日语原文 - 🎯 调整字体大小适合4K屏幕，居中显示
        self.japanese_label = tk.Label(
            self.content_frame,
            text="",
            bg='#2C2C2C',
            fg='#FFFFFF',
            font=('Microsoft YaHei', 18, 'bold'),  # 减小字体从28到20，更适合4K屏幕
            wraplength=2000,  # 🎯 增加换行长度，支持4K屏幕横向显示
            justify='center'  # 🎯 改为居中显示
        )
        self.japanese_label.pack(anchor='center', pady=(0, 10))  # 🎯 改为居中对齐，减少间距
        
        # 中文翻译 - 🎯 调整字体大小适合4K屏幕，居中显示
        self.chinese_label = tk.Label(
            self.content_frame,
            text="",
            bg='#2C2C2C',
            fg='#CCCCCC',
            font=('Microsoft YaHei', 12),  # 减小字体从16到14，更适合4K屏幕
            wraplength=2000,  # 🎯 增加换行长度，支持4K屏幕横向显示
            justify='center'  # 🎯 改为居中显示
        )
        self.chinese_label.pack(anchor='center')  # 🎯 改为居中对齐
        
        # 🎯 底部标题栏（可拖拽区域）
        self.title_frame = tk.Frame(self.main_frame, bg='#404040', height=40)  # 增加高度
        self.title_frame.pack(fill='x', side='bottom')
        self.title_frame.pack_propagate(False)
        
        # 🎯 移除标题标签，只保留控制按钮
        
        # 控制按钮 - 🎯 居中显示
        self.control_frame = tk.Frame(self.title_frame, bg='#404040')
        self.control_frame.pack(expand=True)  # 🎯 居中显示
        
        # 模式切换按钮
        self.mode_btn = tk.Button(
            self.control_frame,
            text="[M]",
            bg='#606060',
            fg='white',
            font=('Arial', 10, 'bold'),  # 调整按钮字体大小
            width=4,
            height=1,
            command=self.toggle_mode,
            relief='flat'
        )
        self.mode_btn.pack(side='left', padx=2)
        
        # 最小化按钮
        self.minimize_btn = tk.Button(
            self.control_frame,
            text="—",
            bg='#606060',
            fg='white',
            font=('Arial', 10, 'bold'),  # 调整按钮字体大小
            width=4,
            height=1,
            command=self.toggle_minimize,
            relief='flat'
        )
        self.minimize_btn.pack(side='left', padx=2)
        
        # 关闭按钮
        self.close_btn = tk.Button(
            self.control_frame,
            text="×",
            bg='#E74C3C',
            fg='white',
            font=('Arial', 10, 'bold'),  # 调整按钮字体大小
            width=4,
            height=1,
            command=self.close_window,
            relief='flat'
        )
        self.close_btn.pack(side='left', padx=2)
        
        # 🎯 设置更大的初始窗口大小，适合4K屏幕横向显示
        self.subtitle_window.geometry("2400x180+100+100")  # 更宽更矮，适合4K屏幕横向显示完整句子
        
        # 添加测试文本
        self.japanese_text = "字幕窗测试中..."
        self.chinese_text = "Subtitle window testing..."
        self.update_display()
        
    def setup_drag_handlers(self):
        """设置拖拽处理"""
        self.title_frame.bind("<Button-1>", self.start_drag)
        self.title_frame.bind("<B1-Motion>", self.drag_window)
        self.title_frame.bind("<ButtonRelease-1>", self.stop_drag)
        
        # 鼠标悬停效果
        self.title_frame.bind("<Enter>", lambda e: self.title_frame.config(bg='#505050'))
        self.title_frame.bind("<Leave>", lambda e: self.title_frame.config(bg='#404040'))
        
    def start_drag(self, event):
        """开始拖拽"""
        self.is_dragging = True
        self.drag_data["x"] = event.x_root - self.subtitle_window.winfo_x()
        self.drag_data["y"] = event.y_root - self.subtitle_window.winfo_y()
        
    def drag_window(self, event):
        """拖拽窗口"""
        if self.is_dragging:
            x = event.x_root - self.drag_data["x"]
            y = event.y_root - self.drag_data["y"]
            self.subtitle_window.geometry(f"+{x}+{y}")
            
    def stop_drag(self, event):
        """停止拖拽"""
        self.is_dragging = False
        
    def toggle_mode(self):
        """切换显示模式"""
        modes = ["bilingual", "japanese_only", "chinese_only"]
        current_index = modes.index(self.display_mode)
        self.display_mode = modes[(current_index + 1) % len(modes)]
        
        # 更新按钮文本
        mode_texts = {"bilingual": "[M]", "japanese_only": "[J]", "chinese_only": "[C]"}
        self.mode_btn.config(text=mode_texts[self.display_mode])
        
        # 更新显示
        self.update_display()
        
    def toggle_minimize(self):
        """切换最小化状态"""
        self.is_minimized = not self.is_minimized
        
        if self.is_minimized:
            # 最小化：只显示标题栏
            self.content_frame.pack_forget()
            self.subtitle_window.geometry("450x25")
        else:
            # 恢复：显示完整内容
            self.content_frame.pack(fill='both', expand=True, padx=5, pady=5)
            self.subtitle_window.geometry("900x250")  # 🎯 恢复时使用更大的窗口
            
    def close_window(self):
        """关闭窗口"""
        self.root.quit()
        self.root.destroy()
        
    def translate_text(self, japanese_text: str) -> str:
        """使用Gemini API翻译日语到中文"""
        try:
            # 文本已经在main.py中清理过了，直接翻译
            if not japanese_text.strip():
                return ""
            
            # 检查翻译缓存
            if hasattr(self, 'translation_cache'):
                cache_key = japanese_text.strip()
                if cache_key in self.translation_cache:
                    print(f"DEBUG: 使用缓存翻译: '{japanese_text[:20]}...'")
                    return self.translation_cache[cache_key]
            else:
                self.translation_cache = {}
            
            # 检查请求频率限制
            current_time = time.time()
            if hasattr(self, 'last_request_time'):
                time_since_last = current_time - self.last_request_time
                if time_since_last < 1.0:  # 至少1秒间隔
                    print(f"DEBUG: 请求过于频繁，等待 {1.0 - time_since_last:.2f} 秒")
                    time.sleep(1.0 - time_since_last)
            
            self.last_request_time = time.time()
            
            print(f"DEBUG: 开始翻译文本: '{japanese_text}'")
            
            url = f"{self.translate_base_url}/{self.gemini_model}:generateContent"
            print(f"DEBUG: API URL: {url}")
            
            # 使用传入的API密钥
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_api_key
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": f"你是命运石之门中的Amadeus红莉栖，把你说的日文翻译成中文。只需提供译文，无需任何解释说明。\n\n日文：{japanese_text}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 200
                }
            }
            
            print(f"DEBUG: 发送请求到Gemini API")
            
            # 添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=10)
                    print(f"DEBUG: API响应状态码: {response.status_code}")
                    
                    if response.status_code == 429:
                        # 429错误，等待更长时间后重试
                        wait_time = (2 ** attempt) * 2  # 指数退避：2秒、4秒、8秒
                        print(f"DEBUG: 收到429错误，等待 {wait_time} 秒后重试 (尝试 {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            time.sleep(wait_time)
                            continue
                        else:
                            print("DEBUG: 达到最大重试次数，返回翻译失败")
                            return "翻译失败"
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    print(f"DEBUG: API响应内容: {result}")
                    
                    if "candidates" in result and len(result["candidates"]) > 0:
                        translated_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                        print(f"DEBUG: 翻译结果: '{translated_text}'")
                        
                        # 缓存翻译结果
                        self.translation_cache[japanese_text.strip()] = translated_text
                        
                        return translated_text
                    else:
                        print("DEBUG: API响应中没有candidates")
                        return "翻译失败"
                        
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429 and attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2
                        print(f"DEBUG: HTTP错误 {e.response.status_code}，等待 {wait_time} 秒后重试")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
                        
            return "翻译失败"
                
        except Exception as e:
            print(f"翻译错误: {e}")
            import traceback
            traceback.print_exc()
            return "翻译失败"
            
    def update_subtitle(self, japanese_text: str):
        """更新字幕内容"""
        if not japanese_text.strip():
            return
            
        # 更新日语原文
        self.japanese_text = japanese_text
        
        # 检查是否需要翻译（更严格的频率控制）
        current_time = time.time()
        if not hasattr(self, 'last_translate_time'):
            self.last_translate_time = 0
        
        # 更严格的翻译条件：完整句子 + 时间间隔 + 长度要求
        should_translate = (
            current_time - self.last_translate_time > 5.0 and  # 至少5秒间隔（进一步减少请求）
            japanese_text.endswith(('。', '！', '？', '.', '!', '?')) and  # 必须是句子结尾
            len(japanese_text.strip()) >= 15 and  # 至少15个字符（增加长度要求）
            (not hasattr(self, 'last_translated_text') or 
             len(japanese_text) - len(self.last_translated_text) > 15)  # 文本增长超过15个字符
        )
        
        if should_translate:
            self.last_translate_time = current_time
            self.last_translated_text = japanese_text
            
            # 异步翻译（不阻塞主流程）
            def translate_async():
                try:
                    chinese_text = self.translate_text(japanese_text)
                    if chinese_text and chinese_text != "翻译失败":
                        self.chinese_text = chinese_text
                        # 在主线程中更新显示
                        self.root.after(0, self.update_display)
                except Exception as e:
                    print(f"翻译异常: {e}")
            
            # 在后台线程中执行翻译
            import threading
            threading.Thread(target=translate_async, daemon=True).start()
        else:
            # 如果不需要翻译，直接更新显示
            self.update_display()
        
    def update_subtitle_with_progress(self, japanese_text: str, chinese_text: str, display_chars: int):
        """更新字幕内容，支持逐字显示"""
        if not japanese_text.strip():
            return
            
        # 更新日语原文
        self.japanese_text = japanese_text
        self.chinese_text = chinese_text
        
        # 根据显示字符数截取文本
        display_japanese = japanese_text[:display_chars] if display_chars <= len(japanese_text) else japanese_text
        display_chinese = chinese_text[:display_chars] if display_chars <= len(chinese_text) else chinese_text
        
        # 立即更新显示
        self._update_display_with_text(display_japanese, display_chinese)
        
    def update_display(self):
        """更新显示内容"""
        if not hasattr(self, 'japanese_text'):
            return
            
        # 根据模式显示内容
        if self.display_mode == "japanese_only":
            self.japanese_label.config(text=self.japanese_text)
            self.chinese_label.config(text="")
        elif self.display_mode == "chinese_only":
            self.japanese_label.config(text="")
            chinese_text = getattr(self, 'chinese_text', "翻译中...")
            self.chinese_label.config(text=chinese_text)
        else:  # bilingual
            self.japanese_label.config(text=self.japanese_text)
            chinese_text = getattr(self, 'chinese_text', "翻译中...")
            self.chinese_label.config(text=chinese_text)
            
    def _update_display_with_text(self, japanese_text: str, chinese_text: str):
        """使用指定文本更新显示内容"""
        if not hasattr(self, 'japanese_text'):
            return
            
        # 根据模式显示内容
        if self.display_mode == "japanese_only":
            self.japanese_label.config(text=japanese_text)
            self.chinese_label.config(text="")
        elif self.display_mode == "chinese_only":
            self.japanese_label.config(text="")
            self.chinese_label.config(text=chinese_text)
        else:  # bilingual
            self.japanese_label.config(text=japanese_text)
            self.chinese_label.config(text=chinese_text)
    
    def update_display_simple(self, japanese_text: str, chinese_text: str = ""):
        """简化的字幕显示更新，用于实时渲染"""
        try:
            # 直接更新显示，不进行复杂的判断
            if self.display_mode == "japanese_only":
                self.japanese_label.config(text=japanese_text)
                self.chinese_label.config(text="")
            elif self.display_mode == "chinese_only":
                self.japanese_label.config(text="")
                self.chinese_label.config(text=chinese_text)
            else:  # bilingual
                self.japanese_label.config(text=japanese_text)
                self.chinese_label.config(text=chinese_text)
        except Exception as e:
            print(f"字幕显示更新失败: {e}")
            
    def run(self):
        """运行字幕窗"""
        # 显示窗口
        self.subtitle_window.deiconify()
        # 启动Tkinter主循环（非阻塞）
        self.root.mainloop()
        
    def show(self):
        """显示窗口"""
        self.subtitle_window.deiconify()
        
    def hide(self):
        """隐藏窗口"""
        self.subtitle_window.withdraw()

# 全局字幕窗实例
subtitle_window_instance = None

def init_subtitle_window(gemini_api_key: str):
    """初始化字幕窗"""
    global subtitle_window_instance
    if subtitle_window_instance is None:
        # 在单独的线程中创建Tkinter窗口
        def create_window():
            global subtitle_window_instance
            try:
                subtitle_window_instance = FloatingSubtitleWindow(gemini_api_key)
                # 启动Tkinter主循环
                subtitle_window_instance.run()
            except Exception as e:
                print(f"字幕窗初始化失败: {e}")
                import traceback
                traceback.print_exc()
                subtitle_window_instance = None
        
        # 启动Tkinter线程
        import threading
        tk_thread = threading.Thread(target=create_window, daemon=True)
        tk_thread.start()
        
        # 等待窗口创建完成
        import time
        time.sleep(0.5)
        
    return subtitle_window_instance

def update_subtitle_text(text: str):
    """更新字幕文本"""
    global subtitle_window_instance
    if subtitle_window_instance:
        subtitle_window_instance.update_subtitle(text)
