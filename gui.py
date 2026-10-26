import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

# --- DPI Awareness for Windows ---
# This is the crucial fix for sharp rendering on high-DPI displays (like 4K with 200% scaling)
if os.name == 'nt': # Only for Windows
    try:
        from ctypes import windll
        # This tells Windows that the app is DPI-aware and can handle its own scaling.
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception: # pragma: no cover
        # Fallback for older Windows versions if shcore is not available
        try:
            windll.user32.SetProcessDPIAware()
        except Exception:
            pass # If all fails, continue without DPI awareness

# 导入您现有的脚本作为核心逻辑库
try:
    import main as core
except ImportError as e:
    # 在GUI启动前，如果核心库缺失，用messagebox提示
    root = tk.Tk()
    root.withdraw() # 隐藏主窗口
    messagebox.showerror("启动错误", f"无法导入 main.py: {e}\n\n请确保 gui.py 和 main.py 在同一个目录下。")
    exit(1)

class VideoSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CC字幕生成工具")
        self.root.geometry("1000x650")

        # --- Style ---
        style = ttk.Style(self.root)
        try:
            # 'clam' 主题在各平台表现一致且美观
            style.theme_use('clam')
        except tk.TclError: # pragma: no cover
            # 如果 'clam' 不可用，则使用默认主题
            pass

        # --- Font Configuration for better Chinese rendering ---
        # 优先使用'Microsoft YaHei UI'，这是Windows上现代且清晰的中文字体
        self.default_font = ('Microsoft YaHei UI', 10)
        self.bold_font = ('Microsoft YaHei UI', 12, 'bold')
        
        # 检查字体是否存在，如果不存在则回退
        try:
            from tkinter import font
            available_fonts = list(font.families())
            if 'Microsoft YaHei UI' not in available_fonts:
                if 'Dengxian' in available_fonts:
                    self.default_font = ('Dengxian', 10)
                    self.bold_font = ('Dengxian', 12, 'bold')
                else: # pragma: no cover
                    # 如果都没有，使用一个通用备用方案
                    self.default_font = ('Arial', 10)
                    self.bold_font = ('Arial', 12, 'bold')
        except Exception: # pragma: no cover
            # 出现任何错误都使用备用方案
            self.default_font = ('Arial', 10)
            self.bold_font = ('Arial', 12, 'bold')

        # 为所有ttk组件设置默认字体
        style.configure('.', font=self.default_font)
        style.configure('TLabel', font=self.default_font)
        style.configure('TButton', font=self.default_font)
        style.configure('TRadiobutton', font=self.default_font)
        style.configure('TLabelframe.Label', font=self.bold_font)

        # --- Class variables ---
        self.source_var = tk.StringVar(value="youtube")
        self.model_var = tk.StringVar(value="small")
        self.input_lang_var = tk.StringVar(value="中文")
        self.status_var = tk.StringVar(value="就绪")
        self.processing = False
        self.text_edit_mode = False  # 文本编辑模式状态
        self.generate_cc_var = tk.BooleanVar(value=False)  # 新增：是否生成CC字幕
        self.translate_method_var = tk.StringVar(value="在线优先")

        # --- Main layout frames ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.Labelframe(main_frame, text="控制面板", padding="10")
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        output_frame = ttk.Labelframe(main_frame, text="输出", padding="10")
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Widgets for Input Frame ---
        self._create_input_widgets(input_frame)

        # --- Widgets for Output Frame ---
        self._create_output_widgets(output_frame)

        # --- Status Bar ---
        status_bar = ttk.Frame(self.root, padding=(10, 5))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_bar, text="状态:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_bar, textvariable=self.status_var, foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=5)

        # --- Initial state ---
        self._update_source_selection()
        self._update_button_states()

    def _create_input_widgets(self, parent):
        ttk.Label(parent, text="1. 选择来源和模型", font=self.bold_font).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Radiobutton(parent, text="YouTube URL", variable=self.source_var, value="youtube", command=self._update_source_selection).pack(anchor=tk.W)
        self.url_entry = ttk.Entry(parent, width=40)
        self.url_entry.pack(anchor=tk.W, fill=tk.X, pady=(0, 5))

        ttk.Radiobutton(parent, text="本地视频或音频文件", variable=self.source_var, value="local", command=self._update_source_selection).pack(anchor=tk.W)
        local_file_frame = ttk.Frame(parent)
        local_file_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
        self.file_entry = ttk.Entry(local_file_frame, width=30)
        self.file_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.browse_button = ttk.Button(local_file_frame, text="浏览...", command=self._browse_file)
        self.browse_button.pack(side=tk.LEFT, padx=(5, 0))

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(parent, text="Whisper 模型 (越大越准，也越慢)").pack(anchor=tk.W)
        model_options = ["small", "medium", "large-v3"]
        self.model_menu = ttk.OptionMenu(parent, self.model_var, self.model_var.get(), *model_options)
        self.model_menu.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))

        # 新增：输入语言选择
        ttk.Label(parent, text="输入语言").pack(anchor=tk.W)
        lang_options = ["中文", "英文", "日文", "泰文"]
        self.lang_menu = ttk.OptionMenu(parent, self.input_lang_var, self.input_lang_var.get(), *lang_options)
        self.lang_menu.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))

        # 新增：CC字幕生成选项
        cc_frame = ttk.LabelFrame(parent, text="CC字幕生成", padding="5")
        cc_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
        
        # CC字幕生成复选框
        self.cc_checkbox = ttk.Checkbutton(cc_frame, text="生成CC字幕文件 (SRT格式)", variable=self.generate_cc_var)
        self.cc_checkbox.pack(anchor=tk.W, pady=(0, 5))
        
        # CC字幕格式说明
        ttk.Label(cc_frame, text="输出格式: SRT (标准字幕格式)", foreground="gray").pack(anchor=tk.W)

        # 新增：翻译方式
        ttk.Label(cc_frame, text="翻译方式").pack(anchor=tk.W, pady=(8, 2))
        translate_options = ["不翻译", "在线优先", "离线"]
        self.translate_method_menu = ttk.OptionMenu(cc_frame, self.translate_method_var, self.translate_method_var.get(), *translate_options)
        self.translate_method_menu.pack(anchor=tk.W, fill=tk.X)

        # 新增：从本地SRT翻译为中文
        self.translate_srt_button = ttk.Button(cc_frame, text="从本地SRT翻译为中文", command=self._start_translate_srt_thread)
        self.translate_srt_button.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

        self.process_button = ttk.Button(parent, text="开始处理", command=self._start_processing_thread)
        self.process_button.pack(fill=tk.X, ipady=5, pady=(0, 10))

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(parent, text="2. 处理文本", font=self.bold_font).pack(anchor=tk.W, pady=(0, 5))
        
        # GPT 总结配置选择
        ttk.Label(parent, text="GPT 总结风格:").pack(anchor=tk.W, pady=(0, 2))
        self.gpt_preset_var = tk.StringVar(value="生动")
        gpt_preset_options = ["保守", "平衡", "生动", "创意"]
        self.gpt_preset_menu = ttk.OptionMenu(parent, self.gpt_preset_var, self.gpt_preset_var.get(), *gpt_preset_options)
        self.gpt_preset_menu.pack(anchor=tk.W, fill=tk.X, pady=(0, 5))
        
        self.summarize_gpt_button = ttk.Button(parent, text="使用 GPT-4 总结", command=lambda: self._start_summarize_thread('GPT-4'))
        self.summarize_gpt_button.pack(fill=tk.X, pady=2)

        self.save_text_button = ttk.Button(parent, text="保存完整文本", command=self._save_text)
        self.save_text_button.pack(fill=tk.X, pady=2)

        self.save_summary_button = ttk.Button(parent, text="保存总结", command=self._save_summary)
        self.save_summary_button.pack(fill=tk.X, pady=2)

        # 新增：CC字幕保存按钮
        self.save_cc_button = ttk.Button(parent, text="保存CC字幕", command=self._save_cc_subtitles)
        self.save_cc_button.pack(fill=tk.X, pady=2)

    def _create_output_widgets(self, parent):
        # 完整转录文本区域
        ttk.Label(parent, text="完整转录文本").pack(anchor=tk.W)
        
        # 文本操作工具栏
        text_toolbar = ttk.Frame(parent)
        text_toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # 加载文本文件按钮
        self.load_text_button = ttk.Button(text_toolbar, text="📁 加载文本文件", command=self._load_text_file)
        self.load_text_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 粘贴文本按钮
        self.paste_text_button = ttk.Button(text_toolbar, text="📋 粘贴文本", command=self._paste_text)
        self.paste_text_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 清空文本按钮
        self.clear_text_button = ttk.Button(text_toolbar, text="🗑️ 清空文本", command=self._clear_text)
        self.clear_text_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 文本编辑状态切换按钮
        self.edit_text_button = ttk.Button(text_toolbar, text="✏️ 编辑模式", command=self._toggle_text_edit_mode)
        self.edit_text_button.pack(side=tk.LEFT, padx=(0, 5))
        
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.text_output = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED, height=15, font=self.default_font)
        text_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_output.yview)
        self.text_output.config(yscrollcommand=text_scroll.set)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 为文本区域添加右键菜单
        self._create_text_context_menu()

        ttk.Label(parent, text="AI 总结").pack(anchor=tk.W)
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(fill=tk.BOTH, expand=True)
        self.summary_output = tk.Text(summary_frame, wrap=tk.WORD, state=tk.DISABLED, height=10, font=self.default_font)
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_output.yview)
        self.summary_output.config(yscrollcommand=summary_scroll.set)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _update_source_selection(self):
        if self.source_var.get() == "youtube":
            self.url_entry.config(state=tk.NORMAL)
            self.file_entry.config(state=tk.DISABLED)
            self.browse_button.config(state=tk.DISABLED)
        else:
            self.url_entry.config(state=tk.DISABLED)
            self.file_entry.config(state=tk.NORMAL)
            self.browse_button.config(state=tk.NORMAL)

    def _browse_file(self):
        filepath = filedialog.askopenfilename(
            title="选择视频或音频文件",
            filetypes=(
                ("Video Files", "*.mp4 *.mkv *.avi *.mov"), 
                ("Audio Files", "*.wav *.m4a *.mp3 *.ogg *.flac"),
                ("All files", "*.*")
            )
        )
        if filepath:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filepath)

    def _update_button_states(self):
        is_processing = self.processing
        has_text = bool(self.text_output.get("1.0", "end-1c").strip())
        has_summary = bool(self.summary_output.get("1.0", "end-1c").strip())
        has_cc = hasattr(self, 'cc_subtitles') and self.cc_subtitles is not None

        state_if_processing = tk.DISABLED if is_processing else tk.NORMAL
        self.process_button.config(state=state_if_processing)
        self.summarize_gpt_button.config(state=tk.DISABLED if is_processing or not has_text else tk.NORMAL)
        self.save_text_button.config(state=tk.DISABLED if is_processing or not has_text else tk.NORMAL)
        self.save_summary_button.config(state=tk.DISABLED if is_processing or not has_summary else tk.NORMAL)
        self.save_cc_button.config(state=tk.DISABLED if is_processing or not has_cc else tk.NORMAL)
        
        # 文本操作按钮状态管理
        if hasattr(self, 'load_text_button'):
            self.load_text_button.config(state=state_if_processing)
        if hasattr(self, 'paste_text_button'):
            self.paste_text_button.config(state=state_if_processing)
        if hasattr(self, 'clear_text_button'):
            self.clear_text_button.config(state=tk.DISABLED if is_processing or not has_text else tk.NORMAL)
        if hasattr(self, 'edit_text_button'):
            self.edit_text_button.config(state=state_if_processing)
        
        self.model_menu.config(state=state_if_processing)
        if hasattr(self, 'lang_menu'):
            self.lang_menu.config(state=state_if_processing)
        self.cc_checkbox.config(state=state_if_processing)
        if hasattr(self, 'translate_method_menu'):
            self.translate_method_menu.config(state=state_if_processing)
        if hasattr(self, 'translate_srt_button'):
            self.translate_srt_button.config(state=state_if_processing)
        # 移除CC格式选择相关代码
        if not is_processing:
            self._update_source_selection()
        else:
            self.url_entry.config(state=tk.DISABLED)
            self.file_entry.config(state=tk.DISABLED)
            self.browse_button.config(state=tk.DISABLED)

    def _update_status(self, message):
        self.status_var.set(message)

    def _update_text_output(self, text_widget, content):
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", content)
        text_widget.config(state=tk.DISABLED)
        self._update_button_states()

    def _save_text(self):
        content = self.text_output.get("1.0", "end-1c")
        filepath = filedialog.asksaveasfilename(
            title="保存完整文本", defaultextension=".txt",
            filetypes=(("Text Files", "*.txt"), ("All files", "*.*"))
        )
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
                self._update_status(f'文本已保存到 {os.path.basename(filepath)}')
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存文件: {e}")

    def _save_summary(self):
        content = self.summary_output.get("1.0", "end-1c")
        filepath = filedialog.asksaveasfilename(
            title="保存总结", defaultextension=".txt",
            filetypes=(("Text Files", "*.txt"), ("All files", "*.*"))
        )
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
                self._update_status(f'总结已保存到 {os.path.basename(filepath)}')
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存文件: {e}")

    def _save_cc_subtitles(self):
        """保存CC字幕文件"""
        if not hasattr(self, 'cc_subtitles') or not self.cc_subtitles:
            messagebox.showwarning("无字幕", "没有可保存的CC字幕文件")
            return
        
        # 获取选择的格式
        format_type = "srt" # 默认保存为SRT格式
        
        # 读取字幕内容
        try:
            with open(self.cc_subtitles["srt"], 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 显示字幕内容预览
            preview_window = tk.Toplevel(self.root)
            preview_window.title(f"CC字幕预览 - SRT格式")
            preview_window.geometry("800x600")
            preview_window.transient(self.root)
            preview_window.grab_set()
            
            # 创建预览界面
            preview_frame = ttk.Frame(preview_window, padding="10")
            preview_frame.pack(fill=tk.BOTH, expand=True)
            
            # 标题
            ttk.Label(preview_frame, text=f"CC字幕内容预览 (SRT格式)", 
                     font=self.bold_font).pack(pady=(0, 10))
            
            # 内容显示区域
            content_frame = ttk.Frame(preview_frame)
            content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            content_text = tk.Text(content_frame, wrap=tk.WORD, font=('Courier', 10))
            content_scroll = ttk.Scrollbar(content_frame, orient=tk.VERTICAL, command=content_text.yview)
            content_text.config(yscrollcommand=content_scroll.set)
            
            content_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            content_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # 插入字幕内容
            content_text.insert("1.0", content)
            content_text.config(state=tk.DISABLED)
            
            # 按钮区域
            button_frame = ttk.Frame(preview_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))
            
            # 保存按钮
            save_button = ttk.Button(button_frame, text="保存到文件", 
                                   command=lambda: self._save_cc_to_file(content, "srt", preview_window))
            save_button.pack(side=tk.RIGHT, padx=(5, 0))
            
            # 关闭按钮
            close_button = ttk.Button(button_frame, text="关闭", 
                                    command=preview_window.destroy)
            close_button.pack(side=tk.RIGHT)
            
            # 复制按钮
            copy_button = ttk.Button(button_frame, text="复制到剪贴板", 
                                   command=lambda: self._copy_to_clipboard(content))
            copy_button.pack(side=tk.LEFT)
            
        except Exception as e:
            messagebox.showerror("读取失败", f"无法读取CC字幕文件: {e}")

    def _save_cc_to_file(self, content, format_type, preview_window):
        """保存CC字幕到文件"""
        filepath = filedialog.asksaveasfilename(
            title="保存CC字幕", 
            defaultextension=".srt", # 默认保存为SRT格式
            filetypes=(
                ("SRT Files", "*.srt"),
                ("All files", "*.*")
            )
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self._update_status(f'CC字幕已保存到 {os.path.basename(filepath)}')
                preview_window.destroy()
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存CC字幕文件: {e}")

    def _copy_to_clipboard(self, content):
        """复制内容到剪贴板"""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self._update_status('CC字幕内容已复制到剪贴板')
        except Exception as e:
            messagebox.showerror("复制失败", f"无法复制到剪贴板: {e}")

    def _start_processing_thread(self):
        self.processing = True
        self._update_button_states()
        self._update_text_output(self.text_output, "")
        self._update_text_output(self.summary_output, "")
        # 清除之前的CC字幕
        if hasattr(self, 'cc_subtitles'):
            delattr(self, 'cc_subtitles')
        thread = threading.Thread(target=self._process_video_worker, daemon=True)
        thread.start()

    def _process_video_worker(self):
        try:
            def status_callback(message): self.root.after(0, self._update_status, message)

            # 获取文本输入
            if self.source_var.get() == "youtube":
                url = self.url_entry.get()
                if not url:
                    raise ValueError("请输入 YouTube URL。")
                status_callback("正在下载 YouTube 媒体… 这可能需要较长时间，请耐心等待")
                # 语言代码映射
                lang_map = {"中文": "zh", "英文": "en", "日文": "ja", "泰文": "th"}
                lang_code = lang_map.get(self.input_lang_var.get(), None)
                try:
                    method = self.translate_method_var.get()
                    translate_to_zh = (method != "不翻译")
                    online_provider = "openai" if method == "在线优先" else None
                    media_path, wav_path, srt_path, _orig_srt_path = core.process_youtube_to_srt(
                        url,
                        model=self.model_var.get(),
                        language=lang_code,
                        progress_callback=status_callback,
                        translate_to_zh=translate_to_zh,
                        keep_original=False,
                        online_provider=online_provider
                    )
                except Exception as e:
                    raise Exception(f"处理 YouTube 链接失败: {e}")
                # 设置 CC 字幕文件
                cc_files = {"srt": srt_path}
                self.root.after(0, self._set_cc_subtitles, cc_files)
                status_callback("SRT 字幕已生成，可点击‘保存CC字幕’进行保存")
                # 在文本区域提示
                label = "中文SRT" if translate_to_zh else "原文SRT"
                info_text_parts = [f"媒体: {media_path}", f"WAV: {wav_path}", f"{label}: {srt_path}"]
                info_text = "\n".join(info_text_parts) + "\n\n已在终端输出 SRT 内容。"
                self.root.after(0, self._update_text_output, self.text_output, info_text)
            else:
                file_path = self.file_entry.get()
                if not file_path or not os.path.exists(file_path):
                    raise ValueError("无效的本地文件路径。")
                status_callback("正在转换为 WAV 并调用 Whisper 生成 SRT… 这可能需要较长时间")
                # 语言代码映射
                lang_map = {"中文": "zh", "英文": "en", "日文": "ja", "泰文": "th"}
                lang_code = lang_map.get(self.input_lang_var.get(), None)
                try:
                    method = self.translate_method_var.get()
                    translate_to_zh = (method != "不翻译")
                    online_provider = "openai" if method == "在线优先" else None
                    wav_path, srt_path, _orig_srt_path = core.process_local_to_srt(
                        file_path,
                        model=self.model_var.get(),
                        language=lang_code,
                        progress_callback=status_callback,
                        translate_to_zh=translate_to_zh,
                        keep_original=False,
                        online_provider=online_provider
                    )
                except Exception as e:
                    raise Exception(f"处理本地文件失败: {e}")
                # 设置 CC 字幕文件
                cc_files = {"srt": srt_path}
                self.root.after(0, self._set_cc_subtitles, cc_files)
                status_callback("SRT 字幕已生成，可点击‘保存CC字幕’进行保存")
                # 在文本区域提示
                label = "中文SRT" if translate_to_zh else "原文SRT"
                info_text_parts = [f"WAV: {wav_path}", f"{label}: {srt_path}"]
                info_text = "\n".join(info_text_parts) + "\n\n已在终端输出 SRT 内容。"
                self.root.after(0, self._update_text_output, self.text_output, info_text)
        except Exception as e:
            self.root.after(0, self._update_status, f'错误: {e}')
            self.root.after(0, messagebox.showerror, "处理错误", str(e))
        finally:
            self.processing = False
            self.root.after(0, self._update_button_states)

    def _start_translate_srt_thread(self):
        # 若未选择翻译方式或选择“不翻译”，提示并返回
        method = self.translate_method_var.get()
        if method == "不翻译":
            messagebox.showwarning("未选择翻译", "请将翻译方式设置为‘在线优先’或‘离线’后再试")
            return
        self.processing = True
        self._update_button_states()
        thread = threading.Thread(target=self._translate_srt_worker, daemon=True)
        thread.start()

    def _translate_srt_worker(self):
        try:
            def status_callback(message): self.root.after(0, self._update_status, message)
            filepath = filedialog.askopenfilename(
                title="选择需要翻译的SRT文件",
                filetypes=(("SRT Files", "*.srt"), ("All files", "*.*"))
            )
            if not filepath:
                return

            method = self.translate_method_var.get()
            status_callback(f"开始翻译 SRT（{method}）…")

            zh_path = None
            if method == "在线优先":
                try:
                    zh_path = core.translate_srt_to_chinese_online(filepath, provider="openai", model="gpt-4o-mini", progress_callback=status_callback)
                except Exception as e:
                    status_callback(f"在线翻译失败，尝试离线翻译: {e}")
                    zh_path = core.translate_srt_to_chinese(filepath, progress_callback=status_callback, source_lang=None, force_all=True)
            else:  # 离线
                zh_path = core.translate_srt_to_chinese(filepath, progress_callback=status_callback, source_lang=None, force_all=True)

            # 设置并更新保存按钮可用
            cc_files = {"srt": zh_path}
            self.root.after(0, self._set_cc_subtitles, cc_files)
            info_text = f"中文SRT: {zh_path}\n\n已生成中文字幕，可点击‘保存CC字幕’进行保存。"
            self.root.after(0, self._update_text_output, self.text_output, info_text)
            status_callback("SRT 翻译完成！")
        except Exception as e:
            self.root.after(0, self._update_status, f"错误: {e}")
            self.root.after(0, messagebox.showerror, "翻译错误", str(e))
        finally:
            self.processing = False
            self.root.after(0, self._update_button_states)

    def _set_cc_subtitles(self, cc_files):
        """设置CC字幕文件路径"""
        self.cc_subtitles = cc_files
        self._update_button_states()

    def _start_summarize_thread(self, summarizer):
        self.processing = True
        self._update_button_states()
        text_to_summarize = self.text_output.get("1.0", "end-1c")
        # 获取选择的 GPT 预设配置
        gpt_preset = self.gpt_preset_var.get() if hasattr(self, 'gpt_preset_var') else "生动"
        thread = threading.Thread(target=self._summarize_worker, args=(text_to_summarize, summarizer, gpt_preset), daemon=True)
        thread.start()

    def _summarize_worker(self, text, summarizer, gpt_preset="生动"):
        try:
            preset_info = f" ({gpt_preset})" if gpt_preset else ""
            self.root.after(0, self._update_status, f'正在使用 {summarizer}{preset_info} 总结...')
            summary = None
            if summarizer == 'GPT-4':
                summary = core.summarize_with_gpt4(text, gpt_preset)
            else:
                raise Exception("不支持的总结器类型")
            
            if summary:
                self.root.after(0, self._update_text_output, self.summary_output, summary)
                self.root.after(0, self._update_status, f'总结完成！(使用 {gpt_preset} 风格)')
            else:
                raise Exception("总结失败，未返回结果。")
        except Exception as e:
            self.root.after(0, self._update_status, f'错误: {e}')
            self.root.after(0, messagebox.showerror, "总结错误", str(e))
        finally:
            self.processing = False
            self.root.after(0, self._update_button_states)

    def _load_text_file(self):
        """加载本地文本文件"""
        try:
            filepath = filedialog.askopenfilename(
                title="选择文本文件",
                filetypes=(
                    ("Text Files", "*.txt *.md *.py *.js *.html *.css"), 
                    ("All files", "*.*")
                )
            )
            if filepath:
                # 尝试不同的编码方式读取文件
                encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']
                content = None
                
                for encoding in encodings:
                    try:
                        with open(filepath, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    messagebox.showerror("读取失败", "无法读取文件，可能是编码问题")
                    return
                
                # 更新文本输出
                self._update_text_output(self.text_output, content)
                self._update_status(f'已加载文本文件: {os.path.basename(filepath)}')
                
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载文件: {e}")

    def _paste_text(self):
        """粘贴文本到文本区域"""
        try:
            # 获取剪贴板内容
            clipboard_content = self.root.clipboard_get()
            if clipboard_content:
                # 如果文本区域是只读状态，先切换到编辑模式
                if self.text_output.cget('state') == tk.DISABLED:
                    self._toggle_text_edit_mode()
                
                # 在光标位置插入文本
                self.text_output.insert(tk.INSERT, clipboard_content)
                self._update_status('已粘贴剪贴板内容')
            else:
                self._update_status('剪贴板为空')
        except tk.TclError:
            self._update_status('剪贴板为空或无法访问')
        except Exception as e:
            messagebox.showerror("粘贴失败", f"无法粘贴文本: {e}")

    def _clear_text(self):
        """清空文本区域"""
        if messagebox.askyesno("确认清空", "确定要清空所有文本吗？"):
            self._update_text_output(self.text_output, "")
            self._update_status('文本已清空')

    def _toggle_text_edit_mode(self):
        """切换文本编辑模式"""
        if self.text_edit_mode:
            # 切换到只读模式
            self.text_output.config(state=tk.DISABLED)
            self.edit_text_button.config(text="✏️ 编辑模式")
            self.text_edit_mode = False
            self._update_status('文本区域已设为只读模式')
        else:
            # 切换到编辑模式
            self.text_output.config(state=tk.NORMAL)
            self.edit_text_button.config(text="🔒 只读模式")
            self.text_edit_mode = True
            self._update_status('文本区域已设为编辑模式')

    def _update_text_output(self, text_widget, content):
        """更新文本输出，保持编辑状态"""
        current_state = text_widget.cget('state')
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", content)
        
        # 恢复原来的状态
        if not self.text_edit_mode:
            text_widget.config(state=tk.DISABLED)
        
        self._update_button_states()

    def _create_text_context_menu(self):
        """创建文本区域的右键菜单"""
        self.text_context_menu = tk.Menu(self.root, tearoff=0)
        self.text_context_menu.add_command(label="📁 加载文本文件", command=self._load_text_file)
        self.text_context_menu.add_command(label="📋 粘贴文本", command=self._paste_text)
        self.text_context_menu.add_separator()
        self.text_context_menu.add_command(label="✏️ 编辑模式", command=self._toggle_text_edit_mode)
        self.text_context_menu.add_command(label="🔒 只读模式", command=self._toggle_text_edit_mode)
        self.text_context_menu.add_separator()
        self.text_context_menu.add_command(label="📝 全选", command=lambda: self.text_output.tag_add(tk.SEL, "1.0", tk.END))
        self.text_context_menu.add_command(label="🗑️ 清空文本", command=self._clear_text)
        self.text_context_menu.add_separator()
        self.text_context_menu.add_command(label="💾 保存文本", command=self._save_text)
        
        # 绑定右键事件
        self.text_output.bind("<Button-3>", self._show_text_context_menu)
        
        # 绑定键盘快捷键
        self.text_output.bind("<Control-a>", lambda e: self.text_output.tag_add(tk.SEL, "1.0", tk.END))
        self.text_output.bind("<Control-v>", lambda e: self._paste_text())
        self.text_output.bind("<Control-o>", lambda e: self._load_text_file())

    def _show_text_context_menu(self, event):
        """显示文本区域的右键菜单"""
        try:
            # 根据当前状态更新菜单项
            if self.text_edit_mode:
                self.text_context_menu.entryconfig("✏️ 编辑模式", state=tk.DISABLED)
                self.text_context_menu.entryconfig("🔒 只读模式", state=tk.NORMAL)
            else:
                self.text_context_menu.entryconfig("✏️ 编辑模式", state=tk.NORMAL)
                self.text_context_menu.entryconfig("🔒 只读模式", state=tk.DISABLED)
            
            # 显示菜单
            self.text_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.text_context_menu.grab_release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoSummarizerApp(root)
    root.mainloop()