import os
import re
import uuid
import subprocess
import glob
from typing import List, Optional, Tuple, Callable

def ensure_downloads_dir() -> str:
    """确保 downloads 目录存在并返回绝对路径"""
    downloads_dir = os.path.join(os.getcwd(), "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    return downloads_dir

def check_command_available(command: str) -> bool:
    """检查命令是否在系统可用"""
    try:
        subprocess.run([command, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

def download_youtube_media(url: str) -> str:
    """
    使用 yt_dlp 下载 YouTube 媒体，优先音频。
    返回下载的文件绝对路径。
    """
    downloads_dir = ensure_downloads_dir()
    try:
        from yt_dlp import YoutubeDL
    except ImportError as e:
        raise RuntimeError("缺少依赖: yt_dlp。请先安装: pip install yt-dlp") from e

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(downloads_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        # 某些情况下会有后处理重命名，这里尝试匹配同 id 的实际文件
        video_id = info.get("id")
        cand = glob.glob(os.path.join(downloads_dir, f"{video_id}.*"))
        if cand:
            # 选择体积最小的（通常是音频）
            filepath = sorted(cand, key=lambda p: os.path.getsize(p))[0]
        if not os.path.isabs(filepath):
            filepath = os.path.abspath(filepath)
        return filepath

def convert_to_wav(input_path: str, target_sample_rate: int = 16000, target_channels: int = 1,
                   progress_callback: Optional[Callable[[str], None]] = None) -> str:
    """
    使用 ffmpeg 转换任意音视频为 wav。
    返回 wav 文件绝对路径。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if not check_command_available("ffmpeg"):
        raise RuntimeError("未检测到 ffmpeg，请先安装 ffmpeg 并配置到 PATH")

    downloads_dir = ensure_downloads_dir()
    base = os.path.splitext(os.path.basename(input_path))[0]
    wav_path = os.path.join(downloads_dir, f"{base}.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", str(target_channels),
        "-ar", str(target_sample_rate),
        wav_path,
    ]

    if progress_callback:
        progress_callback("开始转码为 WAV (ffmpeg)…")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if progress_callback:
            progress_callback("ffmpeg 转码失败")
        raise RuntimeError(f"ffmpeg 转码失败: {result.stderr[:500]}")

    return os.path.abspath(wav_path)

def _resolve_whisper_cpp(model_preference: str) -> Tuple[str, str]:
    """返回 (whisper-cli 路径, 模型文件路径)"""
    project_root = os.getcwd()
    bin_path = os.path.join(project_root, "whisper.cpp", "build", "bin", "whisper-cli")
    if not os.path.exists(bin_path):
        raise RuntimeError(f"未找到 whisper-cli 可执行文件: {bin_path}")

    models_dir = os.path.join(project_root, "whisper.cpp", "models")
    prefer_map = {
        "small": os.path.join(models_dir, "ggml-small.bin"),
        "medium": os.path.join(models_dir, "ggml-medium.bin"),
        "large-v3": os.path.join(models_dir, "ggml-large-v3.bin"),
        "base": os.path.join(models_dir, "ggml-base.bin"),
    }
    # 选择用户偏好
    candidates: List[str] = []
    if model_preference in prefer_map:
        candidates.append(prefer_map[model_preference])
    # 降级候选
    candidates.extend([
        prefer_map["medium"], prefer_map["small"], prefer_map["base"]
    ])

    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break
    if model_path is None:
        raise RuntimeError("未找到可用的 ggml 模型，请运行 whisper.cpp/models/download-ggml-model.sh 下载所需模型")

    return bin_path, model_path

def whisper_cpp_transcribe_to_srt(wav_path: str, model_choice: str = "small",
                                  force_chinese: bool = True,
                                  progress_callback: Optional[Callable[[str], None]] = None,
                                  language_code: Optional[str] = None,
                                  force_retranslate: bool = True,
                                  keep_original: bool = True,
                                  online_provider: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    使用 whisper.cpp 的 whisper-cli 生成 SRT；必要时将结果翻译为简体中文。
    返回最终 SRT 文件路径。
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV 文件不存在: {wav_path}")

    cli_path, model_path = _resolve_whisper_cpp(model_choice)
    downloads_dir = ensure_downloads_dir()
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    out_prefix = os.path.join(downloads_dir, base_name)

    cmd = [
        cli_path,
        "-m", model_path,
        "-f", wav_path,
        "-osrt",
        "-of", out_prefix,
    ]
    if language_code:
        cmd += ["-l", language_code]

    print("执行 whisper.cpp:", " ".join(cmd))
    if progress_callback:
        progress_callback("开始调用 whisper.cpp (生成 SRT)…")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    try:
        detected_lang: Optional[str] = None
        line_count = 0
        while True:
            line = proc.stderr.readline()
            if not line:
                break
            line_count += 1
            line_stripped = line.strip()
            if not line_stripped:
                continue
            print(line_stripped)
            if detected_lang is None and "lang =" in line_stripped:
                # e.g. "lang = ja" / "lang = en"
                try:
                    detected_lang = line_stripped.split("lang =", 1)[1].strip().split(" ")[0]
                except Exception:
                    detected_lang = None
            if progress_callback and (line_count % 10 == 0 or any(k in line_stripped for k in ["processing", "decode", "load", "%"])):
                short = line_stripped
                if len(short) > 120:
                    short = short[:117] + "..."
                progress_callback(f"whisper.cpp: {short}")
    finally:
        proc.wait()

    if proc.returncode != 0:
        err_text = (proc.stderr.read() or "").strip()
        if progress_callback:
            progress_callback("whisper.cpp 运行失败")
        raise RuntimeError(f"whisper.cpp 失败: {err_text[:500]}")

    srt_path = out_prefix + ".srt"
    if not os.path.exists(srt_path):
        # 有时可能落在输入目录，兜底搜索
        base = os.path.splitext(os.path.basename(wav_path))[0]
        cand = sorted(glob.glob(os.path.join(downloads_dir, f"{base}*.srt")))
        if cand:
            srt_path = cand[0]
        else:
            raise FileNotFoundError("未找到 whisper.cpp 生成的 SRT 文件")

    # 输出到终端
    try:
        print("\n--- whisper.cpp 生成的 SRT ---")
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            print(f.read())
        print("--- 结束 ---\n")
    except Exception:
        pass

    # 如果需要强制中文，非中文内容则机器翻译到中文
    orig_srt_path = srt_path

    if force_chinese:
        try:
            if progress_callback:
                progress_callback("检查字幕语言并将 SRT 统一翻译为简体中文…")
            # 根据检测语言选择合适的翻译模型
            lang_for_translate = None
            if language_code:
                lang_for_translate = language_code
            elif 'detected_lang' in locals() and detected_lang:
                lang_for_translate = detected_lang

            if online_provider:
                zh_srt_path = translate_srt_to_chinese_online(
                    orig_srt_path,
                    provider=online_provider,
                    model="gpt-4o-mini",
                    progress_callback=progress_callback,
                )
            else:
                # force_retranslate=True 时，强制对整份 SRT 做中文翻译
                zh_srt_path = translate_srt_to_chinese(
                    orig_srt_path,
                    progress_callback=progress_callback,
                    source_lang=lang_for_translate,
                    force_all=force_retranslate
                )
            srt_path = zh_srt_path
            if not keep_original:
                try:
                    if os.path.exists(orig_srt_path):
                        os.remove(orig_srt_path)
                except Exception:
                    pass
        except Exception as e:
            # 在线失败自动回退到离线
            try:
                if online_provider and progress_callback:
                    progress_callback(f"在线翻译失败，尝试离线翻译: {e}")
                zh_srt_path = translate_srt_to_chinese(
                    orig_srt_path,
                    progress_callback=progress_callback,
                    source_lang=None,
                    force_all=True
                )
                srt_path = zh_srt_path
            except Exception as e2:
                if progress_callback:
                    progress_callback(f"翻译为中文失败: {e2}")

    return srt_path, (orig_srt_path if (keep_original and os.path.exists(orig_srt_path)) else None)

def _contains_cjk(text: str) -> bool:
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def translate_srt_to_chinese(srt_path: str, progress_callback: Optional[Callable[[str], None]] = None,
                             source_lang: Optional[str] = None,
                             force_all: bool = False) -> str:
    """
    将 SRT 文件中的台词翻译为简体中文。保留编号与时间轴，仅替换文本。
    依赖 transformers 与 sentencepiece。
    返回翻译后的 SRT 路径（同目录下添加 .zh-CN 后缀）。
    """
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT 文件不存在: {srt_path}")

    if progress_callback:
        progress_callback("正在加载翻译模型（Helsinki-NLP/opus-mt-en-zh）… 首次使用会自动下载")

    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError("缺少 transformers 依赖，请先安装: pip install transformers sentencepiece") from e

    # 根据源语言选择模型，默认英->中
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    if source_lang:
        sl = source_lang.lower()
        if sl.startswith("ja"):
            model_name = "Helsinki-NLP/opus-mt-ja-zh"
        elif sl.startswith("en"):
            model_name = "Helsinki-NLP/opus-mt-en-zh"
        elif sl.startswith("ko"):
            model_name = "Helsinki-NLP/opus-mt-ko-zh"
        elif sl.startswith("de"):
            model_name = "Helsinki-NLP/opus-mt-de-zh"
        elif sl.startswith("fr"):
            model_name = "Helsinki-NLP/opus-mt-fr-zh"
        elif sl.startswith("th"):
            model_name = "Helsinki-NLP/opus-mt-th-zh"
        # 其他语言可继续扩展

    if progress_callback:
        progress_callback(f"加载翻译模型: {model_name}")
    translator = pipeline("translation", model=model_name, tokenizer=model_name)

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = [b for b in content.split("\n\n") if b.strip()]
    translated_blocks: List[str] = []

    total_blocks = len(blocks)
    for idx, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 2:
            translated_blocks.append(block)
            continue
        number_line = lines[0]
        time_line = lines[1]
        text_lines = lines[2:] if len(lines) > 2 else []

        # 翻译文本行（force_all=True 时强制翻译所有文本行）
        translated_text_lines: List[str] = []
        batch: List[str] = []
        positions: List[int] = []
        for i, t in enumerate(text_lines):
            if not t.strip():
                translated_text_lines.append("")
                continue
            # 强制翻译：无条件加入批量
            if force_all:
                positions.append(i)
                batch.append(t)
                translated_text_lines.append("")
            else:
                # 如果明确知道源语言且不是中文，则无论是否含 CJK 都执行翻译
                if source_lang and not source_lang.lower().startswith("zh"):
                    positions.append(i)
                    batch.append(t)
                    translated_text_lines.append("")
                else:
                    if _contains_cjk(t):
                        translated_text_lines.append(t)
                    else:
                        positions.append(i)
                        batch.append(t)
                        translated_text_lines.append("")

        if batch:
            if progress_callback and idx % 10 == 0:
                progress_callback(f"正在翻译为简体中文… 进度 {idx}/{total_blocks}")
            # 分批避免过长
            start = 0
            while start < len(batch):
                sub = batch[start:start+32]
                results = translator(sub, max_length=512)
                for j, r in enumerate(results):
                    chinese = r["translation_text"].strip()
                    translated_text_lines[positions[start + j]] = chinese
                start += 32

        new_block = "\n".join([number_line, time_line] + translated_text_lines)
        translated_blocks.append(new_block)

    zh_srt_path = os.path.splitext(srt_path)[0] + ".zh-CN.srt"
    with open(zh_srt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(translated_blocks) + "\n")

    if progress_callback:
        progress_callback("简体中文字幕生成完成")

    return zh_srt_path

def translate_srt_to_chinese_online(srt_path: str,
                                    provider: str = "openai",
                                    model: str = "gpt-4o-mini",
                                    progress_callback: Optional[Callable[[str], None]] = None) -> str:
    """
    使用在线服务将整份 SRT 翻译为简体中文，严格保持编号和时间轴不变，仅翻译文本。
    当前实现：OpenAI Chat API（需要环境变量 OPENAI_API_KEY）。
    """
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT 文件不存在: {srt_path}")

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = [b for b in content.split("\n\n") if b.strip()]
    total = len(blocks)

    def chunks(lst: List[str], n: int) -> List[List[str]]:
        return [lst[i:i+n] for i in range(0, len(lst), n)]

    zh_blocks: List[str] = []

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("未检测到 OPENAI_API_KEY，无法使用在线翻译")
        use_new = False
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            use_new = True
        except Exception:
            import openai as openai_legacy
            openai_legacy.api_key = api_key

        sys_prompt = (
            "你是一名专业字幕翻译，要求：\n"
            "- 输入是一段或多段标准 SRT 字幕块；\n"
            "- 你必须保持每个块的编号与时间轴完全不变；\n"
            "- 只翻译每个块中的台词文本为简体中文；\n"
            "- 不要添加任何额外内容；\n"
            "- 输出严格为同样数量的 SRT 块，块与块之间用一个空行分隔。"
        )

        for idx, group in enumerate(chunks(blocks, 40), 1):
            if progress_callback:
                progress_callback(f"在线翻译中（OpenAI）… 进度 {min(idx*40, total)}/{total}")
            chunk_text = "\n\n".join(group)
            user_prompt = (
                "请将下面的 SRT 字幕块仅翻译台词文本为简体中文，保持编号和时间轴不变，并用相同的 SRT 格式返回：\n\n"
                + chunk_text
            )
            try:
                if use_new:
                    rsp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                    )
                    out = rsp.choices[0].message.content.strip()
                else:
                    out = openai_legacy.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                    )["choices"][0]["message"]["content"].strip()
            except Exception as e:
                raise RuntimeError(f"在线翻译失败: {e}")

            parts = [p for p in out.split("\n\n") if p.strip()]
            if len(parts) != len(group):
                parts = [out]
            zh_blocks.extend(parts)
    else:
        raise RuntimeError(f"未知或未实现的在线翻译提供商: {provider}")

    zh_srt_path = os.path.splitext(srt_path)[0] + ".zh-CN.srt"
    with open(zh_srt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(zh_blocks) + "\n")

    if progress_callback:
        progress_callback("在线中文 SRT 生成完成")

    return zh_srt_path

def process_youtube_to_srt(url: str, model: str = "small", language: Optional[str] = None,
                           progress_callback: Optional[Callable[[str], None]] = None,
                           translate_to_zh: bool = True,
                           keep_original: bool = True,
                           online_provider: Optional[str] = None) -> Tuple[str, str, str, Optional[str]]:
    """
    从 YouTube 链接下载 -> 转 WAV -> whisper 转写为 SRT。
    返回 (downloaded_media, wav_path, srt_path)
    """
    if progress_callback:
        progress_callback("开始下载 YouTube 媒体…")
    media_path = download_youtube_media(url)
    if progress_callback:
        progress_callback("下载完成，开始转码为 WAV…")
    wav_path = convert_to_wav(media_path, progress_callback=progress_callback)
    if progress_callback:
        progress_callback("转码完成，开始转写为 SRT…")
    # 使用 whisper.cpp 生成 SRT；根据 translate_to_zh 决定是否翻译为中文
    srt_path, orig_srt_path = whisper_cpp_transcribe_to_srt(
        wav_path,
        model_choice=model,
        force_chinese=translate_to_zh,
        progress_callback=progress_callback,
        language_code=language,
        force_retranslate=True,
        keep_original=keep_original,
        online_provider=(online_provider if translate_to_zh else None)
    )
    return media_path, wav_path, srt_path, orig_srt_path

def process_local_to_srt(local_path: str, model: str = "small", language: Optional[str] = None,
                         progress_callback: Optional[Callable[[str], None]] = None,
                         translate_to_zh: bool = True,
                         keep_original: bool = True,
                         online_provider: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
    """
    本地文件 -> 转 WAV -> whisper 转写为 SRT。
    返回 (wav_path, srt_path)
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"本地文件不存在: {local_path}")
    # 已经是 wav 则直接使用
    if os.path.splitext(local_path)[1].lower() == ".wav":
        wav_path = os.path.abspath(local_path)
    else:
        if progress_callback:
            progress_callback("开始转码为 WAV…")
        wav_path = convert_to_wav(local_path, progress_callback=progress_callback)
    if progress_callback:
        progress_callback("开始转写为 SRT…")
    # 使用 whisper.cpp 生成 SRT；根据 translate_to_zh 决定是否翻译为中文
    srt_path, orig_srt_path = whisper_cpp_transcribe_to_srt(
        wav_path,
        model_choice=model,
        force_chinese=translate_to_zh,
        progress_callback=progress_callback,
        language_code=language,
        force_retranslate=True,
        keep_original=keep_original,
        online_provider=(online_provider if translate_to_zh else None)
    )
    return wav_path, srt_path, orig_srt_path

def generate_srt_from_transcript(transcript_text: str, task_id: str, language: str = "zh") -> str:
    """
    将Whisper转录文本转换为标准SRT格式字幕
    
    Args:
        transcript_text: Whisper转录的文本
        task_id: 任务ID，用于文件命名
        language: 语言代码
    
    Returns:
        SRT字幕文件路径
    """
    try:
        # 在终端中显示生成的CC字幕内容
        print(f"\n=== 生成CC字幕 (任务ID: {task_id}) ===")
        print(f"转录文本长度: {len(transcript_text)} 字符")
        print()
        
        # 智能文本分段：按句号、问号、感叹号分割
        sentences = re.split(r'[。！？.!?]', transcript_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 如果句子太少，按逗号分割
        if len(sentences) < 3:
            sentences = re.split(r'[，,;；]', transcript_text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # 如果还是太少，按固定长度分割
        if len(sentences) < 3:
            sentences = []
            text = transcript_text
            chunk_size = 50  # 每段50个字符
            while len(text) > 0:
                if len(text) <= chunk_size:
                    sentences.append(text)
                    break
                # 寻找合适的分割点
                split_point = chunk_size
                for i in range(chunk_size, max(30, len(text) - 20), -1):
                    if text[i] in '。！？.!?，,;； ':
                        split_point = i + 1
                        break
                sentences.append(text[:split_point])
                text = text[split_point:].strip()
        
        # 过滤空句子并确保每段都有合理长度
        sentences = [s for s in sentences if s.strip() and len(s) > 5]
        
        if not sentences:
            raise ValueError("无法从转录文本中提取有效句子")
        
        # 计算总字符数和估计总时长
        total_chars = sum(len(s) for s in sentences)
        
        # 根据语言调整阅读速度（中文字符每分钟约300个，英文约150个单词）
        has_chinese = any(any('\u4e00' <= char <= '\u9fff' for char in s) for s in sentences)
        chars_per_minute = 300 if has_chinese else 150
        
        # 计算总时长（秒）
        total_duration = max(30, (total_chars * 60) // chars_per_minute)
        
        # 生成SRT格式字幕
        srt_content = []
        current_time = 0
        
        for i, sentence in enumerate(sentences, 1):
            # 根据句子长度分配时间，确保每句至少3秒
            sentence_duration = max(3, (len(sentence) * total_duration) // total_chars)
            
            # 确保不超过总时长
            if current_time + sentence_duration > total_duration:
                sentence_duration = max(3, total_duration - current_time)
            
            start_time = current_time
            end_time = current_time + sentence_duration
            
            # 格式化时间码：时:分:秒,毫秒
            start_str = f"{start_time//3600:02d}:{(start_time%3600)//60:02d}:{start_time%60:02d},000"
            end_str = f"{end_time//3600:02d}:{(end_time%3600)//60:02d}:{end_time%60:02d},000"
            
            # 构建SRT条目：编号 + 时间码 + 文本 + 空行
            srt_entry = f"{i}\n{start_str} --> {end_str}\n{sentence}\n"
            srt_content.append(srt_entry)
            
            current_time = end_time
            
            # 如果已经达到总时长，停止
            if current_time >= total_duration:
                break
        
        # 生成SRT文件
        srt_file_path = f"downloads/{task_id}_cc.srt"
        
        # 确保downloads目录存在
        os.makedirs("downloads", exist_ok=True)
        
        with open(srt_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))
        
        # 在终端中显示SRT内容
        print("--- SRT格式CC字幕 ---")
        print("\n".join(srt_content))
        print()
        
        print("=== CC字幕生成完成 ===")
        print(f"生成的文件: SRT: {srt_file_path}")
        print()
        
        return srt_file_path
        
    except Exception as e:
        print(f"生成CC字幕失败: {e}")
        raise e

def demo_cc_generation():
    """演示CC字幕生成功能"""
    print("=== CC字幕生成功能演示 ===")
    print()
    
    # 测试文本
    test_text = """卖去机盒老观众们肯定知道这是我们极客湾的经典节目之一了每一代的硬件更新我们都会打造一台ITX的核弹在小提机的机箱里塞下当前游戏性能最强的硬件不计成本地去打造出一台性能密度超高的ITX主机上一代的卖去机盒4.0是我们在2023年做的因为当时旗舰ITX 4090的Founder Sedation加上13900K的CPU 5090都出了这4090还能用吗感觉好卡呀这13900K就更不用说了我都缩刚完售后换了一颗了感觉这样下去电脑根本就用不成太难受了那么是时候全面升级一下了这就是新一代的卖去机盒5.0"""
    
    print("原始文本:")
    print(test_text)
    print()
    print(f"文本长度: {len(test_text)} 字符")
    print()
    
    try:
        # 生成CC字幕
        task_id = str(uuid.uuid4())
        print(f"生成任务ID: {task_id}")
        print()
        
        srt_file_path = generate_srt_from_transcript(test_text, task_id, "zh")
        
        print("=== 生成结果 ===")
        if os.path.exists(srt_file_path):
            print(f"✅ SRT文件生成成功: {srt_file_path}")
            print("文件内容:")
            with open(srt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
        else:
            print(f"❌ SRT文件不存在: {srt_file_path}")
        
        print("\n✅ 演示完成！")
        print("\n使用方法:")
        print("1. 调用 generate_srt_from_transcript(文本, 任务ID, 语言)")
        print("2. 函数会自动生成SRT格式字幕文件")
        print("3. 在终端中显示生成的字幕内容")
        print("4. 文件保存在 downloads/ 目录下")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_cc_generation()
