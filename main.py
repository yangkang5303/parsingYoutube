import yt_dlp
import whisper
import os
# from transformers import pipeline
from openai import OpenAI
try:
    from config import OPENAI_API_KEY
except ImportError:
    print("请创建 config.py 文件并设置 OPENAI_API_KEY")
    print("可以参考 config.example.py 文件的格式")
    exit(1)
from typing import List
from tqdm import tqdm

client = OpenAI(api_key=OPENAI_API_KEY)

def get_video_info(url):
    """获取视频信息和字幕"""
    # 基本配置
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['zh-Hans', 'zh', 'en'],
        'skip_download': True,
        'subtitlesformat': 'vtt',
        'outtmpl': 'subtitles.%(ext)s',
        'cookiesfrombrowser': ('chromium', os.path.expanduser('~/snap/chromium/common/chromium/Default/')),
        'verbose': True,  # 添加详细输出
        'no_warnings': False,  # 显示警告信息
        'extract_flat': False,  # 获取完整信息
        'force_generic_extractor': False,  # 不使用通用提取器
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info
        except Exception as e:
            print(f"获取视频信息失败: {str(e)}")
            return None

def extract_text_from_vtt(vtt_path):
    """从 VTT 文件中提取文本"""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 跳过 VTT 头部
    lines = content.split('\n')
    text_lines = []
    for line in lines:
        # 跳过时间戳、WEBVTT 标记和空行
        if (line.strip() and 
            not line.strip().startswith('WEBVTT') and 
            not '-->' in line and 
            not line.strip().replace('->', '').replace('.', '').replace(':', '').isdigit()):
            text_lines.append(line.strip())
    return ' '.join(text_lines)

def download_video(url, output_dir="downloads"):
    """下载视频"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 基本配置
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'cookiesfrombrowser': ('chromium', os.path.expanduser('~/snap/chromium/common/chromium/Default/')),  # 使用 cookiesfrombrowser
        'verbose': True,  # 添加详细输出
        'no_warnings': False,  # 显示警告信息
        'extract_flat': False,  # 获取完整信息
        'force_generic_extractor': False,  # 不使用通用提取器
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            info = ydl.extract_info(url, download=False)
            video_id = info['id']
            video_ext = info['ext']
            video_path = os.path.join(output_dir, f"{video_id}.{video_ext}")
            return video_path
        except Exception as e:
            print(f"下载视频失败: {str(e)}")
            return None

def transcribe_video(video_path):
    """使用 Whisper 转录视频
    
    Args:
        video_path: 视频文件路径
    Returns:
        str: 转录的文本内容
    """
    # 显示加载模型进度
    print("正在加载 Whisper 模型...")
    with tqdm(total=1, desc="加载模型", ncols=100, colour='green') as pbar:
        model = whisper.load_model("medium") # available models: tiny, base, small, medium, large
        pbar.update(1)
    print("开始转录，使用模型: medium")
    
    try:
        print("正在处理音频...")
        # 转录配置
        transcribe_options = {
            "language": "zh",  # 指定中文语言
            "task": "transcribe",  # 转录任务（不是翻译）
            "temperature": 0.0,  # 降低温度以获得更稳定的结果
            "best_of": 1,  # 减少采样以加快速度
            "beam_size": 5,  # beam search 大小
            "patience": 1.0,  # beam search 耐心值
            "length_penalty": 0,  # 长度惩罚系数
            "compression_ratio_threshold": 2.4,  # gzip压缩比阈值
            "logprob_threshold": -1.0,  # 平均对数概率阈值
            "no_speech_threshold": 0.6,  # 无语音阈值
            "fp16": True,  # 使用 FP16 推理
            "condition_on_previous_text": True,  # 基于前文进行预测
            "initial_prompt": None,  # 初始提示词
            "suppress_tokens": "-1",  # 抑制特殊字符token
            "verbose": True  # 启用详细输出，显示转录进度
        }

        # 执行转录
        result = model.transcribe(video_path, **transcribe_options)
        
        if not result or "text" not in result:
            print("\n转录失败：未能获取文本结果")
            return None
            
        print("\n转录完成")
        return result["text"]
    except Exception as e:
        print(f"\n转录过程出错: {str(e)}")
        return None

def split_text(text: str, max_length: int = 13000) -> List[str]:
    """将长文本分割成小段"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize_with_gpt4(text: str, api_key: str) -> str:
    """使用 GPT-4 总结文本"""

    # 分割长文本
    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        try:
            # 对英文文本
            if any(ord(c) < 128 for c in chunk):
                prompt = ("Condense the provided text into concise bulletpoints,offering 2 critical points and 3 interesting facts respond in Chinese language: \n\n" + chunk)
            # 对中文文本
            else:
                prompt = "请总结以下文本的要点：\n\n" + chunk

            response = client.chat.completions.create(model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "提取文本中的关键信息并生成10条bulletpoints,提供2条批判性思考方向。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=3000)

            summary = response.choices[0].message.content.strip()
            summaries.append(summary)

        except Exception as e:
            print(f"总结时出错: {str(e)}")
            continue

    # 如果有多个片段，再次总结
    if len(summaries) > 1:
        try:
            final_prompt = "请将以下多个片段的总结整合成一个连贯的总结：\n\n" + "\n\n".join(summaries)
            response = client.chat.completions.create(model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个专业的文本总结助手，善于整合多个总结并生成最终总结,提供2条批判性思考方向,提供3条有趣的事实。字数不限"},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.7,
            max_tokens=1000)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"整合总结时出错: {str(e)}")

    return "\n\n".join(summaries)

def print_available_subtitles(info):
    print("\n可用的手动字幕：")
    if 'subtitles' in info:
        for lang, subs in info['subtitles'].items():
            print(f"- {lang}")
    
    print("\n可用的自动生成字幕：")
    if 'automatic_captions' in info:
        for lang, subs in info['automatic_captions'].items():
            print(f"- {lang}")

def save_text_to_file(text: str, filename: str = "output.txt"):
    """保存文本到文件"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n文本已保存到文件: {filename}")
        return True
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        return False

def summarize_with_ollama(text: str) -> str:
    """使用本地 Ollama API 调用 Llama3.2 进行总结"""
    try:
        import requests
        
        # Ollama API 端点
        url = "http://localhost:11434/api/generate"
        
        # 准备提示词
        prompt = """请总结以下文本的要点，并提供：
        1. 10个关键要点
        2. 2个批判性思考方向
        3. 3个有趣的事实

        文本内容：
        """ + text
        
        # 发送请求
        response = requests.post(url, json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"API 请求失败: {response.status_code}")
            
    except Exception as e:
        print(f"使用 Ollama 总结时出错: {str(e)}")
        return None

def process_text(text: str):
    """处理文本的主函数"""
    while True:
        print("\n请选择处理方式：")
        print("1. 使用 GPT-4 总结文本")
        print("2. 使用本地 Ollama (Llama2) 总结文本")
        print("3. 保存文本到文件")
        print("4. 退出")
        
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == "1":
            try:
                summary = summarize_with_gpt4(text, OPENAI_API_KEY)
                if summary:
                    print("\n视频总结：")
                    print(summary)
                else:
                    print("\n生成总结失败")
            except Exception as e:
                print(f"\n总结过程出错: {str(e)}")
                
        elif choice == "2":
            try:
                summary = summarize_with_ollama(text)
                if summary:
                    print("\n视频总结：")
                    print(summary)
                else:
                    print("\n生成总结失败")
            except Exception as e:
                print(f"\n总结过程出错: {str(e)}")
                
        elif choice == "3":
            filename = input("\n请输入保存的文件名 (默认为 output.txt): ").strip()
            if not filename:
                filename = "output.txt"
            save_text_to_file(text, filename)
            
        elif choice == "4":
            break
            
        else:
            print("\n无效的选项，请重新选择")

def main():
    print("\n请选择操作：")
    print("1. 从 YouTube 下载并处理视频")
    print("2. 处理本地视频文件")
    
    choice = input("\n请输入选项 (1-2): ").strip()
    
    if choice == "1":
        # 原有的 YouTube 处理流程
        url = input("请输入 YouTube 视频 URL: ")
        process_youtube_video(url)
    elif choice == "2":
        # 处理本地视频
        video_path = input("请输入视频文件路径: ")
        if os.path.exists(video_path):
            text = transcribe_video(video_path)
            if text is None:
                print("转录失败")
                return
            
            print("\n提取的文本：")
            print(text)
            process_text(text)
        else:
            print("找不到指定的视频文件")
    else:
        print("无效的选项")

def process_youtube_video(url):
    """处理 YouTube 视频的原有流程"""
    # 将原来 main 函数中的 YouTube 处理逻辑移到这里
    info = get_video_info(url)
    if info:
        print_available_subtitles(info)
    else:
        print("未找到视频信息") 
        return
    
    # 检查可用的字幕
    has_subtitles = False
    text = None  # 初始化为 None
    subtitle_lang = None

    # 按优先级检查字幕可用性
    for lang in ['zh-Hans','zh-Hans-zh-CN', 'zh-Hans-zh-CN', 'zh-TW', 'zh-Hant-TW', 'zh', 'en']:
        if 'subtitles' in info and lang in info['subtitles']:
            has_subtitles = True
            subtitle_lang = lang
            print(f"找到{lang}手动添加的字幕，正在处理...")
            break
        elif 'automatic_captions' in info and lang in info['automatic_captions']:
            has_subtitles = True
            subtitle_lang = lang
            print(f"找到{lang}自动生成的字幕，正在处理...")
            break

    if has_subtitles:
        # 下载字幕
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': [subtitle_lang],
            'skip_download': True,
            'subtitlesformat': 'vtt',
            'outtmpl': 'subtitles',
            'cookiesfrombrowser': ('chromium', os.path.expanduser('~/snap/chromium/common/chromium/Default/')),  # 使用 cookiesfrombrowser
            'verbose': True,
            'no_warnings': False,
            'extract_flat': False,
            'force_generic_extractor': False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # 读取字幕文件
        subtitle_file = f'subtitles.{subtitle_lang}.vtt'
        if os.path.exists(subtitle_file):
            text = extract_text_from_vtt(subtitle_file)
            print("\n提取的文本：")
            print(text)
            # 使用新的处理函数
            process_text(text)
            return
        else:
            has_subtitles = False

    if not has_subtitles or not text:
        print("未找到字幕或字幕处理失败，将下载视频并使用 Whisper 转录...")
        video_path = download_video(url)
        if video_path:
            text = transcribe_video(video_path)
            if text is None:
                print("转录失败")
                return
            # 可选：删除视频文件
            # os.remove(video_path)
        else:
            print("处理失败")
            return
    
    # 确保文本不为空
    if not text or text.strip() == "":
        print("未能获取到有效文本")
        return
        
    print("\n提取的文本：")
    print(text)
    
    # 使用新的处理函数
    process_text(text)

if __name__ == "__main__":
    main() 
