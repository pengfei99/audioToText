from faster_whisper import WhisperModel


def transcribe_multilingual_audio(audio_path:str):
    print("正在加载模型 (首次运行会自动下载 large-v3 模型，约 3GB，请保持网络畅通)...")

    # 1. 模型选择：强烈建议使用 "large-v3"
    # 如果有 NVIDIA 显卡，请将 device 改为 "cuda", compute_type 改为 "float16" 速度会快 10 倍以上
    # 如果只有 CPU，使用 "cpu" 和 "int8" (或 "int8_float16") 可以节省内存并提速
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print("开始转写，正在自动检测法语/英语切换...")

    # 2. 核心配置
    segments, info = model.transcribe(
        audio_path,
        language=None,  # None 表示启用自动语言检测
        beam_size=5,  # 束搜索大小，5 是准确率和速度的最佳平衡
        vad_filter=True,  # 🔥 开启 VAD 过滤，这是多语言无缝切换的关键！
        vad_parameters=dict(
            min_silence_duration_ms=500,  # 静音超过 500ms 即视为句子结束并切分
            speech_pad_ms=100  # 在语音前后各保留 100ms，防止切断单词
        ),
        word_timestamps=False  # 如果需要精确到每个词的时间轴，可设为 True
    )

    # info.language 是整个音频中占比最大的“主要语言”
    print(f"\n音频整体检测到的主要语言: {info.language.upper()} (置信度: {info.language_probability:.2%})\n")
    print("-" * 50)

    # 3. 遍历并打印结果
    for segment in segments:
        # 格式化时间 [00:12.34 -> 00:15.67]
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        print(f"[{start_time} -> {end_time}] {segment.text.strip()}")


def format_timestamp(seconds: float):
    """辅助函数：将秒数转换为 HH:MM:SS.mmm 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# 使用示例
if __name__ == "__main__":
    audio_dir = "C:/Users/pliu/Documents/tools/llama.cpp/sample_data"
    names = ["aimes.mp3","clinstructions.mp3"]
    for name in names:
        transcribe_multilingual_audio(f"{audio_dir}/{name}")