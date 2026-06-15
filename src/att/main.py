from src.att.core.transcriptor import Transcriptor

# 使用示例
if __name__ == "__main__":
    # create transcriptor
    transcriptor = Transcriptor()
    audio_dir = "C:/Users/pliu/Documents/tools/llama.cpp/sample_data"
    names = ["aimes.mp3","clinstructions.mp3"]
    for name in names:
        transcriptor.transcribe_multilingual_audio(f"{audio_dir}/{name}")