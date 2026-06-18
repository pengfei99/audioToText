from att.core.fh_transcriptor import WhisperTranscriptor
from att.core.vox_transcriptor import VoxTranscriptor


def run_whisper_mp3_example():
    # create transcriptor
    transcriptor = WhisperTranscriptor()

    # use transcriptor to process mp3 output text
    audio_dir = "C:/Users/pliu/Documents/tools/llama.cpp/sample_data"
    names = ["aimes.mp3", "clinstructions.mp3", "ZOOM0001.mp3"]

    for name in names:
        transcriptor.transcribe_multilingual_audio(f"{audio_dir}/{name}",
                                                   output_file=f"{audio_dir}/{name}_transcript.txt")

def run_whisper_wav_example():
    # create transcriptor
    transcriptor = WhisperTranscriptor()

    # use transcriptor to process mp3 output text
    audio_dir = "C:/Users/pliu/Documents/tools/llama.cpp/sample_data/wav_files"
    names = ["aimes.wav", "clinstructions.wav", "ZOOM0001.wav"]

    for name in names:
        transcriptor.transcribe_multilingual_audio(f"{audio_dir}/{name}",
                                                   output_file=f"{audio_dir}/{name}_transcript.txt")


def run_voxtral_example():
    transcriptor = VoxTranscriptor()


def main():
    # run_whisper_mp3_example()
    run_whisper_wav_example()


if __name__ == "__main__":
    main()
