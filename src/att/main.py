from pathlib import Path

from att.conf.constants import MODEL_ROOT_DIR, DATA_DIR
from att.core.fh_transcriptor import WhisperTranscriptor
from att.core.vox_transcriptor import VoxTranscriptor
from att.core.wcpp_transcriptor import WCPPTranscriptor
from att.helper.log_manager import setup_logger


def run_whisper_mp3_example():
    # create transcriptor
    transcriptor = WhisperTranscriptor()

    # use transcriptor to process mp3 output text
    audio_dir = DATA_DIR.as_posix()
    names = ["aimes.mp3", "clinstructions.mp3", "ZOOM0001.mp3"]

    for name in names:
        transcriptor.transcribe_multilingual_audio(f"{audio_dir}/{name}",
                                                   output_file=f"{audio_dir}/{name}_transcript.txt")


def run_whisper_wav_example():
    # create transcriptor
    transcriptor = WhisperTranscriptor()

    # use transcriptor to process mp3 output text
    audio_dir = (DATA_DIR / "wav_files")
    names = ["aimes.wav", "clinstructions.wav", "ZOOM0001.wav"]

    for name in names:
        audio_file = audio_dir / name
        transcript_file = audio_dir / f"{name}_transcript.txt"
        transcriptor.transcribe_multilingual_audio(audio_file.as_posix(),
                                                   output_file=transcript_file.as_posix())


def run_voxtral_example():
    transcriptor = VoxTranscriptor()


def run_whisper_cpp_example():
    model_path = MODEL_ROOT_DIR / "ggml-large-v3-turbo.bin"

    transcriber = WCPPTranscriptor(
        model_path=model_path.as_posix(),
        n_threads=8,  # Adjust based on your CPU
        language="auto"
    )
    audio_dir = (DATA_DIR / "wav_files")
    for name in ["aimes.wav", "clinstructions.wav"]:
        audio_file = audio_dir / name
        result = transcriber.transcribe(
            audio_path=audio_file.as_posix(),
            output_dir="transcriptions",  # optional
            # You can add more options here like:
            # translate=True, duration=..., etc.
        )
        transcriber.print_segments(result)


def main():
    log_dir = Path(__file__).parent.parent.parent / "logs"
    print(log_dir.as_posix())
    setup_logger(log_dir=log_dir)
    # run_whisper_mp3_example()
    run_whisper_wav_example()
    # run_whisper_cpp_example()


if __name__ == "__main__":
    main()
