from typing import Optional

from faster_whisper import WhisperModel
from pathlib import Path


from src.att.conf.constants import MODEL_ROOT_DIR, MODEL_NAME


class WhisperTranscriptor:
    def __init__(self, model_path: str = None, device_type: str = "cpu", compute_type: str = "int8"):
        self.model = self._init_model(model_path, device_type, compute_type)

    @staticmethod
    def _is_model_path_valid(local_model_path: Path) -> bool:
        if not local_model_path.is_dir():
            err_msg = f"the provided model path does not exist {local_model_path.as_posix()}"
            print(err_msg)
            return False
        else:
            return True

    def _init_model(self, model_path: str, device_type: str, compute_type: str) -> WhisperModel:
        """
        This function initiates and returns a WhisperModel object.
        :param model_path:
        :param device_type:
        :param compute_type:
        :return:
        """
        # 1. we highly recommend "large-v3" as model
        # 2. If you have an NVIDIA CPU，change device_type to "cuda", compute_type to "float16". This will improve speed
        if model_path:
            local_model_path = Path(model_path)
        else:
            local_model_path = MODEL_ROOT_DIR / f"{MODEL_NAME}"
        # if the provided path is not valid, use the default one
        if not self._is_model_path_valid(local_model_path):
            local_model_path = MODEL_ROOT_DIR / f"{MODEL_NAME}"
        print(local_model_path)
        model = WhisperModel(
            model_size_or_path=local_model_path.as_posix(),
            device=device_type,
            compute_type=compute_type,
        )
        return model

    def transcribe_multilingual_audio(self, audio_path: str, language: str = "fr", beam_size: int = 5,
                                      vad_filter: bool = True, output_file: Optional[str] = None):
        audio_path = Path(audio_path)
        if not audio_path.is_file() or audio_path.suffix != ".mp3":
            err_msg = f"the provided audio path does not exist {audio_path.as_posix()}"
            print(err_msg)
            return

        print("Start transcription...")
        print("Detecting language...")
        # the core transcribe logic
        segments, info = self.model.transcribe(
            audio_path.as_posix(),
            language=language,  # if none, means auto-detect
            beam_size=beam_size,  # search beam size, 5 is the best balance between speed and accuracy, the bigger the number, it will explore more translation paths, it much slower, but finds the most accurate sentence structure.
            vad_filter=vad_filter,  # turn on VAD filter enables multiple language switch in the same audio.
            condition_on_previous_text=True, # Helps the model use the context of the previous French sentence to correctly transcribe the next one.
            initial_prompt="Bonjour, ceci est une transcription en français avec des passages en anglais.",
            # ^ This single line forces the model to expect French vocabulary and punctuation.
            no_speech_threshold=0.6,  # Prevents the model from hallucinating text during silent pauses
            vad_parameters=dict(
                min_silence_duration_ms=500,  # 500ms silence means a new sentence split 即视为句子结束并切分
                speech_pad_ms=100  # set 100ms cap before and after each speech to avoid word split
            ),
            word_timestamps=False  # if you want timestamp of each word, set it to True
        )

        # info.language indicates the main language in the audio
        lang_info = f"\nDetecting language: {info.language.upper()} (language_probability: {info.language_probability:.2%})\n"
        print(lang_info)
        print("-" * 50)

        # Prepare output file (if provided)
        file_handle = None
        if output_file:
            try:
                file_handle = open(output_file, "w", encoding="utf-8")
                # Write header
                file_handle.write(f"Transcription of: {audio_path}\n")
                file_handle.write(f"Detected language: {info.language.upper()}\n")
                file_handle.write("-" * 60 + "\n\n")
            except Exception as e:
                print(f"Could not open output file: {e}")
                file_handle = None

        # print the transcription
        for segment in segments:
            # reformat timestamp
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            output_text = f"[{start_time} -> {end_time}] {segment.text.strip()}"
            print(f"{output_text}")
            # Write to file (very memory efficient - writes line by line)
            if file_handle:
                file_handle.write(output_text + "\n")

        # Close file properly
        if file_handle:
            file_handle.close()
            print(f"\n Transcription saved to: {output_file}")



def format_timestamp(seconds: float):
    """
    This function convert seconds into timestamp in HH:MM:SS.mmm format.
    :param seconds:
    :return:
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"



