from faster_whisper import WhisperModel
from pathlib import Path

from src.att.conf.constants import MODEL_ROOT_DIR, MODEL_NAME


class Transcriptor:
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
        :param language:
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
                                      vad_filter: bool = True):
        print("Start transcription...")
        print("Detecting language...")
        # the core transcribe logic
        segments, info = self.model.transcribe(
            audio_path,
            language=language,  # if none, means auto-detect
            beam_size=beam_size,  # search beam size, 5 is the best balance between speed and accuracy
            vad_filter=vad_filter,  # turn on VAD filter enables multiple language switch in the same audio.
            vad_parameters=dict(
                min_silence_duration_ms=500,  # 500ms silence means a new sentence split 即视为句子结束并切分
                speech_pad_ms=100  # set 100ms cap before and after each speech to avoid word split
            ),
            word_timestamps=False  # if you want timestamp of each word, set it to True
        )

        # info.language indicates the main language in the audio
        print(
            f"\nDetecting language: {info.language.upper()} (language_probability: {info.language_probability:.2%})\n")
        print("-" * 50)

        # print the transcription
        for segment in segments:
            # reformat timestamp
            start_time = self.format_timestamp(segment.start)
            end_time = self.format_timestamp(segment.end)
            print(f"[{start_time} -> {end_time}] {segment.text.strip()}")

    @staticmethod
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
