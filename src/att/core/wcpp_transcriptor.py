"""
A simple and efficient Python module for transcribing audio using whisper.cpp
with the ggml-large-v3-turbo model.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from pywhispercpp.model import Model

from att.conf.constants import MODEL_ROOT_DIR


class WCPPTranscriptor:
    def __init__(
            self,
            model_path: str = "ggml-large-v3-turbo.bin",
            n_threads: int = 8,
            language: str = "auto",
            print_realtime: bool = False,
            print_progress: bool = True,
            **kwargs
    ):
        """
        Initialize the transcriber.
        :param model_path: Path to your .bin model file
        :param n_threads: Number of CPU threads (adjust to your CPU cores)
        :param language: 'auto' or specific language code (en, fr, es, etc.)
        :param print_realtime: Show transcription in real-time
        :param print_progress: Show progress bar
        :param kwargs: Any other whisper.cpp parameters
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = Model(
            model_path,
            n_threads=n_threads,
            language=language,
            print_realtime=print_realtime,
            print_progress=print_progress,
            **kwargs
        )

    def transcribe(self, audio_path: str, output_dir: Optional[str] = None, **transcribe_kwargs) -> List[Dict]:
        """
        Transcribe an MP3 (or any audio file) and optionally save outputs.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Transcribe
        segments = self.model.transcribe(str(audio_path), **transcribe_kwargs)

        # Save outputs if output_dir is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = audio_path.stem

            # Save plain text
            with open(output_dir / f"{base_name}.txt", "w", encoding="utf-8") as f:
                for segment in segments:
                    f.write(f"{segment.text.strip()}\n")

            # Save SRT subtitles
            with open(output_dir / f"{base_name}.srt", "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{self._format_timestamp(segment.t0)} --> {self._format_timestamp(segment.t1)}\n")
                    f.write(f"{segment.text.strip()}\n\n")

        return segments

    def _format_timestamp(self, ms: int) -> str:
        """Convert milliseconds to SRT timestamp format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        millis = ms % 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    def print_segments(self, segments: List[Dict]):
        """Pretty print the transcription segments"""
        for segment in segments:
            print(f"[{segment.t0 / 1000:.2f} → {segment.t1 / 1000:.2f}] {segment.text}")
