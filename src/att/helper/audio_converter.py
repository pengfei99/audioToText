"""Module for batch converting MP3 files to WAV format using FFmpeg.

This module replicates the functionality of the windows cmd batch command:
for %i in (*.mp3) do ffmpeg -i "%i" -ar 16000 -ac 1 -c:a pcm_s16le "wav_files\\%~ni.wav"
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# A tuple of supported input extensions (must be lowercase)
SUPPORTED_FORMATS = (".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".opus")

def convert_mp3_to_wav(
    source_dir: str | Path = ".", output_dir: str | Path = "wav_files"
) -> None:
    """Converts all .mp3 files in source_dir to 16kHz, mono, 16-bit PCM .wav files.

    Args:
        source_dir: Directory to scan for MP3 files. Defaults to current directory.
        output_dir: Directory where WAV files will be saved.
    """
    src_path = Path(source_dir)
    out_path = Path(output_dir)

    # Best Practice 1: Ensure the output directory exists before processing
    out_path.mkdir(parents=True, exist_ok=True)

    # Gather all files, then filter by extensions in our supported list
    audio_files = [
        f for f in src_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not audio_files:
        logger.warning(f"No supported audio files found in '{src_path.resolve()}'.")
        return

    logger.info(f"Found {len(audio_files)} file(s) to convert.")

    for file_path in audio_files:
        wav_file = out_path / f"{file_path.stem}.wav"
        # -y option will overwrite the output file if exists
        cmd = [
            "ffmpeg", "-y",
            "-i", str(file_path),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(wav_file),
        ]

        logger.info(f"Converting ({file_path.suffix.upper()}): {file_path.name} -> {wav_file.name}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            logger.critical("FFmpeg is not installed or not found in your system PATH.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert {file_path.name}. Error: {e.stderr.strip()}")


if __name__ == "__main__":
    # Allows the script to be run directly from the terminal
    convert_mp3_to_wav()