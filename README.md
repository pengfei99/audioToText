# audioToText

In this repo, we developed a tool which can read a .mp3 file and output the transcript in the console or/and a file.

We use the `faster-whisper`(https://pypi.org/project/faster-whisper/) package and the model `faster-whisper-large-v3`.


Below is an example on how to use it.

```python
from src.att.core.transcriptor import Transcriptor


if __name__ == "__main__":
    # create transcriptor
    transcriptor = Transcriptor()

    # use transcriptor to output text
    audio_dir = "C:/Users/pliu/Documents/tools/llama.cpp/sample_data"
    names = ["aimes.mp3","clinstructions.mp3","ZOOM0001.mp3"]


    for name in names:
        transcriptor.transcribe_multilingual_audio(f"{audio_dir}/{name}", output_file=f"{audio_dir}/{name}_transcript.txt")
```
