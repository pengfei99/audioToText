# audioToText

In this repo, we evaluate different `Automatic Speech Recognition(ASR)` models and `inference engine`. 

The context of the project:
- we have multiple audio records(.mp3) of visio-conference. We want to automatically write meeting notes
- we don't have gpu to run the model
- the speed is not our primary goal. the accuracy is.





## 2. Inference engine Evaluation

We have few inference engines:
- faster-whisper
- Hugging Face Transformers
- vllm
- whisper.cpp (llama.cpp for arm)

> The choice of inference engine will be highly influenced by the model choice. For example, Faster-Whisper does not support
Voxtral models. If you choose Voxtal, the inference engines must be `vllm` or `Hugging Face Transformers`

> There are inference engines(e.g. mlx-audio) developed for MacOS. As we will never use MacOS, we don't evaluate them.

### 2.1 faster-whisper

`faster-whisper` is a reimplementation of `OpenAI's Whisper` using `CTranslate2`, which is a `fast inference engine for Transformer models`.

You can find the git repo [here](https://github.com/SYSTRAN/faster-whisper)

The below metric shows a bench on running `whisper-Large-v2` model on GPU which they have published.

| Implementation | Precision | Beam size | Time | VRAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper | fp16 | 5 | 2m23s | 4708MB |
| whisper.cpp (Flash Attention) | fp16 | 5 | 1m05s | 4127MB |
| transformers (SDPA)[^1] | fp16 | 5 | 1m52s | 4960MB |
| faster-whisper | fp16 | 5 | 1m03s | 4525MB |
| faster-whisper (`batch_size=8`) | fp16 | 5 | 17s | 6090MB |
| faster-whisper | int8 | 5 | 59s | 2926MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 16s | 4500MB |

> You can notice it consumes less resources and has better response time


### 2.2 Hugging Face transformers

### 2.3 vllm

### 2.4


## 3. Automatic Speech Recognition model evaluation

There are many ASR models, if you want a full list, you can visit the hugging face [repo](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)

We have only tested:
- voxtral-Mini-3B-2507:
- 

### 3.1  Voxtral models 

Voxtral is a serious next-generation ASR model families. It provides different type models for difference use case:

- VoxtralMini-4B-Realtime-2602 : for streaming
- Voxtral-Mini-3B-2507 : 

> our benchmark uses a quantified model(Q5_K_M) from https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF

###

## 


## Model eval: Voxtral vs Faster-Whisper vs NeMo Canary

By reading this [article](https://weesperneonflow.ai/en/blog/2026-03-31-voxtral-whisper-open-source-speech-models-comparison-2026/),
we want to compare the Voxtral model with the faster-whisper

| Feature             | Voxtral Mini 4B Realtime 2602                                                    | Faster-Whisper Large v3                                                                 |
|---------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Primary Design      | Real-time streaming (designed for <500ms latency)                                | Offline batch processing (designed for maximum accuracy on pre-recorded files).         |
| Speed               | Extremely Fast. Benchmarks show it transcribing 1-minute audio in ~3 seconds 34. | Fast (for its size). Transcribes 1-minute audio in ~8 seconds on similar hardware 34.   |
| Supported Languages | 13 Languages                                                                     | 99 Languages                                                                            |
| Accuracy (EN/FR)    | Superior. Consistently outperforms Whisper on the 13 languages it covers.        | Excellent. Still highly accurate, but slightly behind Voxtral on mainstream languages." |
| Ease of Use (MP3)   | Harder. Requires decoding MP3 to raw PCM chunks and streaming them.              | Easier. Natively accepts .mp3 file paths directly.                                      |



## 4. The prototype

We also developed a tool which can read a `.mp3` file and output the transcript in the console or/and a file.

We use the `faster-whisper`(https://pypi.org/project/faster-whisper/) package and the model `faster-whisper-large-v3`.


Below is an example on how to use it.

```python
from src.att.core.fh_transcriptor import Transcriptor


if __name__ == "__main__":
    # create transcriptor
    transcriptor = Transcriptor()

    # use transcriptor to output text
    audio_dir = "C:/Users/pliu/Documents/tools/llama.cpp/sample_data"
    names = ["aimes.mp3","clinstructions.mp3","ZOOM0001.mp3"]


    for name in names:
        transcriptor.transcribe_multilingual_audio(f"{audio_dir}/{name}", output_file=f"{audio_dir}/{name}_transcript.txt")
```

### 4.1 data cleaning

Poor audio quality often hurts accuracy more than the model choice. The recommendation is to convert your audio files
into `Mono 16kHz wav`. For example, we can use `ffmpeg`

```shell
ffmpeg -i input.mp3 \
       -ac 1 \
       -ar 16000 \
       output.wav
```

```python
import subprocess

subprocess.run([
    "ffmpeg",
    "-i", "input.mp3",
    "-ac", "1",
    "-ar", "16000",
    "output.wav"
])
```

### 4.2  Main workflow

1. Discover MP3 files
2. Convert to WAV
3. Run VAD
4. Transcribe
5. Save TXT
6. Save SRT
7. Log results