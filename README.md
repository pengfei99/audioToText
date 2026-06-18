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

### 2.4 whisper.cpp 


## 3. Automatic Speech Recognition model evaluation

There are many ASR models, if you want a full list, you can visit the hugging face [repo](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)


### 3.1  Voxtral models 

Voxtral is a serious next-generation ASR model families. It provides different type models for difference use case:

| Model Variant           | Size | Best For          | CPU Suitability    | Notes                  |
|-------------------------|------|-------------------|--------------------|------------------------|
| Voxtral Realtime 4B     | ~4B  | Real-time + files | Excellent (pure C) | Recommended            |
| Voxtral Mini (original) | ~3B  | General use       | Good               | Less optimized tooling |
| Voxtral Small / 24B     | 24B  | High accuracy     | Poor               | Too big for most CPUs  |


The model name which you can download from hugging face
- VoxtralMini-4B-Realtime-2602 : A pure C implementation exsits [here](https://github.com/antirez/voxtral.c)
- Voxtral-Mini-3B-2507 : The model is 17.4GB in safe tensor format.

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

### 4 audio data cleaning

### 4.1 ffmpeg introduction

`ffmpeg` is a free, open-source, command-line tool used to record, convert, stream, and edit audio and video files. 
In fact, it is the underlying engine that powers almost all media software in the world, including VLC Media Player, 
YouTube, Spotify, Zoom, and professional video editors.

> In my project, all the python lib that I used(e.g. faster-whisper, whisper-cpp, pydub) to process audio files(e.g. mp3, wav, etc)
> it's actually ffmpeg that reads the raw file and send the `raw sound waves` to the python library.
> 
Key Characteristics of FFmpeg:
- `Command-Line Only`: Unlike normal software, FFmpeg does not have a graphical user interface (GUI).
- `incredibly fast`: It is written in C and highly optimized. It can convert a 2-hour movie file in seconds.
- `supports almost every format`: It can handle virtually every audio and video format ever created (MP3, WAV, MP4, MKV, FLAC, AAC, etc.).

Installation via package manager in windows

```powershell
# run this command in powershell, tested on win 10/11
winget install ffmpeg

# if you have choco
choco install ffmpeg
```

Installation via package manager in debian

```shell
sudo apt update
sudo apt install ffmpeg

# check version after install
ffmpeg -version
```

> You can also download the pre-built from [here](https://www.gyan.dev/ffmpeg/builds/?spm=a2ty_o01.29997173.0.0.572e55fb3irbCV)

### 4.2 Improve the input audio file `quality`

First, a very important clarification: `Converting audio to Mono 16kHz WAV actually decreases the audio quality for human listening`, 
but it drastically `improves the accuracy, stability, and speed for the AI model`.

The recommendation is to convert your audio files into `Mono 16kHz wav`. Because:

- human speech almost entirely lives between 85 Hz and 8,000 Hz (8kHz). ASR training data all in 16kHz
- audio files has two channel(i.e. left and right). ASR model only need one
- uncompressed WAV file is just a list raw PCM(Pulse Code Modulation) with metadata(e.g. 1 channel, sample rate is 16KHz)

> PCM(Pulse Code Modulation): is the digital representation of audio wave, which all ASR model reads as input

We can use `ffmpeg` to convert any audio format to `Mono 16kHz wav`. The below example shows how to convert mp3.

```shell
ffmpeg -i your_messy_audio.mp3 -ar 16000 -ac 1 -c:a pcm_s16le perfect_ai_audio.wav
```
- `-i your_messy_audio.mp3`: specify The input file.
- `-ar 16000`: sets the sample rate to exactly 16kHz.
- `-ac 1`: Forces the audio into Mono (1 channel).
- `-c:a pcm_s16le`: encodes the output as a standard 16-bit uncompressed WAV file.
- `perfect_ai_audio.wav`: The output file name.

Open a cmd, and run the below command, all mp3 file will be converted to wav and the output will be stored in `./wav_files`
```cmd
for %i in (*.mp3) do ffmpeg -i "%i" -ar 16000 -ac 1 -c:a pcm_s16le "wav_files\%~ni.wav"
```



## 5. The prototype

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



### 4.2  Main workflow

1. Discover MP3 files
2. Convert to WAV
3. Run VAD
4. Transcribe
5. Save TXT
6. Save SRT
7. Log results