from pathlib import Path
from typing import Optional

import torch
import librosa
from transformers import VoxtralProcessor, VoxtralForConditionalGeneration

from src.att.conf.constants import MODEL_ROOT_DIR, MODEL_NAME


class VoxTranscriptor:
    def __init__(self, model_path: Optional[str] = None, device_type: str = "cpu", compute_type: str = "float16"):
        """
        Initializes the Voxtral Transcriptor.
        Note: Causal audio models perform best on GPU with float16 or bfloat16.
        """
        # Adapt compute type for Hugging Face (e.g., int8 mapping or native torch types)
        self.torch_dtype = torch.float16 if compute_type == "float16" else torch.float32
        if device_type == "cuda" and compute_type == "bfloat16":
            self.torch_dtype = torch.bfloat16

        self.device = torch.device(device_type)
        self.processor, self.model = self._init_model(model_path)

    @staticmethod
    def _is_model_path_valid(local_model_path: Path) -> bool:
        if not local_model_path.is_dir():
            print(f"The provided model path does not exist: {local_model_path.as_posix()}")
            return False
        return True

    def _init_model(self, model_path: Optional[str]):
        """
        Initiates and returns the VoxtralProcessor and VoxtralForConditionalGeneration objects.
        """
        if model_path:
            local_model_path = Path(model_path)
        else:
            local_model_path = MODEL_ROOT_DIR / f"{MODEL_NAME}"

        # Fallback if specific local path is broken
        if not self._is_model_path_valid(local_model_path):
            # You can also use a HF repo string direct fallback like "mistralai/Voxtral-Mini-4B-Realtime-2602"
            local_model_path = MODEL_ROOT_DIR / f"{MODEL_NAME}"

        print(f"Loading Voxtral from: {local_model_path.as_posix()}")

        processor = VoxtralProcessor.from_pretrained(local_model_path.as_posix())
        model = VoxtralForConditionalGeneration.from_pretrained(
            local_model_path.as_posix(),
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device.type == "cuda" else None
        ).to(self.device)

        return processor, model

    def transcribe_multilingual_audio(self, audio_path: str, language: str = "fr",
                                      output_file: Optional[str] = None):
        """
        Transcribes the target audio using Voxtral's causal attention pipeline.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file() or audio_path.suffix != ".mp3":
            print(f"The provided audio path does not exist or is not an MP3: {audio_path.as_posix()}")
            return

        print("Start transcription...")

        # Load audio; Voxtral's encoder expects a native 16kHz sample rate mono signal
        audio_array, sampling_rate = librosa.load(audio_path.as_posix(), sr=16000)

        # Prepare inputs using the multimodal VoxtralProcessor
        # Prompting the language directly into the text component to match your requested language parameter
        prompt_text = f"<|start_of_transcription|><|{language}|><|transcribe|>"
        inputs = self.processor(text=prompt_text, audios=audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.torch_dtype if torch.is_floating_point(v) else None) for k, v in
                  inputs.items()}

        print("Processing audio representations through Causal Encoder & Language Decoder...")

        # Execute Generation (Temperature 0.0 is strictly recommended by Mistral for Voxtral)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,  # Temperature = 0.0 equivalent
                use_cache=True
            )

        # Extract generated tokens (stripping out original input prompts)
        num_input_tokens = inputs["input_ids"].shape[1]
        transcription_tokens = generated_ids[0][num_input_tokens:]

        # Decode logits to text output
        transcription_text = self.processor.decode(transcription_tokens, skip_special_tokens=True).strip()

        print("-" * 50)
        print(f"Target Language Processing Context: {language.upper()}")
        print("-" * 50)
        print(transcription_text)

        # Handle file stream safely if requested
        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Transcription of: {audio_path}\n")
                    f.write(f"Forced Target Language: {language.upper()}\n")
                    f.write("-" * 60 + "\n\n")
                    f.write(transcription_text + "\n")
                print(f"\nTranscription successfully saved to: {output_file}")
            except Exception as e:
                print(f"Could not write to output file: {e}")