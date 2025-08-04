import torch
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torchaudio
from file_worker import write_json, read_json, create_or_pass_dir
from utils_vad import read_audio

import re

class QwenAudioWorker:
    def __init__(self, device_id):
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-Audio-Chat",
            trust_remote_code=True
        ).to(self.device).eval()

        self.model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen-Audio-Chat",
            trust_remote_code=True
        )

        self.model.generation_config.min_new_tokens = 1

        assert torch.cuda.is_available(), "CUDA is not available"

    def extract_text_in_quotes(self,text):

        matches = re.findall(r'"(.*?)"', text)

        return ' '.join(matches)


    def get_transcription(self, audio_path):

        waveform = read_audio(audio_path)


        query = self.tokenizer.from_list_format([
            {'audio': audio_path},
            {'text': 'Recognize'},
        ])


        response, history = self.model.chat(
            self.tokenizer,
            query=query,
            history=None,
            system='Please listen to the provided audio file carefully and convert the spoken words into accurate, well-formatted English text. Write only the words of the speaker without additional instructions from yourself.'
        )

        return self.extract_text_in_quotes(response)
