import torch
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torchaudio
from file_worker import write_json, read_json, create_or_pass_dir
from utils_vad import read_audio

import re

class QwenAudioWorker:
    def __init__(self, device_id):
        """
        Инициализация QwenAudioWorker.
        device_id: ID устройства (например, 0 для CUDA).
        """
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        # Загрузка токенизатора и модели
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-Audio-Chat", 
          #  device_map="auto",  # Использование автоматического распределения модели по устройствам
            trust_remote_code=True
        ).to(self.device).eval()  # Перевод модели в режим оценки (eval)

        # Настройка параметров генерации
        self.model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen-Audio-Chat", 
            trust_remote_code=True
        )

        # Добавление параметра min_new_tokens
        self.model.generation_config.min_new_tokens = 1

        assert torch.cuda.is_available(), "CUDA is not available"

    def extract_text_in_quotes(self,text):
        # Регулярное выражение для поиска текста между кавычками
        matches = re.findall(r'"(.*?)"', text)
        # Объединяем найденные строки в одну без запятых
        return ' '.join(matches)


    def get_transcription(self, audio_path):
        """
        Метод для получения транскрипции аудиофайла.
        audio_path: Путь к аудиофайлу.
        """
        # Чтение аудио с помощью кастомной функции read_audio
        waveform = read_audio(audio_path)
        
        # Подготовка запроса для модели
        query = self.tokenizer.from_list_format([
            {'audio': audio_path},
            {'text': 'Recognize'},
        ])

        # Генерация ответа от модели
        response, history = self.model.chat(
            self.tokenizer, 
            query=query, 
            history=None, 
            system='Please listen to the provided audio file carefully and convert the spoken words into accurate, well-formatted English text. Write only the words of the speaker without additional instructions from yourself.'
        )
        
        return self.extract_text_in_quotes(response)
