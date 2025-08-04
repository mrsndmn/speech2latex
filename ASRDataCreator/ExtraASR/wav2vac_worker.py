import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from file_worker import write_json, read_json, create_or_pass_dir
import os
import torchaudio
import numpy as np
from utils_vad import get_speech_timestamps, read_audio


class Wav2Vec2Worker:
    def __init__(self, device_id,):
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english").to(self.device)

        assert torch.cuda.is_available(), "CUDA is not available"


    def get_transcription(self,audio_path):
         # tokenize
        input_values = self.processor(read_audio(audio_path),sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(self.device)

        with torch.no_grad():
            # retrieve logits
            logits = self.model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.lower()
    def get_batched_transcriptions(self, audio_paths, batch_size=8):
        """
        Метод для транскрибирования коротких аудиофайлов батчами.
        audio_paths: список путей к аудиофайлам.
        batch_size: количество аудио для обработки за один проход модели.
        """
        all_transcriptions = []  # Список для хранения транскрипций для всех файлов
        input_values_list = []

        # Загружаем и подготавливаем все аудиофайлы
        for audio_path in audio_paths:
            print(f"Processing file: {audio_path}")
            wav = read_audio(audio_path)

            # Преобразуем аудио в input_values
            input_values = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_values.squeeze(0).to(self.device)
            input_values_list.append(input_values)

        # Разбиваем аудиофайлы на батчи
        for i in range(0, len(input_values_list), batch_size):
            batch = input_values_list[i:i + batch_size]

            # Применяем padding для батча аудиофайлов с помощью метода self.processor.pad
            padded_batch = self.processor.pad({"input_values": batch}, padding=True, return_tensors="pt").input_values.to(self.device)

            # Получаем предсказания для текущего батча
            with torch.no_grad():
                logits = self.model(padded_batch).logits

            # Предсказываем ID для каждого аудио в батче
            predicted_ids = torch.argmax(logits, dim=-1)

            # Декодируем предсказания в текст
            transcriptions = self.processor.batch_decode(predicted_ids)

            # Сохраняем транскрипции для каждого аудио
            for j, transcription in enumerate(transcriptions):
                all_transcriptions.append({
                    "file": audio_paths[i + j],
                    "transcription": transcription.lower()
                })

        return all_transcriptions