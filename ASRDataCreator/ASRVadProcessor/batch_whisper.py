import os
import glob
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def resample_audio(file_path, target_sr=16000):
    audio, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    return audio.squeeze().numpy()

def get_audio_files(root_dir):
    audios_paths = []
    for dirpath, dirnames, filenames in  os.walk(root_dir):
        for filename in filenames:
            audios_paths.append((dirpath,filename))
    return audios_paths



def extract_extension(file_name):
    return os.path.splitext(file_name)[0]

import numpy as np

def pad_audio_to_length(audio, target_length=3000):
    if len(audio) >= target_length:
        return audio
    else:
        return np.pad(audio, (0, target_length - len(audio)), 'constant')


def main(root_dir):
    # Загрузка модели и процессора
    model_name = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")

    # Получаем список аудио файлов
    audio_paths = get_audio_files(root_dir)
    audio_data = []
    metadata = []

    # Ресемплирование и загрузка аудио
    print("Загрузка и ресемплирование аудио файлов...")
    for audio_path in tqdm(audio_paths):
        audio = resample_audio(os.path.join(*audio_path), target_sr=16000)
        audio_data.append(audio)
        file_dir, file_name = audio_path
        file_id = extract_extension(file_name)
        metadata.append({
            'audio_path': os.path.join(file_dir,file_name),
            'WAVID': file_id
        })

    # Сортируем аудио по длине
    print("Сортировка аудио по длине...")
    audio_lengths = [(i, len(audio)) for i, audio in enumerate(audio_data)]
    sorted_audio_indices = sorted(range(len(audio_lengths)), key=lambda k: audio_lengths[k][1])

    # Параметры батча
    batch_size = 8  # Подберите оптимальное значение в зависимости от вашей GPU

    # Обработка батчей
    print("Обработка аудио по батчам...")
    results = []
    for i in tqdm(range(0, len(sorted_audio_indices), batch_size)):
        batch_indices = sorted_audio_indices[i:i + batch_size]
        batch_audio = [pad_audio_to_length(audio_data[idx]) for idx in batch_indices]
        batch_metadata = [metadata[idx] for idx in batch_indices]

        # Подготовка входных данных
        inputs = processor(batch_audio, return_tensors="pt", padding=True, max_length=3000, sampling_rate=16000)
        input_features = inputs.input_features.to("cuda:2")

        # Получение предсказаний
        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        # Декодирование предсказаний
        transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # Сохранение результатов
        for meta, transcription in zip(batch_metadata, transcriptions):
            meta['asr_text'] = transcription
            results.append(meta)
            

    # Сохранение результатов в CSV
    df = pd.DataFrame(results)
    output_file = 'transcriptions.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Транскрипции сохранены в файл {output_file}")

if __name__ == "__main__":
    root_dir =  "/home/jovyan/Nikita/SberSaluth/test_synthesized_audios"
    main(root_dir)
