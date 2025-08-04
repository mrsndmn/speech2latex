import os
import datasets
import whisper
import torch
import torchaudio
from tqdm.auto import tqdm

if __name__ == "__main__":

    dataset_path = "AAAI2025/MathSpeech"
    dataset = datasets.load_dataset(dataset_path, split="train")

    dataset_transcribtions = []

    model = whisper.load_model("large")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for item in tqdm(dataset):
        audio = torch.tensor(item['audio']['array'], device=device).to(torch.float32)
        sample_rate = item['audio']['sampling_rate']

        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)

        result = model.transcribe(audio, language="en", fp16=False)
        dataset_transcribtions.append(result['text'])

    print(dataset_transcribtions)

    dataset = dataset.add_column("whisper_text", dataset_transcribtions)
    dataset.save_to_disk("MathSpeech_whisper_transcribed")

    dataset.push_to_hub("marsianin500/MathSpeech_whisper_transcribed")
