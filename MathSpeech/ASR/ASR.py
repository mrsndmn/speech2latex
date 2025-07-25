import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import os
import pandas as pd
import json
import requests
import whisper
from jiwer import wer, cer
from tqdm.auto import tqdm
import argparse

import os
import whisper
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Dataset & collate utilities for parallel audio loading and preprocessing
# -----------------------------------------------------------------------------


class WhisperAudioDataset(Dataset):
    """Loads and preprocesses each audio file for Whisper."""

    def __init__(self, files: List[str]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)  # 30-s fixed length
        mel = whisper.log_mel_spectrogram(audio)
        return mel, path


def _collate_whisper(batch):
    """Stack mels into a single tensor and return paths list."""
    mels, paths = zip(*batch)  # type: ignore
    mel_batch = torch.stack(mels)
    return mel_batch, list(paths)


def transcribe_batch(dataloader: List[str], model_size="base", language=None):
    """
    Transcribes a list of audio files using Whisper in batches.

    Args:
        dataloader (DataLoader): DataLoader object containing audio files.
        model_size (str): Size of Whisper model to use (tiny, base, small, medium, large).
        language (str): Optional language hint (e.g., 'en').

    Returns:
        List[dict]: List of transcription results per audio file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.compile(whisper.load_model(model_size, device=device))
    results = []


    options = whisper.DecodingOptions(language=language, fp16=torch.cuda.is_available())

    for mel_batch, idxs in tqdm(dataloader, desc=f"Processing {model_size} model", leave=False):
        mel_batch = mel_batch.to(device)
        decoded = whisper.decode(model, mel_batch, options)

        for idx, dec in zip(idxs, decoded):
            results.append({
                "idx": idx,
                "text": dec.text
            })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch transcribe MathSpeech audios with Whisper")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of audio files processed per forward pass")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader worker processes for audio loading")
    parser.add_argument("--num_samples", type=int, default=1101, help="Total expected number of audio files (indices 1..N)")
    parser.add_argument("--audio_dir", type=str, default="./MathSpeech", help="Directory containing <index>.mp3 files")
    parser.add_argument("--excel_path", type=str, default="./MathSpeech.xlsx", help="Path to original Excel with transcriptions")
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Use the new transcribe_batch helper to process all available audios once per
    # model. We keep two passes (base & small) so GPU memory stays reasonable.
    # -----------------------------------------------------------------------------

    BATCH_SIZE = args.batch_size
    num_samples = args.num_samples
    num_workers = args.num_workers

    # Reload dataframe with user-provided path (keep above usages consistent)
    df = pd.read_excel(args.excel_path)

    path_to_idx = {}

    # Collect paths that really exist and remember original indices
    audio_paths = []
    for idx in range(num_samples):
        p = os.path.join(args.audio_dir, f"{idx + 1}.mp3")
        if os.path.exists(p):
            audio_paths.append(p)
        else:
            print(f"index {idx + 1} is None.")

    # Initialize result containers
    base_result_list = [''] * num_samples
    small_result_list = [''] * num_samples

    dataset = WhisperAudioDataset(audio_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_whisper,
    )

    # Run transcription passes
    print("Starting base model transcription …")
    base_transcripts = transcribe_batch(dataloader, model_size="base", language="en")
    for res in base_transcripts:
        base_result_list[res["idx"]] = res["text"]

    print("Starting small model transcription …")
    small_transcripts = transcribe_batch(dataloader, model_size="small", language="en")
    for res in small_transcripts:
        small_result_list[res["idx"]] = res["text"]

    df["whisper_base_predSE"] = base_result_list
    df["whisper_small_predSE"] = small_result_list

    df.to_csv('../Experiments/result_ASR.csv', index=False)
    df.to_csv('../Ablation_Study/result_ASR.csv', index=False)
