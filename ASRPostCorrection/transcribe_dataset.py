import argparse
import os
from typing import List, Dict

import datasets
import pandas as pd
import torch
import torchaudio
import whisper
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a dataset split with Whisper and export a CSV."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="marsianin500/Speech2Latex",
        help="HuggingFace dataset path or local path (default: marsianin500/Speech2Latex)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="equations_train",
        help="Dataset split to use (default: equations_train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of samples to process; use -1 for all (default: 10)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["small", "base"],
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model sizes to run (default: small base)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language hint for Whisper (default: en)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help='Device to use for inference (default: "auto")',
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help='Path to output CSV file (default: "<split>.csv")',
    )
    return parser.parse_args()


def resolve_device(user_device: str) -> str:
    if user_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return user_device


def load_models(model_names: List[str], device: str) -> Dict[str, "whisper.Whisper"]:
    models: Dict[str, "whisper.Whisper"] = {}
    for name in model_names:
        models[name] = whisper.load_model(name, device=device)
    return models


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    dataset = datasets.load_dataset(args.dataset_path, split=args.split)
    if args.limit is not None and args.limit >= 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    models = load_models(args.models, device=device)

    # Prepare collectors
    collected_texts: Dict[str, List[str]] = {f"whisper_{m}_text": [] for m in args.models}
    latex: List[str] = []
    pronunciation: List[str] = []

    for item in tqdm(dataset):
        # Always work on CPU tensor for torchaudio resample stability, then pass numpy to whisper
        audio_tensor = torch.tensor(item["audio_path"]["array"], dtype=torch.float32, device="cpu")
        sample_rate = item["audio_path"]["sampling_rate"]

        pronunciation.append(item["pronunciation"])
        latex.append(item["sentence"])

        if sample_rate != 16000:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)

        # Whisper accepts torch or numpy; pass numpy array
        audio_np = audio_tensor.numpy()

        for model_name, model in models.items():
            result = model.transcribe(audio_np, language=args.language, fp16=(device == "cuda"))
            collected_texts[f"whisper_{model_name}_text"].append(result["text"])

    data = {
        **collected_texts,
        "latex": latex,
        "pronunciation": pronunciation,
    }
    df = pd.DataFrame.from_dict(data)

    output_file = args.output_file or f"{args.split}_whisper_small_base_transcriptions.csv"
    df.to_csv(output_file, index=False)
    print("saved to", output_file)


if __name__ == "__main__":
    main()
