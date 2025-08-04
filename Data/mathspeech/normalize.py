from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
import pandas as pd
import re
import os

from process_formula import NormalizeFormula


if __name__ == "__main__":

    hf_ds = load_dataset('mrsndmn/MathSpeech_whisper_transcribed')
    original_length = len(hf_ds)

    def validate_formulas(batch):
        norm = NormalizeFormula(check_node=False)

        latex_no_dollar = []
        for latex in batch['LaTeX']:
            latex = latex.replace('\\displaystyle', '')
            latex = latex.removesuffix('$').removeprefix('$')
            latex = latex.strip()
            latex_no_dollar.append(latex)

        normalized = norm(latex_no_dollar)

        return {
            # mark whether the formula successfully compiles (non-empty string)
            'latex_normalized': [n for n in normalized]
        }

    # number of parallel processes; default to all CPU cores if available
    num_proc = min(32, os.cpu_count() or 1)  # cap at 8 to avoid oversubscription
    batch_size = 100

    print(f"\nRunning LaTeX compilation check in parallel using {num_proc} workers …")

    hf_ds = hf_ds.map(
        validate_formulas,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Validate formulas double",
    )

    orig_len = len(hf_ds['train'])
    hf_ds = hf_ds.filter(lambda x: x['latex_normalized'] != '')
    new_len = len(hf_ds['train'])

    print(f"Filtered out: {orig_len - new_len} samples")

    hf_ds.save_to_disk('./MathSpeech_whisper_transcribed_normalized')

    hf_ds.push_to_hub('mrsndmn/MathSpeech_whisper_transcribed_normalized')
