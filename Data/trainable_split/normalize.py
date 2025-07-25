from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
import pandas as pd
import re
import os

from process_formula import NormalizeFormula


if __name__ == "__main__":

    hf_ds = Dataset.load_from_disk('./equations_test_new')
    hf_ds = hf_ds.remove_columns(set(hf_ds.column_names) - set(['whisper_text', 'sentence', 'language']))
    original_length = len(hf_ds)

    def remove_unescaped_percent(item):
        if item['language'] != 'eng':
            return False

        if re.findall(r'(?<!\\)\s*%', item['sentence']):
            return False
        return True

    hf_ds = hf_ds.filter(remove_unescaped_percent)
    print("Filtered out: ", original_length - len(hf_ds), " samples")
    original_length = len(hf_ds)


    def validate_formulas(batch):
        norm = NormalizeFormula(check_node=False)
        normalized = norm(batch['sentence'])

        return {
            # mark whether the formula successfully compiles (non-empty string)
            'sentence': [ f"$ {s} $" for s in batch['sentence']],
            'formula_normalized': [f"$ {n} $" for n in normalized]
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

    def cleanup_final(item):
        # Not compiled
        if item['formula_normalized'] in ['$  $', '']:
            return False
        # Too short item
        if len(item['formula_normalized']) < 9:
            return False

        if 'undefined' in item['formula_normalized']:
            return False

        return True

    orig_length = len(hf_ds)
    hf_ds = hf_ds.filter(cleanup_final)
    print("Filtered out: ", orig_length - len(hf_ds), " samples")

    df_normalized = hf_ds.to_pandas()
    df_normalized.to_csv('s2l_equations_test_normalized_en.csv', index=False)

