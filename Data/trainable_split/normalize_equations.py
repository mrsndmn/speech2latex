from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
import pandas as pd
import re
import os
import sys
from process_formula import NormalizeFormula


if __name__ == "__main__":

    # hf_ds = Dataset.load_from_disk('./equations_test_new')
    hf_ds = load_dataset('marsianin500/Speech2Latex', num_proc=32)
    hf_ds.pop('equations')
    hf_ds.pop('sentences_test')
    hf_ds.pop('sentences_train')

    print('hf_ds', hf_ds)

    for dataset_split in hf_ds.keys():


        # hf_ds = hf_ds.remove_columns(set(hf_ds.column_names) - set(['whisper_text', 'sentence', 'language']))
        original_length = len(hf_ds[dataset_split])

        def remove_unescaped_percent(item):
            sentence = item.get('sentence')
            if not isinstance(sentence, str):
                return False
            if re.findall(r'(?<!\\)\s*%', sentence):
                return False
            return True

        hf_ds[dataset_split] = hf_ds[dataset_split].filter(remove_unescaped_percent)
        print("Filtered out: ", original_length - len(hf_ds[dataset_split]), " samples")
        original_length = len(hf_ds[dataset_split])

        def validate_formulas(batch):
            norm = NormalizeFormula(check_node=False)
            normalized = norm(batch['sentence'])

            return {
                # mark whether the formula successfully compiles (non-empty string)
                # 'sentence': [ f"$ {s} $" for s in batch['sentence']],
                'sentence_normalized': normalized
            }

        # number of parallel processes; default to all CPU cores if available
        num_proc = min(32, os.cpu_count() or 1)  # cap at 8 to avoid oversubscription
        batch_size = 100

        print(f"\nRunning LaTeX compilation check in parallel using {num_proc} workers ...")

        hf_ds[dataset_split] = hf_ds[dataset_split].map(
            validate_formulas,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Validate formulas double",
        )

        def cleanup_final(item):
            # Not compiled
            if item['sentence_normalized'] in ['$  $', '']:
                return False
            # Too short item
            if len(item['sentence_normalized']) < 5:
                return False

            if 'undefined' in item['sentence_normalized']:
                return False

            return True

        orig_length = len(hf_ds[dataset_split])
        hf_ds[dataset_split] = hf_ds[dataset_split].filter(cleanup_final)
        print("Filtered out: ", orig_length - len(hf_ds[dataset_split]), " samples")

    hf_ds.save_to_disk(f'./s2l_equations_normalized')

    breakpoint()

