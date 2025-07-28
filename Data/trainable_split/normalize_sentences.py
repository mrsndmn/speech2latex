from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
import pandas as pd
import re
import os
import sys
from process_formula import NormalizeFormula

import ast

if __name__ == "__main__":

    # hf_ds = Dataset.load_from_disk('./equations_test_new')
    hf_ds = load_dataset('marsianin500/Speech2Latex', num_proc=32)
    hf_ds.pop('equations')
    hf_ds.pop('equations_test')
    hf_ds.pop('equations_train')

    print('hf_ds', hf_ds)

    num_proc = min(32, os.cpu_count() or 1)  # cap at 8 to avoid oversubscription
    # num_proc = None

    for dataset_split in hf_ds.keys():

        # hf_ds[dataset_split] = hf_ds[dataset_split].select(range(2000))

        # hf_ds = hf_ds.remove_columns(set(hf_ds.column_names) - set(['whisper_text', 'sentence', 'language']))
        original_length = len(hf_ds[dataset_split])

        def remove_unescaped_percent(item):
            sentence = item.get('sentence')
            if not isinstance(sentence, str):
                return False
            if re.findall(r'(?<!\\)\s*%', sentence):
                return False
            return True

        hf_ds[dataset_split] = hf_ds[dataset_split].filter(remove_unescaped_percent, num_proc=num_proc)
        print("Filtered out: ", original_length - len(hf_ds[dataset_split]), " samples")
        original_length = len(hf_ds[dataset_split])

        def validate_formulas(batch):

            evaluated_formulas_info = []

            all_formulas = []
            for sentence, formula_info in zip(batch['sentence'], batch['formula_info']):
                formula_infos = ast.literal_eval(formula_info)
                evaluated_formulas_info.append(formula_infos)
                for formula_info in formula_infos:
                    formula_latex = formula_info[-1]
                    all_formulas.append(formula_latex)

            all_formulas = list(set(all_formulas))
            norm = NormalizeFormula(check_node=False)
            normalized = norm(all_formulas)

            normalized_formulas = dict()
            for formula_latex, normalized_formula in zip(all_formulas, normalized):
                normalized_formulas[formula_latex] = normalized_formula

            sentences_normalized = []
            for sentence, formula_infos in zip(batch['sentence'], evaluated_formulas_info):
                normalized_sentence = sentence
                processed_formulas = set()
                for formula_info in sorted(formula_infos, key=lambda x: len(x[-1]), reverse=True):
                    formula_latex = formula_info[-1]
                    if formula_latex in processed_formulas:
                        continue

                    normalized_formula = normalized_formulas[formula_latex]
                    if normalized_formula == '':
                        normalized_sentence = ''
                        break

                    assert formula_latex in normalized_sentence, f'`{formula_latex}` not in `{normalized_sentence}`'
                    normalized_sentence = normalized_sentence.replace(formula_latex, normalized_formula)
                    processed_formulas.add(formula_latex)

                sentences_normalized.append(normalized_sentence)

            return {
                # mark whether the formula successfully compiles (non-empty string)
                # 'sentence': [ f"$ {s} $" for s in batch['sentence']],
                'sentence_normalized': sentences_normalized
            }

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

            # if 'undefined' in item['sentence_normalized']:
            #     return False

            return True

        orig_length = len(hf_ds[dataset_split])
        hf_ds[dataset_split] = hf_ds[dataset_split].filter(cleanup_final, num_proc=num_proc)
        print("Filtered out: ", orig_length - len(hf_ds[dataset_split]), " samples")

    hf_ds.save_to_disk(f'./s2l_sentences_normalized')

    breakpoint()

