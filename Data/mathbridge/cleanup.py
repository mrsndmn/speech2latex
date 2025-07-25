from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
import pandas as pd
import re

from process_formula import NormalizeFormula


if __name__ == "__main__":

    dataset = load_dataset('Kyudan/MathBridge', split='train')

    # Cleanup the dataset
    # Remove Duplicates

    df = pd.DataFrame({
        "equation": dataset['equation'],
        "spoken_English": dataset['spoken_English'],
        "context_after": dataset['context_after'],
        "context_before": dataset['context_before'],
    })

    df['equation_norm'] = df['equation'].apply(lambda x: x.replace(' ', '').replace('\displaystyle', '').replace('\operatorname', '').lower())

    orig_length = len(df)
    print("Length of the dataset: ", orig_length)

    df = df.drop_duplicates(subset=['equation_norm'])
    df = df.drop(columns=['equation_norm'])

    print("Length of the dataset after removing duplicates: ", len(df))
    print("Number of duplicates: ", orig_length - len(df))

    hf_ds = Dataset.from_pandas(df)
    original_length = len(hf_ds)

    # Detects english sentences in equations eg:
    # $ The cat san on the mat. $
    # We require formula contain at least one digit or special symbol or operator ...
    simple_formula_pattern = re.compile(r'^\s*\$[\sa-zA-Z._\-\(\)\.,~]+\$\s*$')

    def filter_simple_samples(sample):
        if 'None' in sample['spoken_English']:
            return False

        if 'for outcome' in sample['equation'].lower():
            return False

        if sample['spoken_English'] == 'dot dot dot':
            return False

        if len(sample['spoken_English']) < 5:
            return False

        if len(sample['spoken_English']) < len(sample['equation']) * 0.8:
            return False

        # if '\\bm' in sample['equation']:
        #     return False
        return simple_formula_pattern.match(sample['equation']) is None

    hf_ds = hf_ds.filter(filter_simple_samples)

    print("Filtered out simple samples: ", original_length - len(hf_ds), " samples")

    valid_formulas_indexes = []

    batch_size = 1000

    # Validate formulas with NormalizeFormula in parallel to speed up LaTeX compilation check
    import os

    def validate_formulas(batch):
        norm = NormalizeFormula()
        normalized = norm(batch['equation'])
        return {
            # mark whether the formula successfully compiles (non-empty string)
            'is_valid_formula': [n != '' for n in normalized]
        }

    # number of parallel processes; default to all CPU cores if available
    num_proc = min(16, os.cpu_count() or 1)  # cap at 8 to avoid oversubscription

    print(f"\nRunning LaTeX compilation check in parallel using {num_proc} workers …")
    hf_ds = hf_ds.map(
        validate_formulas,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Validate formulas",
    )

    # keep only successfully compiled formulas
    before_compile = len(hf_ds)
    hf_ds_compiled = hf_ds.filter(lambda x: x['is_valid_formula'])
    hf_ds_compiled = hf_ds_compiled.remove_columns('is_valid_formula')
    print("Filtered out non-compilable formulas:", before_compile - len(hf_ds_compiled), "samples")

    # replace old variable name for downstream code compatibility
    hf_ds_clean_final = hf_ds_compiled

    print("Length of the dataset after compilation validation:", len(hf_ds_clean_final))

    hf_ds_clean_final.save_to_disk('MathBridge_train_cleaned.dataset')
    print("Saved cleaned dataset to MathBridge_train_cleaned.dataset")

    df = hf_ds_clean_final.to_pandas()

    # plot length distribution of the equation
    plt.hist(df['equation'].apply(len), bins=100)
    plt.title('Length Distribution of Equation')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig('MathBridge_train_cleaned_equation_length_distribution.png')
    print("Saved equation length distribution to MathBridge_train_cleaned_equation_length_distribution.png")

    plt.close()
    plt.clf()

    # plot length distribution of the spoken_English
    plt.hist(df['spoken_English'].apply(len), bins=100)
    plt.title('Length Distribution of Spoken English')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig('MathBridge_train_cleaned_spoken_English_length_distribution.png')
    print("Saved spoken English length distribution to MathBridge_train_cleaned_spoken_English_length_distribution.png")
    plt.close()
    plt.clf()

    df.to_csv('MathBridge_train_cleaned.csv', index=False)

    print(f"Saved {len(df)} rows to MathBridge_train_cleaned.csv")



