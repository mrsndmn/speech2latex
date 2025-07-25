import random
from tqdm.auto import tqdm
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import math

if __name__ == "__main__":
    dataset = load_dataset('marsianin500/Speech2Latex', split='equations', num_proc=16)

    full_equations = pd.read_csv('equations.csv')

    df_test = pd.read_csv('test_ENG_to_submit.csv')

    test_indices = []

    test_audio_paths = set(df_test['audio_path'].unique())

    full_equations['audio_path'] = full_equations['audio_path'].apply(lambda x: x.removeprefix('/home/jovyan/shares/SR006.nfs2/shares/SR006.nfs2/speech2latex/dataset/'))

    print("Splitting dataset based on audio_path correspondence...")

    test_formulas = set(df_test['latex'].unique())

    for idx, item in enumerate(tqdm(dataset)):
        audio_path = full_equations['audio_path'][idx]
        if audio_path in test_audio_paths:
            test_indices.append(idx)

            test_formulas.add(item['sentence'])
        else:
            if item['sentence'] in test_formulas:
                test_indices.append(idx)
                continue

    for idx, item in enumerate(tqdm(dataset)):
        audio_path = full_equations['audio_path'][idx]
        if item['sentence'] in test_formulas:
            test_indices.append(idx)
    # Create the splits
    test_dataset = dataset.select(set(test_indices))
    dev_dataset = dataset.select(set(range(len(dataset))) - set(test_indices))

    print(f"Test split: {len(test_dataset)} samples")
    print(f"Dev split: {len(dev_dataset)} samples")

    test_dataset.save_to_disk('equations_test_new')
    dev_dataset.save_to_disk('equations_dev_new')
