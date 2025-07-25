import os
from glob import glob
import pandas as pd
import numpy as np
from datasets import Dataset

def process_train():

    total_shards = 6

    all_dataframes = []

    for i in range(total_shards):
        shard_file = f"./MathSpeech_s2l_equations_full_shard_{i}_of_{total_shards}_generated.csv"
        df = pd.read_csv(shard_file)
        # print(df.head())

        all_dataframes.append(df)

    all_dataframes = pd.concat(all_dataframes)

    dataset = Dataset.load_from_disk(f"../../Data/trainable_split/equations_dev_new/")

    for pandas_latex, dataset_latex in zip(all_dataframes['LaTeX'], dataset['sentence']):
        if dataset_latex is None and np.isnan(pandas_latex):
            continue
        assert pandas_latex == dataset_latex, f"LaTeX mismatch: {pandas_latex} != {dataset_latex}"

    all_dataframes['whisper_large_transcription'] = dataset['whisper_text']
    all_dataframes['language'] = dataset['language']

    all_dataframes = all_dataframes[all_dataframes['language'] == 'eng']

    result_file = f"./MathSpeech_s2l_equations_full_generated_with_whisper_en.csv"
    all_dataframes.to_csv(result_file, index=False)
    print(f"saved train to {result_file}")


def process_test():

    test_dataset = Dataset.load_from_disk(f"../../Data/trainable_split/equations_test_new/")

    test_generation_normalized = pd.read_csv(f"./s2l_equations_test_full_normalized.csv")

    for pandas_latex, dataset_latex in zip(test_generation_normalized['LaTeX'], test_dataset['sentence']):
        assert pandas_latex == dataset_latex, f"LaTeX mismatch: {pandas_latex} != {dataset_latex}"

    test_generation_normalized['whisper_large_transcription'] = test_dataset['whisper_text']

    test_generation_normalized['language'] = test_dataset['language']

    test_generation_normalized = test_generation_normalized[test_generation_normalized['language'] == 'eng']


    result_file = f"./s2l_equations_test_full_normalized_with_whisper_en.csv"
    test_generation_normalized.to_csv(result_file, index=False)
    print(f"saved test to {result_file}")


if __name__ == "__main__":

    process_train()
    process_test()