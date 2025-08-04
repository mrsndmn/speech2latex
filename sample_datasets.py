
import os
from datasets import load_dataset, Audio, Dataset

if __name__ == "__main__":

    NUM_SAMPLES = 10

    # Speech2Latex Equations and Sentences
    os.makedirs('./sample_datasets', exist_ok=True)

    dataset_dict = load_dataset('marsianin500/Speech2Latex', num_proc=32)

    for key in dataset_dict.keys():
        dataset_dict[key] = dataset_dict[key].shuffle(seed=42).select(range(NUM_SAMPLES))

    dataset_dict.save_to_disk(f'./sample_datasets/speech2latex_equations_sentences_{NUM_SAMPLES}_samples')

    # MathBridge Cleaned Subset Data
    dataset_dict = load_dataset('marsianin500/Speech2LatexMathBridge', num_proc=32)

    for key in dataset_dict.keys():
        dataset_dict[key] = dataset_dict[key].shuffle(seed=42).select(range(NUM_SAMPLES))

    dataset_dict.save_to_disk(f'./sample_datasets/speech2latex_mathbridge_{NUM_SAMPLES}_samples')


