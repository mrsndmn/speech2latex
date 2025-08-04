
import os
from datasets import load_dataset, Audio

if __name__ == "__main__":

    NUM_SAMPLES = 100

    # Speech2Latex Equations and Sentences
    os.makedirs('./sample_datasets', exist_ok=True)

    dataset_dict = load_dataset('marsianin500/Speech2Latex', num_proc=32)

    dataset_dict.cast_column('audio_path', Audio(sampling_rate=16000))

    for key in dataset_dict.keys():
        dataset_dict[key] = dataset_dict[key].shuffle(seed=42).select(range(NUM_SAMPLES)).map(lambda x: x)

    dataset_dict.save_to_disk('./sample_datasets/speech2latex_equations_sentences_100_samples')

    # MathBridge Cleaned Subset Data
    dataset_dict = load_dataset('marsianin500/Speech2LatexMathBridge', num_proc=32)

    for key in dataset_dict.keys():
        dataset_dict[key] = dataset_dict[key].shuffle(seed=42).select(range(NUM_SAMPLES)).map(lambda x: x)

    dataset_dict.save_to_disk('./sample_datasets/speech2latex_mathbridge_100_samples')


