import torch
import torchaudio

import os
from datasets import load_dataset, Audio, Dataset

if __name__ == "__main__":

    NUM_SAMPLES = 10

    # Speech2Latex Equations and Sentences
    os.makedirs('./sample_datasets', exist_ok=True)

    dataset_dict = load_dataset('marsianin500/Speech2Latex', num_proc=32)

    for key in dataset_dict.keys():
        os.makedirs(f'./sample_datasets/{key}', exist_ok=True)
        shuffled_dataset = dataset_dict[key].shuffle(seed=42).select(range(NUM_SAMPLES))
        for i, item in enumerate(shuffled_dataset):
            audio_path = item['audio_path']['array']
            sampling_rate = item['audio_path']['sampling_rate']

            torchaudio.save(f'./sample_datasets/{key}/{i:02d}.wav', torch.from_numpy(audio_path).unsqueeze(0), sampling_rate)


    # MathBridge Cleaned Subset Data
    dataset_dict = load_dataset('marsianin500/Speech2LatexMathBridge', num_proc=32)

    for key in dataset_dict.keys():
        os.makedirs(f'./sample_datasets/{key}', exist_ok=True)
        shuffled_dataset = dataset_dict[key].shuffle(seed=42).select(range(NUM_SAMPLES))

        key = f'mathbridge_{key}'
        for i, item in enumerate(shuffled_dataset):
            audio_array = item['audio_path']['array']
            sampling_rate = item['audio_path']['sampling_rate']

            torchaudio.save(f'./sample_datasets/{key}/{i:02d}.wav', torch.from_numpy(audio_array).unsqueeze(0), sampling_rate)



