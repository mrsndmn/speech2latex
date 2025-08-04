import torch
import torchaudio

import os
from datasets import load_dataset, Audio, Dataset

if __name__ == "__main__":

    NUM_SAMPLES = 5

    # Speech2Latex Equations and Sentences
    os.makedirs('./sample_datasets', exist_ok=True)

    dataset_dict = load_dataset('marsianin500/Speech2Latex', num_proc=32)

    for key in dataset_dict.keys():
        os.makedirs(f'./sample_datasets/{key}', exist_ok=True)

        shuffled_dataset = dataset_dict[key]

        for language in [ 'eng', 'ru' ]:

            if 'sentence' in key and language == 'ru':
                continue

            print(f'Processing {key} for {language}')

            shuffled_dataset = shuffled_dataset.filter(lambda x: x['language'] == language, num_proc=32)

            shuffled_dataset_tts = shuffled_dataset.filter(lambda x: int(x['is_tts']) == 1, num_proc=32).select(range(NUM_SAMPLES))
            shuffled_dataset_human = shuffled_dataset.filter(lambda x: int(x['is_tts']) == 0, num_proc=32).select(range(NUM_SAMPLES))

            assert len(shuffled_dataset_tts) == NUM_SAMPLES
            assert len(shuffled_dataset_human) == NUM_SAMPLES

            for i, item in enumerate(shuffled_dataset_tts):
                audio_path = item['audio_path']['array']
                sampling_rate = item['audio_path']['sampling_rate']

                torchaudio.save(f'./sample_datasets/{key}/tts_{language}_{i:02d}.wav', torch.from_numpy(audio_path).unsqueeze(0), sampling_rate)

            for i, item in enumerate(shuffled_dataset_human):
                audio_path = item['audio_path']['array']
                sampling_rate = item['audio_path']['sampling_rate']

                torchaudio.save(f'./sample_datasets/{key}/human_{language}_{i:02d}.wav', torch.from_numpy(audio_path).unsqueeze(0), sampling_rate)


        # # MathBridge Cleaned Subset Data
        # dataset_dict = load_dataset('marsianin500/Speech2LatexMathBridge', num_proc=32)

        # for key in dataset_dict.keys():
        #     shuffled_dataset = dataset_dict[key].shuffle(seed=42).select(range(NUM_SAMPLES))

        #     key = f'mathbridge_{key}'
        #     os.makedirs(f'./sample_datasets/{key}', exist_ok=True)
        #     for i, item in enumerate(shuffled_dataset):
        #         audio_array = item['audio_path']['array']
        #         sampling_rate = item['audio_path']['sampling_rate']

        #         torchaudio.save(f'./sample_datasets/{key}/{i:02d}.wav', torch.from_numpy(audio_array).unsqueeze(0), sampling_rate)
