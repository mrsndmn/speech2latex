import time
import os
import pandas as pd

import datasets


if __name__ == "__main__":

    dataset_path = './MathBridge_train_cleaned_normalized_train_dataset'

    if not os.path.exists(dataset_path):

        df = pd.read_csv('./MathBridge_train_cleaned_normalized_train.csv')

        dataset = datasets.Dataset.from_pandas(df)

        dataset = dataset.remove_columns(['formula_normalized_2'])

        wav_files = set(map(lambda x: f'./mathbridge_cleaned_tts_wavs/{x}', os.listdir('./mathbridge_cleaned_tts_wavs')))

        dataset = dataset.map(lambda x: { 'audio_path': "./mathbridge_cleaned_tts_wavs/" + str(x['__index_level_0__']) + '.wav' }, batched=False)
        dataset = dataset.remove_columns(['__index_level_0__'])
        dataset_filtered = dataset.filter(lambda x: x['audio_path'] in wav_files)

        print('len(dataset)', len(dataset_filtered))

        dataset_filtered = dataset_filtered.cast_column('audio_path', datasets.Audio(sampling_rate=16000))

        dataset_filtered.save_to_disk(dataset_path)
    else:
        dataset_filtered = datasets.Dataset.load_from_disk(dataset_path)

    while True:
        ok = False
        try:
            dataset_filtered.push_to_hub('marsianin500/Speech2Latex', split='equations_mathbridge_clean')
            ok = True
        except Exception as e:
            print(e)
            time.sleep(10)

        if ok:
            break

