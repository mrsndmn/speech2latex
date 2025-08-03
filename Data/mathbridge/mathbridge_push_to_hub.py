import time
import os
import pandas as pd

import datasets


if __name__ == "__main__":

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    dataset_path = './MathBridge_train_cleaned_normalized_train_dataset'

    if not os.path.exists(dataset_path):

        df = pd.read_csv('./mb_whisper.csv')

        dataset = datasets.Dataset.from_pandas(df)

        dataset = dataset.remove_columns(['formula_normalized_2'])

        wav_files = set(map(lambda x: f'./mathbridge_cleaned_tts_wavs/{x}', os.listdir('./mathbridge_cleaned_tts_wavs')))

        dataset = dataset.rename_columns({
            'equation': 'sentence',
            'spoken_English': 'pronunciation',
            'formula_normalized': 'sentence_normalized'
        })

        dataset = dataset.map(lambda x: {
            'audio_path': "./mathbridge_cleaned_tts_wavs/" + str(x['index']) + '.wav',
            'sentence': x['sentence'].removeprefix('$ ').removesuffix(' $'),
            'sentence_normalized': x['sentence_normalized'].removeprefix('$ ').removesuffix(' $'),
        }, batched=False)
        dataset = dataset.remove_columns(['__index_level_0__', 'index'])
        dataset_filtered = dataset.filter(lambda x: x['audio_path'] in wav_files)

        print('len(dataset)', len(dataset_filtered))
        assert len(wav_files) == len(dataset_filtered)

        dataset_filtered = dataset_filtered.cast_column('audio_path', datasets.Audio(sampling_rate=16000))

        dataset_filtered.save_to_disk(dataset_path, num_proc=32)
    else:
        dataset_filtered = datasets.Dataset.load_from_disk(dataset_path)

    # dataset_filtered = dataset_filtered.select(range(10))
    # dataset_filtered = dataset_filtered.remove_columns(['context_before', 'context_after'])

    while True:
        ok = False
        try:
            dataset_filtered.push_to_hub(
                'marsianin500/Speech2LatexMathBridge',
                split='equations_mathbridge_clean',
                max_shard_size='5GB'
            )
            ok = True
        except Exception as e:
            print("Failed to upload dataset to hub!")
            print(e)
            # breakpoint()
            time.sleep(10)

        if ok:
            break

