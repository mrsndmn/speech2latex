import sys
import time

from datasets import Dataset, DatasetDict, load_dataset

def retry_untill_success(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            time.sleep(10)

if __name__ == "__main__":

    def push_to_hub(dataset_path):
        sentences_ds = DatasetDict.load_from_disk(dataset_path)
        for split in sentences_ds.keys():
            sentences_ds[split].push_to_hub('marsianin500/Speech2Latex', split=split, max_shard_size='2GB')
        return True

    retry_untill_success(push_to_hub, './s2l_equations_normalized')
    retry_untill_success(push_to_hub, './s2l_sentences_normalized')
