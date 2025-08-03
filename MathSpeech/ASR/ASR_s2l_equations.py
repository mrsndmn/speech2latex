# Hugging Face datasets provides the audio column, but we alias here to avoid
# confusion with the torch.utils.data.Dataset class.
import datasets
from datasets import Dataset as HFDataset, Audio
import argparse
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
import pandas as pd
import whisper
import numpy as np
from ASR import transcribe_batch

# -----------------------------------------------------------------------------
# Custom Dataset that wraps a Hugging Face `Dataset` to return Whisper-ready
# log-mel spectrogram tensors one at a time so it can be consumed by a standard
# PyTorch DataLoader.
# -----------------------------------------------------------------------------


class S2LEquationsWhisperAudioDataset(TorchDataset):
    """Loads and preprocesses each audio sample for Whisper."""

    def __init__(self, dataset: HFDataset):
        # Keep a reference to the underlying Hugging Face dataset holding the
        # raw audio samples.
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """Return a tuple (mel, idx) where `mel` is the log-mel spectrogram
        tensor of the audio sample at position `idx` in the underlying dataset.
        """

        item = self.dataset[idx]

        # The HF Audio feature returns a dict with keys: path, array,
        # sampling_rate. We only need the raw waveform.
        audio_dict = item["audio_path"]
        audio = audio_dict["array"]

        # Whisper expects a float32 waveform and a fixed 30-second input length.
        audio = whisper.pad_or_trim(audio.astype(np.float32))
        mel = whisper.log_mel_spectrogram(audio)

        return mel, idx

# Collate function identical to the one used in generic ASR helper.
def _collate_whisper(batch):
    mels, idxs = zip(*batch)  # type: ignore
    mel_batch = torch.stack(mels)
    return mel_batch, list(idxs)


if __name__ == "__main__":

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='base')
    parser.add_argument('--shard_number', type=int, default=0)
    parser.add_argument('--total_shards', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, default='../../Data/trainable_split/equations_dev_new/')
    parser.add_argument('--dataset_split', type=str, default=None)
    parser.add_argument('--dataset_column', type=str, default='sentence_normalized')
    parser.add_argument('--output_prefix', type=str, default='../Experiments/result_ASR_s2l_equations_')

    args = parser.parse_args()

    model_type = args.model_type
    assert model_type in ['base', 'small'], 'Model type must be either base or small'

    print(f'ASR S2L Equations {model_type}')

    if args.dataset_path.startswith('marsianin500/'):
        dataset = datasets.load_dataset(args.dataset_path)
        dataset = dataset[args.dataset_split]
    else:
        dataset = HFDataset.load_from_disk(args.dataset_path)

    # Cast the audio column to the expected sampling rate for Whisper.
    dataset = dataset.cast_column('audio_path', Audio(sampling_rate=16000))

    # ---------------------------------------------------------------------
    # Optional sharding: process only a subset of the dataset when running
    # multiple jobs in parallel. The dataset is split into `total_shards`
    # contiguous pieces and only the shard with index `shard_number` is kept.
    # ---------------------------------------------------------------------
    if args.total_shards > 1:
        assert 0 <= args.shard_number < args.total_shards, (
            "shard_number must be in the range [0, total_shards)."
        )
        dataset = dataset.shard(
            num_shards=args.total_shards,
            index=args.shard_number,
            contiguous=True,
        )
        print(
            f"Processing shard {args.shard_number + 1}/{args.total_shards}: {len(dataset)} samples"
        )

    df_dict = {
        'LaTeX': dataset[args.dataset_column],
    }

    batch_size = 64
    num_workers = 32

    whisper_dataset = S2LEquationsWhisperAudioDataset(dataset)
    dataloader = DataLoader(
        whisper_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_whisper,
    )

    results_ordered = [ '' ] * len(dataset)

    results = transcribe_batch(dataloader, model_size=args.model_type, language="en")

    for result in results:
        results_ordered[result['idx']] = result['text']

    df_key = f'whisper_{model_type}_predSE'

    df_dict[df_key] = results_ordered

    df = pd.DataFrame(df_dict)

    df.to_csv(f'{args.output_prefix}{model_type}_shard_{args.shard_number}_of_{args.total_shards}.csv', index=False)