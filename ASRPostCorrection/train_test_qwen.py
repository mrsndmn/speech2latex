import sys
import datasets
from pytorch_lightning.loggers import WandbLogger, CSVLogger

import argparse
import json
import os
import pickle
import random
import string

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from tqdm.auto import tqdm

from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
# from peft import LoraConfig, get_peft_model, TaskType
from pytorch_lightning.loggers import CSVLogger
from dataset import ASRDataset, get_collate_function, get_dataloader
        
import pandas as pd

from s2l.eval import LatexInContextMetrics


from test_qwen import batched_model_generation

from qwen_pl import Model_pl, Config

def test(
        model,
        tokenizer,
        test_dataset,
        pron_column_name = 'whisper_text',
        latex_column_name = 'sentence',
    ):

    DEVICE='cuda'

    torch.set_default_dtype(torch.bfloat16)

    assert tokenizer.padding_side == 'left'

    model.eval()
    # model = torch.compile(model, mode='reduce-overhead')

    outputs = defaultdict(list)

    test_dataset = ASRDataset(test_dataset, pron_column_name=pron_column_name, latex_column_name=latex_column_name)

    # formulas normalization will be performed in batched_model_generation
    collate_function = get_collate_function(tokenizer, process_formulas=None)

    batch_size = 32
    test_loader = get_dataloader(test_dataset, batch_size, collate_function, num_workers=0, train=False)

    for batch in tqdm(test_loader):

        generated_latex = batched_model_generation(model, tokenizer, batch, device=DEVICE)

        predicted_text = generated_latex['predicted_text']
        target_text = generated_latex['target_text']

        outputs['latex_pred'].extend(predicted_text)
        outputs['latex_true'].extend(target_text)

    return outputs


if __name__ == "__main__":
    # jobs env crutches
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import nltk
    nltk.download('punkt_tab')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config-4.json')
    parser.add_argument('--dataset_split', required=True, choices=['sentences', 'equations'])
    parser.add_argument('--latex_column_name', required=True, choices=['sentence', 'sentence_normalized'])
    parser.add_argument('--language', required=True, choices=['eng', 'ru', 'multilingual'])
    parser.add_argument('--data_type', required=True, choices=['human', 'synthetic_small', 'synthetic_full', 'mix', 'mix_full'])

    parser.add_argument('--few_train_samples', type=int, default=None)
    parser.add_argument('--few_test_samples', type=int, default=None)
    parser.add_argument('--test_equations', action='store_true')
    parser.add_argument('--test_equations_math_speech_normalized', action='store_true')
    parser.add_argument('--test_equations_my_normalized', action='store_true')
    parser.add_argument('--test_equations_unnormalized', action='store_true')
    parser.add_argument('--test_sentences', action='store_true')
    args = parser.parse_args()


    with open(args.config, 'r') as config_file:
        config_dict = json.load(config_file)

    cfg = Config(**config_dict)

    torch.set_default_dtype(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_ckpt, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_ckpt, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
    # model = AutoModelForCausalLM.from_pretrained("ckpts/asr-sentence/version_24/tuned-model")

    ### Work with data
    collate_function = get_collate_function(tokenizer)

    train_dataset_split = args.dataset_split
    test_dataset_split = args.dataset_split

    train_dataset = datasets.load_dataset('marsianin500/Speech2Latex', split=f'{args.dataset_split}_train', num_proc=32)
    test_dataset = datasets.load_dataset('marsianin500/Speech2Latex', split=f'{args.dataset_split}_test', num_proc=32)

    pron_column_name = 'whisper_text'
    latex_column_name = args.latex_column_name

    columns_to_keep = {pron_column_name, latex_column_name, 'is_tts', 'language'}

    train_dataset = train_dataset.remove_columns(set(train_dataset.column_names) - columns_to_keep)
    test_dataset = test_dataset.remove_columns(set(test_dataset.column_names) - columns_to_keep)

    if 'full' in args.data_type:
        big_synthetic_dataset = datasets.Dataset.load_from_disk('../Data/mathbridge/MathBridge_train_cleaned_normalized_train_dataset/')
        big_synthetic_dataset = big_synthetic_dataset.add_column('is_tts', [1] * len(big_synthetic_dataset))
        big_synthetic_dataset = big_synthetic_dataset.add_column('language', ['eng'] * len(big_synthetic_dataset))

        big_synthetic_dataset = big_synthetic_dataset.remove_columns(set(big_synthetic_dataset.column_names) - columns_to_keep)

        print("Adding synthetic dataset to train dataset")
        print("Train dataset size", len(train_dataset))
        print("Synthetic dataset size", len(big_synthetic_dataset))

        train_dataset = datasets.concatenate_datasets([train_dataset, big_synthetic_dataset])
        print("Concatenated dataset size", len(train_dataset))

    def filter_by_language_and_data_type(item):
        if args.language != 'multilingual' and item['language'] != args.language:
            return False

        if args.data_type != 'mix':
            if 'synthetic' in args.data_type and item['is_tts'] == 0:
                return False
            elif args.data_type == 'human' and item['is_tts'] == 1:
                return False

        return True

    if args.few_train_samples is not None:
        train_dataset = train_dataset.select(range(args.few_train_samples))

    train_dataset = train_dataset.filter(filter_by_language_and_data_type)

    print("Train dataset size", len(train_dataset))
    print("Test dataset size", len(test_dataset))

    train_dataset = ASRDataset(train_dataset, pron_column_name=pron_column_name, latex_column_name=latex_column_name)
    train_loader = get_dataloader(train_dataset, cfg.batch_size, collate_function, cfg.num_workers, train=True)

    module = Model_pl(cfg, len(train_loader), model, tokenizer)

    random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    csv_logger = CSVLogger(save_dir=f"ckpts/{cfg.exp_name}/{args.dataset_split}_{args.latex_column_name}_{args.language}_{args.data_type}_{random_chars}")
    os.makedirs(csv_logger.save_dir, exist_ok=True)

    print("output dir", csv_logger.save_dir)

    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        logger=csv_logger,
        enable_checkpointing=False,
        # val_check_interval=0.2,
        # limit_train_batches=1,
    )

    trainer.fit(
        model=module,
        train_dataloaders=train_loader,
    )

    module.eval()
    module.to('cuda')

    if args.language != 'multilingual':
        test_dataset = test_dataset.filter(lambda x: x['language'] == args.language)

    if args.few_test_samples is not None:
        test_dataset = test_dataset.select(range(args.few_test_samples))

    # Compute Metrics
    results_save_dir = csv_logger.save_dir

    outputs = test(
        model,
        tokenizer,
        test_dataset,
        latex_column_name=latex_column_name,
    )

    evaluation_df = pd.DataFrame({**outputs, 'is_tts': test_dataset['is_tts']})

    evaluation_df.to_csv(os.path.join(results_save_dir, 'evaluation_generations.csv'), index=False)

    evaluation_df_mix = evaluation_df.copy()
    evaluation_df_artificial = evaluation_df[ evaluation_df['is_tts'] == 1 ].copy()
    evaluation_df_humans = evaluation_df[ evaluation_df['is_tts'] == 0 ].copy()

    # Mix metrics
    metrics_splits = [
        (evaluation_df_artificial, 'artificial'),
        (evaluation_df_humans, 'humans'),
        (evaluation_df_mix, 'mix'),
    ]

    for evaluation_df, test_split in metrics_splits:
        print(f"Computing metrics for {test_split}")

        in_context_metrics = LatexInContextMetrics()
        metrics_values = in_context_metrics.compute_all(evaluation_df['latex_pred'].values.tolist(), evaluation_df['latex_true'].values.tolist(), compute_text_only=(args.dataset_split == 'sentences'))
        in_context_metrics.dump(metrics_values)

        output_file_path = os.path.join(results_save_dir, f'{test_split}_metrics.json')
        with open(output_file_path, 'w') as f:
            json.dump(metrics_values, f)
            print(f"Metrics for {test_split} saved to {output_file_path}")


