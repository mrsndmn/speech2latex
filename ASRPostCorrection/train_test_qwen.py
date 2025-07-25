
from pytorch_lightning.loggers import WandbLogger

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

from train_qwen import Model_pl, Config

def test(
        model, test_file_csv,
        pron_column_name = 'whisper_text',
        latex_column_name = 'sentence',
        few_samples = None,
        compute_text_only = True,
    ):

    DEVICE='cuda'

    torch.set_default_dtype(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', padding_side="left")

    model.eval()
    model = torch.compile(model, mode='reduce-overhead')

    outputs = defaultdict(list)
    df = pd.read_csv(test_file_csv)
    if few_samples is not None:
        df = df.sample(few_samples, random_state=42)

    np.random.seed(42)
    df = df.fillna({"pron": "", "latex":""})

    val_dataset = ASRDataset(df, pron_column_name=pron_column_name, latex_column_name=latex_column_name)

    # formulas normalization will be performed in batched_model_generation
    collate_function = get_collate_function(tokenizer, process_formulas=None)

    batch_size = 32
    val_loader = get_dataloader(val_dataset, batch_size, collate_function, num_workers=0, train=False)

    for batch in tqdm(val_loader):

        generated_latex = batched_model_generation(model, tokenizer, batch, device=DEVICE)

        predicted_text = generated_latex['predicted_text']
        target_text = generated_latex['target_text']

        outputs['latex_pred'].extend(predicted_text)
        outputs['latex_true'].extend(target_text)

    result_file_name = 'predictions_result_{i}.csv'
    num_tries = 1000
    for i in range(num_tries):
        if not os.path.exists(result_file_name.format(i=i)):
            result_file_name = result_file_name.format(i=i)
            break

        if i == num_tries:
            print("Failed to save evaluation results")
            result_file_name = None

    in_context_metrics = LatexInContextMetrics()
    metrics_values = in_context_metrics.compute_all(outputs['latex_pred'], outputs['latex_true'], compute_text_only=compute_text_only)

    in_context_metrics.dump(metrics_values)

    return metrics_values


if __name__ == "__main__":
    # jobs env crutches
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import nltk
    nltk.download('punkt_tab')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config-4.json')
    parser.add_argument('--train_df', type=str)
    parser.add_argument('--few_train_samples', type=int, default=None)
    parser.add_argument('--val_df', type=str)
    parser.add_argument('--few_val_samples', type=int, default=None)
    parser.add_argument('--few_test_samples', type=int, default=None)
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--test_equations', action='store_true')
    parser.add_argument('--test_equations_math_speech_normalized', action='store_true')
    parser.add_argument('--test_equations_my_normalized', action='store_true')
    parser.add_argument('--test_equations_unnormalized', action='store_true')
    parser.add_argument('--test_sentences', action='store_true')
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)

    with open(args.config, 'r') as config_file:
        config_dict = json.load(config_file)

    cfg = Config(**config_dict)

    torch.set_default_dtype(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_ckpt, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_ckpt, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
    # model = AutoModelForCausalLM.from_pretrained("ckpts/asr-sentence/version_24/tuned-model")

    os.makedirs(f"ckpts/{cfg.exp_name}", exist_ok=True)

    ### Work with data
    collate_function = get_collate_function(tokenizer)
    train_df = pd.read_csv(args.train_df)
    train_df = train_df.fillna({"pron": "", "latex":""})

    if args.few_train_samples is not None:
        train_df = train_df.sample(args.few_train_samples, random_state=42)

    val_df = None
    if args.val_df is not None:
        val_df = pd.read_csv(args.val_df)
        # val_df = val_df[val_df['is_tts'] == 1]
        val_df = val_df.fillna({"pron": "", "latex":""})
        np.random.seed(42)
        if args.few_val_samples is not None:
            val_df = val_df.sample(args.few_val_samples, random_state=42)
        else:
            val_df = val_df.sample(cfg.batch_size * 10, random_state=42)

    pron_column_name = cfg.pron_column_name
    # pron_column_name = 'pronunciation'
    latex_column_name = cfg.latex_column_name
    train_dataset = ASRDataset(train_df, pron_column_name=pron_column_name, latex_column_name=latex_column_name)
    train_loader = get_dataloader(train_dataset, cfg.batch_size, collate_function, cfg.num_workers, train=True)

    val_dataset = None
    val_loader = None
    if val_df is not None:
        val_dataset = ASRDataset(val_df, pron_column_name=pron_column_name, latex_column_name=latex_column_name)
        val_loader = get_dataloader(val_dataset, cfg.batch_size, collate_function, cfg.num_workers, train=False)

    module = Model_pl(cfg, len(train_loader), torch.compile(model), tokenizer)

    random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=6))

    wandb_logger = WandbLogger(project="speech2latex", name=f"{cfg.exp_name}_{random_chars}")

    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        logger=wandb_logger,
        enable_checkpointing=False,
        val_check_interval=None,
        # val_check_interval=0.2,
        limit_val_batches=10,
        # limit_train_batches=1,
    )

    trainer.fit(model=module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
    )

    module.eval()
    module.to('cuda')

    if args.test_sentences:
        print("Artificial test")
        metrics_a = test(
            model,
            "../Data/latex_in_context_tts/final_table_latex_text_tts_humans_test-2-artificial.csv",
            few_samples=args.few_test_samples,
        )

        print("Humans test")
        metrics_h = test(
            model,
            "../Data/latex_in_context_tts/final_table_latex_text_tts_humans_test-2-humans.csv",
            few_samples=args.few_test_samples,
        )

        with open(os.path.join(experiment_dir, 'humans_metrics.pickle'), 'wb') as f:
            pickle.dump(metrics_h, f)

        with open(os.path.join(experiment_dir, 'artificial_metrics.pickle'), 'wb') as f:
            pickle.dump(metrics_a, f)

    if args.test_equations:
        raise NotImplementedError("Test equations not implemented")

    if args.test_equations_math_speech_normalized:
        metrics_a = test(
            model,
            "../MathSpeech/Experiments/s2l_equations_test_full_normalized_with_whisper.csv",
            few_samples=args.few_test_samples,
            pron_column_name = 'whisper_large_transcription',
            latex_column_name = 'MathSpeech_LaTeX_result',
            compute_text_only = False,
        )

    if args.test_equations_my_normalized:
        metrics_a = test(
            model,
            "../Data/trainable_split/s2l_equations_test_normalized_en.csv",
            few_samples=args.few_test_samples,
            pron_column_name = 'whisper_text',
            latex_column_name = 'formula_normalized',
            compute_text_only = False,
        )

    if args.test_equations_unnormalized:
        metrics_a = test(
            model,
            "../Data/trainable_split/s2l_equations_test_normalized_en.csv",
            few_samples=args.few_test_samples,
            pron_column_name = 'whisper_text',
            latex_column_name = 'sentence',
            compute_text_only = False,
        )
