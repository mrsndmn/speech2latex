import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
from tqdm.auto import tqdm
import evaluate
import re


from typing import List

import numpy as np
import pandas as pd

import os
from collections import defaultdict
import argparse

from dataset import ASRDataset, get_collate_function, get_dataloader

from s2l.eval import LatexInContextMetrics


def batched_model_generation(model, tokenizer, batch, device=None):
    
    batch_size = batch['input_ids'].shape[0]
    max_new_tokens = batch['assistant_masks'].sum(dim=-1).max().item() * 2
    gen_params = {
        "do_sample": False,
        "top_p": None,
        "top_k": None,
        "temperature": None,
        "min_new_tokens": 1,
        "max_new_tokens": max_new_tokens,
        "early_stopping": True,
        "num_beams": 3,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "forced_eos_token_id": tokenizer.eos_token_id,
        "stop_strings": [tokenizer.eos_token, '<|im_end|>'],
        "tokenizer": tokenizer,
        "use_cache": True,
        "no_repeat_ngram_size": 4,
        "num_return_sequences": 1,
    }
    
    with torch.no_grad():
        out = model.generate(
            inputs=batch['generation_input_ids'].to(device),
            attention_mask=batch['generation_attention_mask'].to(device),
            **gen_params,
        )

    assistant_answer = batch['input_ids'].clone().to(device)
    assistant_answer[~batch['assistant_masks'].bool().to(device)] = tokenizer.pad_token_id
    
    generated_tokens_mask = torch.ones([batch_size, max_new_tokens], device=out.device)
    generation_assistant_masks = torch.cat([batch['generation_assistant_masks'].to(device), generated_tokens_mask], dim=-1)
    generation_assistant_masks = generation_assistant_masks[:, :out.shape[1]]

    out[~generation_assistant_masks.bool()] = tokenizer.pad_token_id

    predicted_text_with_special_tokens = tokenizer.batch_decode(out, skip_special_tokens=False)
    predicted_text = []

    
    for latex in predicted_text_with_special_tokens:
        latex: str
        while latex.startswith('<|endoftext|>'):
            latex = latex.removeprefix('<|endoftext|>')
        
        if '<|im_end|>' in latex:
            im_end_index = latex.index('<|im_end|>')
            latex = latex[:im_end_index]
        elif '<|endoftext|>' in latex:
            end_of_text_index = latex.index('<|endoftext|>')
            latex = latex[:end_of_text_index]

        predicted_text.append(latex.strip())

    target_text = tokenizer.batch_decode(assistant_answer, skip_special_tokens=True)
    target_text = [ t.strip() for t in target_text ]

    print("predicted     ", predicted_text[0])
    print("target_text   ", target_text[0])

    return {
        "predicted_text": predicted_text,
        "target_text": target_text,
    }


# Test Qwen2.5 MathSpeech
# python -m pdb -c continue test_qwen.py --ckpt ckpts/tts-in-context/version_59 --test_file_csv ./MathSpeech_whisper_transcribed.csv --exp_name math_speech_bench

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0, help="cuda number, if CUDA_VISIBLE_DEVICES wasn't used")
    parser.add_argument('--test_file_csv', type=str, help="path to test csv file")
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--compute_text_only', action='store_true', default=False)
    parser.add_argument('--compute_formulas_only', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--select_dataset_items', type=int, default=0)
    parser.add_argument('--ckpt', type=str, help="path to checkpoint")
    args = parser.parse_args()

    DEVICE = 'cuda:' + str(args.cuda)
    path_to_to_ckpts = args.ckpt
    exp_name = args.exp_name

    compute_text_only = args.compute_text_only
    compute_formulas_only = args.compute_formulas_only

    torch.set_default_dtype(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(path_to_to_ckpts, 'tokenizer'))
    model = AutoModelForCausalLM.from_pretrained(os.path.join(path_to_to_ckpts, 'tuned-model'), attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16).to(DEVICE)

    # model = PeftModel.from_pretrained(
    #     model,
    #     os.path.join(path_to_to_ckpts, 'tuned-model')
    # )
    model.eval()
    # model = torch.compile(model, mode='reduce-overhead')

    outputs = defaultdict(list)
    df = pd.read_csv(args.test_file_csv)

    if args.select_dataset_items is not None and args.select_dataset_items > 0:
        df = df.head(args.select_dataset_items).tail(1)

    np.random.seed(42)
    df = df.fillna({"pron": "", "latex":""})

    # pron_column_name = 'transcription'
    # pron_column_name = 'pronunciation'
    # latex_column_name = 'sentence'
    if 'mathspeech' in args.test_file_csv.lower():
        # pron_column_name = 'whisper_text'
        pron_column_name = 'transcription'
        latex_column_name = 'latex'
    else:
        pron_column_name = 'whisper_transcription'
        latex_column_name = 'latex_with_dollars'

    val_dataset = ASRDataset(df, pron_column_name=pron_column_name, latex_column_name=latex_column_name)

    # formulas normalization will be performed in batched_model_generation
    collate_function = get_collate_function(tokenizer, path_to_to_ckpts, process_formulas=None)

    batch_size = args.batch_size
    val_loader = get_dataloader(val_dataset, batch_size, collate_function, num_workers=0, train=False)

    for batch in tqdm(val_loader):

        generated_latex = batched_model_generation(model, tokenizer, batch, device=DEVICE)

        predicted_text = generated_latex['predicted_text']
        target_text = generated_latex['target_text']

        outputs['latex_pred'].extend(predicted_text)
        outputs['latex_true'].extend(target_text)

    result_file_name = 'predictions_result_{exp_name}_{i}.csv'
    num_tries = 1000
    for i in range(num_tries):
        file_name = result_file_name.format(exp_name=exp_name, i=i)
        if not os.path.exists(file_name):
            result_file_name = file_name
            break

        if i == num_tries:
            print("Failed to save evaluation results")
            result_file_name = None


    if result_file_name is not None:
        res_df = pd.DataFrame(outputs)
        file_path = os.path.join(path_to_to_ckpts, result_file_name)
        res_df.to_csv(file_path, index=False)
        print("Saved results to ", file_path)

    in_context_metrics = LatexInContextMetrics()
    metrics_values = in_context_metrics.compute_all(outputs['latex_pred'], outputs['latex_true'], compute_text_only=compute_text_only, compute_formulas_only=compute_formulas_only)

    in_context_metrics.dump(metrics_values)

    breakpoint()

    from s2l.normalization import normalize_in_context_formulas_bulk
    pred_norm = normalize_in_context_formulas_bulk(outputs['latex_pred'])
    latex_true_norm = normalize_in_context_formulas_bulk(outputs['latex_true'])

    for x, y, lp, lt in zip(pred_norm, latex_true_norm, outputs['latex_pred'], outputs['latex_true']):
        if x != lp:
            print("pred      ", x)
            print("pred orig ", lp)
        if y != lt:
            print("latex     ", y)
            print("latex orig", lt)

    metrics_values = in_context_metrics.compute_all(pred_norm, latex_true_norm, compute_text_only=compute_text_only, compute_formulas_only=compute_formulas_only)
    in_context_metrics.dump(metrics_values)

    breakpoint()