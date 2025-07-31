import os
import pandas as pd
from collections import defaultdict
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets

from s2l.eval import LatexInContextMetrics

from dataset import ASRDataset, get_collate_function, get_dataloader
from test_qwen import batched_model_generation
from tqdm.auto import tqdm

import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default='mrsndmn/MathSpeech_whisper_transcribed', choices=['mrsndmn/MathSpeech_whisper_transcribed', 'mrsndmn/MathSpeech_whisper_transcribed_normalized'])

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model + '/tokenizer')
    model = AutoModelForCausalLM.from_pretrained(args.model + '/tuned-model')
    model.to(DEVICE)

    val_dataset = datasets.load_dataset(args.dataset, split='train')

    # latex_column = 'LaTeX'
    latex_column = 'latex_normalized'
    assert args.dataset == 'mrsndmn/MathSpeech_whisper_transcribed_normalized'
    # if args.dataset == 'mrsndmn/MathSpeech_whisper_transcribed_normalized':
    #     latex_column = 'latex_normalized'

    val_dataset = val_dataset.map(lambda x: { latex_column: x[latex_column].removesuffix('$').removeprefix('$') })

    batch_size = args.batch_size
    collate_function = get_collate_function(tokenizer, process_formulas=None, latex_column=latex_column, whisper_column='whisper_text')
    val_loader = get_dataloader(val_dataset, batch_size, collate_function, num_workers=0, train=False)

    outputs = defaultdict(list)

    for batch in tqdm(val_loader):

        generated_latex = batched_model_generation(model, tokenizer, batch, device=DEVICE)

        predicted_text = generated_latex['predicted_text']
        target_text = generated_latex['target_text']

        outputs['latex_pred'].extend(predicted_text)
        outputs['latex_true'].extend(target_text)

    pd.DataFrame(outputs).to_csv(os.path.join(args.model, f'mathspeech_generations_{latex_column}.csv'))


    metrics = LatexInContextMetrics()
    metrics_values = metrics.compute_all(outputs['latex_pred'], outputs['latex_true'], compute_text_only=False, compute_formulas_only=False)
    print("\n\nMetrics")
    metrics.dump(metrics_values)

    output_file = os.path.join(args.model, f'mathspeech_metrics_{latex_column}.json')
    with open(output_file, 'w') as f:
        json.dump(metrics_values, f)


