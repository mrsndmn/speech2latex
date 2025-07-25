import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
from tqdm.auto import tqdm
import evaluate

import numpy as np
import pandas as pd

import os
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help="cuda number, if CUDA_VISIBLE_DEVICES wasn't used")
parser.add_argument('--lang', type=str, help="ru or eng")
parser.add_argument('--ckpt', type=str, help="path to checkpoint")
args = parser.parse_args()

DEVICE = 'cuda:' + str(args.cuda)
path_to_to_ckpts = args.ckpt

tokenizer = AutoTokenizer.from_pretrained(os.path.join(path_to_to_ckpts, 'tokenizer'))
model = AutoModelForCausalLM.from_pretrained(os.path.join(path_to_to_ckpts, 'tuned-model')).to(DEVICE)
# model = PeftModel.from_pretrained(
#     model,
#     os.path.join(path_to_to_ckpts, 'tuned-model')
# )
model.eval()
model = torch.compile(model, mode='reduce-overhead')

gen_params = {
    "do_sample": False,
    "max_new_tokens": 200,
    "min_new_tokens": 1,
    "early_stopping": True,
    "num_beams": 3,
    "repetition_penalty": 1.0,
    "remove_invalid_values": True,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.eos_token_id,
    "forced_eos_token_id": tokenizer.eos_token_id,
    "use_cache": True,
    "no_repeat_ngram_size": 4,
    "num_return_sequences": 1,
}

wer = evaluate.load('wer')
cer = evaluate.load('cer')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
sacrebleu = evaluate.load("sacrebleu")
rouge1 = evaluate.load("rouge")
chrf = evaluate.load("chrf")


outputs = defaultdict(list)
df = pd.read_csv(f"test_{args.lang}.csv")
# df["pron"] = df["whisper_transcription"].apply(lambda s: s.strip())
np.random.seed(42)
df = df.sample(frac=1)
df = df.iloc[:int(df.shape[0] * 0.1)]
df = df.fillna({"pron": " ", "latex": "$"})
for i in tqdm(range(df.shape[0])):
    pron, latex = df.iloc[i]['pron'], str(df.iloc[i]['latex'])
    pron_tokens = tokenizer.encode(pron, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        out = model.generate(inputs=pron_tokens, **gen_params)
    
    
    generated_texts = tokenizer.batch_decode(out, skip_special_tokens=True)[0].replace(pron, "")
    outputs['latex_pred'].append(generated_texts)
    outputs['latex_true'].append(latex)
    outputs['pron'].append(pron)

    outputs['cer'].append(cer.compute(predictions=[generated_texts], references=[latex]))
    outputs['wer'].append(wer.compute(predictions=[generated_texts], references=[latex]))

    outputs['cer_lower'].append(cer.compute(predictions=[generated_texts.lower()], references=[latex.lower()]))
    outputs['wer_lower'].append(wer.compute(predictions=[generated_texts.lower()], references=[latex.lower()]))
    
    if generated_texts and latex:
        outputs['rouge1'].append(rouge1.compute(predictions=[generated_texts], references=[latex])['rouge1'])
        outputs['chrf'].append(chrf.compute(predictions=[generated_texts], references=[latex])['score'] / 100)
        outputs['chrfpp'].append(chrf.compute(predictions=[generated_texts], references=[latex], word_order=2)['score'] / 100)
        outputs['bleu'].append(bleu.compute(predictions=[generated_texts], references=[latex])['bleu'])
        outputs['sbleu'].append(sacrebleu.compute(predictions=[generated_texts], references=[latex])['score'] / 100)
        outputs['meteor'].append(meteor.compute(predictions=[generated_texts], references=[latex])['meteor'])

        outputs['rouge1_lower'].append(rouge1.compute(predictions=[generated_texts.lower()], references=[latex.lower()])['rouge1'])
        outputs['chrf_lower'].append(chrf.compute(predictions=[generated_texts.lower()], references=[latex.lower()])['score'] / 100)
        outputs['chrfpp_lower'].append(chrf.compute(predictions=[generated_texts.lower()], references=[latex.lower()], word_order=2)['score'] / 100)
        outputs['bleu_lower'].append(bleu.compute(predictions=[generated_texts.lower()], references=[latex.lower()])['bleu'])
        outputs['sbleu_lower'].append(sacrebleu.compute(predictions=[generated_texts.lower()], references=[latex.lower()])['score'] / 100)
        outputs['meteor_lower'].append(meteor.compute(predictions=[generated_texts.lower()], references=[latex.lower()])['meteor'])

    else:
        print(f"{pron=}, {latex=}, {generated_texts=}")
        outputs['rouge1'].append(0)
        outputs['chrf'].append(0)
        outputs['chrfpp'].append(0)
        outputs['bleu'].append(0)
        outputs['sbleu'].append(0)
        outputs['meteor'].append(0)

        outputs['rouge1_lower'].append(0)
        outputs['chrf_lower'].append(0)
        outputs['chrfpp_lower'].append(0)
        outputs['bleu_lower'].append(0)
        outputs['sbleu_lower'].append(0)
        outputs['meteor_lower'].append(0)

    if i % 100 == 0:
        print(outputs['pron'][-1], outputs['latex_pred'][-1], outputs['cer'][-1])

res_df = pd.DataFrame(outputs)
res_df.to_csv(os.path.join(path_to_to_ckpts, f'res_{args.lang}.csv'), index=False)

# metric (lower or higher is better - best value possible) = value
print(f"wer (l - 0) = {np.mean(outputs['wer']):.4f}, cer (l - 0) = {np.mean(outputs['cer']):.4f}, rouge1 (h - 1) = {np.mean(outputs['rouge1']):.4f}, chrf (h - 100) = {np.mean(outputs['chrf']):.4f}, chrf++ (h - 100) = {np.mean(outputs['chrfpp']):.4f}, bleu (h - 1) = {np.mean(outputs['bleu']):.4f}, sbleu (h - 1) = {np.mean(outputs['sbleu']):.4f}, meteor (h - 1) = {np.mean(outputs['meteor']):.4f}")
