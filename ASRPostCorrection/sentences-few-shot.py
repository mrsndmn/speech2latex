from copy import deepcopy
from tqdm import tqdm
from datasets import load_dataset
import torch
import argparse
from s2l.eval import LatexInContextMetrics
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument('--data_type', required=True, choices=['human', 'synthetic_small', 'mix'])
    parser.add_argument('--n_few_shot', type=int, default=5)
    parser.add_argument('--test_few_samples', type=int, default=None)

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    system_prompt = """You are a mathematics text processing assistant. Your task is to convert informal mathematical expressions into proper LaTeX format while keeping all other text unchanged. Maintain the original structure and wording, only modifying mathematical notation. Respond only with the processed text, no additional commentary."""

    dataset_test = load_dataset("marsianin500/Speech2Latex", split='sentences_test')
    dataset_train = load_dataset("marsianin500/Speech2Latex", split='sentences_train')

    dataset_test = dataset_test.remove_columns(set(dataset_test.column_names) - {'whisper_text', 'sentence_normalized', 'is_tts'})
    dataset_train = dataset_train.remove_columns(set(dataset_train.column_names) - {'whisper_text', 'sentence_normalized', 'is_tts'})

    dataset_train = dataset_train.shuffle(seed=42).select(range(1000))

    if args.data_type == 'human':
        dataset_train = dataset_train.filter(lambda x: x['is_tts'] == 0)
    elif args.data_type == 'synthetic_small':
        dataset_train = dataset_train.filter(lambda x: x['is_tts'] == 1)
    else:
        pass

    assert len(dataset_train) > args.n_few_shot

    few_shot_examples = dataset_train.select(range(args.n_few_shot))

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for example in few_shot_examples:
        messages.append({"role": "user", "content": f"{example['whisper_text']}"})
        messages.append({"role": "assistant", "content": f"{example['sentence_normalized']}"})

    generations = []
    references = []

    if args.test_few_samples is not None:
        dataset_test = dataset_test.select(range(args.test_few_samples))

    for example in tqdm(dataset_test):
        messages_copy = deepcopy(messages)
        messages_copy.append({"role": "user", "content": f"{example['whisper_text']}"})

        tokenized_chat = tokenizer.apply_chat_template(
            messages_copy,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        generated_ids = model.generate(
            tokenized_chat,
            do_sample = False,
            max_new_tokens=len(example['whisper_text']),
            # do_sample=True,
            # temperature=0.3,
            # top_p=0.9,
            # pad_token_id=tokenizer.eos_token_id
        )

        generations.append(tokenizer.batch_decode(
            generated_ids[:, tokenized_chat.shape[-1]:],
            skip_special_tokens=True,
            return_full_text=False
        )[0])

        references.append(example['sentence_normalized'])


    model_slug = args.model.split('/')[-1]
    few_shot_exps_dir = f'ckpts_few_shot/{model_slug}_{args.data_type}_n_few_shot_{args.n_few_shot}'
    os.makedirs(few_shot_exps_dir, exist_ok=True)

    df = pd.DataFrame({'generations': generations, 'references': references, 'is_tts': dataset_test['is_tts']})
    df.to_csv(f'{few_shot_exps_dir}/generations_references.csv', index=False)

    df_tts = df[df['is_tts'] == 1]
    df_human = df[df['is_tts'] == 0]

    in_context_metrics = LatexInContextMetrics()

    metrics_tts = in_context_metrics.compute_all(df_tts['generations'], df_tts['references'])
    metrics_human = in_context_metrics.compute_all(df_human['generations'], df_human['references'])

    with open(f'{few_shot_exps_dir}/metrics_tts.json', 'w') as f:
        json.dump(metrics_tts, f)
    print(f"Results for TTS saved to {few_shot_exps_dir}/metrics_tts.json")

    with open(f'{few_shot_exps_dir}/metrics_human.json', 'w') as f:
        json.dump(metrics_human, f)
    print(f"Results for human saved to {few_shot_exps_dir}/metrics_human.json")

    print("Metrics for TTS")
    in_context_metrics.dump(metrics_tts)
    print("Metrics for human")
    in_context_metrics.dump(metrics_human)