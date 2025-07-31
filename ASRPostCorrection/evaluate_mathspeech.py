from collections import defaultdict
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets

from s2l.eval import LatexInContextMetrics

from dataset import ASRDataset, get_collate_function, get_dataloader
from test_qwen import batched_model_generation
from tqdm.auto import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    collate_function = get_collate_function(tokenizer, process_formulas=None)

    val_dataset = datasets.load_dataset('mrsndmn/MathSpeech_whisper_transcribed')

    batch_size = args.batch_size
    val_loader = get_dataloader(val_dataset, batch_size, collate_function, num_workers=0, train=False)

    outputs = defaultdict(list)

    for batch in tqdm(val_loader):

        generated_latex = batched_model_generation(model, tokenizer, batch, device=DEVICE)

        predicted_text = generated_latex['predicted_text']
        target_text = generated_latex['target_text']

        outputs['latex_pred'].extend(predicted_text)
        outputs['latex_true'].extend(target_text)


    metrics = LatexInContextMetrics()

