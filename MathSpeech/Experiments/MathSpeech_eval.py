import argparse
import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

class MathASR(torch.nn.Module):
    """Wrapper that holds the two-stage T5 pipeline (error correction → LaTeX translation)."""

    def __init__(self, tokenizer, model1, model2, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        self.device = device

    @torch.no_grad()
    def translate_batch(
        self,
        asr_beam1,
        asr_beam2,
        max_length_input: int,
        max_length_correct: int,
        max_length_output: int,
        num_beams: int = 5,
    ):
        """Translate a batch of ASR hypotheses into LaTeX.

        Args:
            asr_beam1 (List[str]): first ASR hypothesis per sample.
            asr_beam2 (List[str]): second ASR hypothesis per sample.
        Returns:
            List[str]: LaTeX strings corresponding to each input sample.
        """
        assert len(asr_beam1) == len(asr_beam2)

        # 1. Build encoder input strings
        inputs_text = [
            f"translate ASR to truth: {b1} || {b2}"
            for b1, b2 in zip(asr_beam1, asr_beam2)
        ]

        # 2. Tokenise with *left* padding
        batch_enc = self.tokenizer(
            inputs_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length_input,
        ).to(self.device)

        # 3. Run error-corrector model
        corrected_ids = self.model1.generate(
            **batch_enc,
            max_length=max_length_correct,
            num_beams=num_beams,
            early_stopping=True,
        )

        corrected_sentences = [
            self.tokenizer.decode(ids[1:-1], skip_special_tokens=False).strip()
            for ids in corrected_ids
        ]

        corrected_sentences = [c.replace("<pad>", "").replace("</s>", "") for c in corrected_sentences]

        # 4. Tokenise corrected sentences
        batch_enc2 = self.tokenizer(
            corrected_sentences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length_correct,
        ).to(self.device)

        # 5. Run translation model
        latex_ids = self.model2.generate(
            **batch_enc2,
            max_length=max_length_output,
            num_beams=num_beams,
            early_stopping=True,
        )

        latex_outputs = [
            self.tokenizer.decode(ids[1:-1], skip_special_tokens=False).replace(" ", "")
            for ids in latex_ids
        ]
        latex_outputs = [c.replace("<pad>", "").replace("</s>", "") for c in latex_outputs]

        return latex_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MathSpeech evaluation script")
    parser.add_argument("--input_csv", default="./result_ASR.csv", help="Path to ASR result CSV")
    parser.add_argument(
        "--output_csv",
        default="MathSpeech_LaTeX_result.csv",
        help="Where to store LaTeX predictions",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="./MathSpeech_checkpoint.pth",
        help="Checkpoint with trained MathASR weights",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="AAAI2025/MathSpeech_Ablation_Study_LaTeX_translator_T5_small",
        help="HuggingFace model path or name for the tokenizer",
    )
    parser.add_argument(
        "--t5_backbone",
        default="google-t5/t5-small",
        help="HuggingFace model name for both T5 stages",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length_input", type=int, default=540)
    parser.add_argument("--max_length_correct", type=int, default=275)
    parser.add_argument("--max_length_output", type=int, default=275)
    parser.add_argument("--num_beams", type=int, default=5)
    return parser


def load_model_and_tokenizer(args):
    # Initialise tokenizer with *left* padding
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = "left"

    # Two identical T5 backbones (could be swapped for distinct checkpoints if needed)
    model_corrector = T5ForConditionalGeneration.from_pretrained(args.t5_backbone)
    model_corrector.resize_token_embeddings(len(tokenizer))

    model_translator = T5ForConditionalGeneration.from_pretrained(args.t5_backbone)
    model_translator.resize_token_embeddings(len(tokenizer))

    model = MathASR(tokenizer, model_corrector, model_translator, args.device)

    # Optional fine-tuned weights
    if os.path.isfile(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    else:
        raise ValueError(f"[WARNING] Checkpoint '{args.checkpoint_path}' not found – using base models.")

    model.eval()
    return model


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    beam1 = df["whisper_base_predSE"].tolist()
    beam2 = df["whisper_small_predSE"].tolist()

    model = load_model_and_tokenizer(args)

    model = torch.compile(model)

    predictions = []
    for start in tqdm(range(0, len(beam1), args.batch_size)):
        end = min(start + args.batch_size, len(beam1))
        batch_preds = model.translate_batch(
            beam1[start:end],
            beam2[start:end],
            max_length_input=args.max_length_input,
            max_length_correct=args.max_length_correct,
            max_length_output=args.max_length_output,
            num_beams=args.num_beams,
        )
        predictions.extend(batch_preds)

    df["MathSpeech_LaTeX_result"] = predictions
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to '{args.output_csv}'")


if __name__ == "__main__":
    main()