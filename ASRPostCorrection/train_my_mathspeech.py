from __future__ import annotations

import argparse
import os
import random
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


MAX_LENGTH1 = 540  # input (ASR text)
MAX_LENGTH2 = 275  # corrected text (stage 1 target and stage 2 input)
MAX_LENGTH3 = 275  # LaTeX target (stage 2 target)


class MathASR(torch.nn.Module):
    def __init__(self, tokenizer, model_name1, model_name2, device):
        super(MathASR, self).__init__()
        self.tokenizer = tokenizer
        self.model1 = model_name1
        self.model1.to(device)
        self.model2 = model_name2
        self.model2.to(device)
        self.device = device

    def forward(self, input_ids, attention_mask_correct, attention_mask_translate, labels_correct, labels_translate):
        input_ids = input_ids.contiguous()
        attention_mask_correct = attention_mask_correct.contiguous()
        labels_correct = labels_correct.contiguous()
        attention_mask_translate = attention_mask_translate.contiguous()
        labels_translate = labels_translate.contiguous()

        outputs1 = self.model1(input_ids=input_ids, attention_mask=attention_mask_correct, labels=labels_correct)
        loss1 = outputs1.loss
        logits1 = outputs1.logits

        intermediate_ids = torch.argmax(logits1, dim=-1).detach()

        outputs2 = self.model2(input_ids=intermediate_ids, attention_mask=attention_mask_translate, labels=labels_translate)
        loss2 = outputs2.loss

        total_loss = 0.3 * loss1 + 0.7 * loss2
        return total_loss, outputs1.logits, outputs2.logits


class MathSpeechDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: T5Tokenizer,
        max_input_len: int = MAX_LENGTH1,
        max_correct_len: int = MAX_LENGTH2,
        max_latex_len: int = MAX_LENGTH3,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_correct_len = max_correct_len
        self.max_latex_len = max_latex_len

        df = pd.read_csv(csv_path)
        # Ensure required columns exist
        for col in ["pronunciation", "whisper_text", "sentence"]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {csv_path}")
        # Drop rows with missing values in required fields
        df = df.dropna(subset=["pronunciation", "whisper_text", "sentence"]).reset_index(drop=True)
        self.pronunciation = df["pronunciation"].astype(str).tolist()
        self.asr_text = df["whisper_text"].astype(str).tolist()
        self.latex = df["sentence"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.asr_text)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_text = self.asr_text[idx]
        corrected_text = self.pronunciation[idx]
        latex_text = self.latex[idx]

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        correct_enc = self.tokenizer(
            corrected_text,
            max_length=self.max_correct_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        latex_enc = self.tokenizer(
            latex_text,
            max_length=self.max_latex_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = input_enc.input_ids.squeeze(0)
        attention_mask_correct = input_enc.attention_mask.squeeze(0)

        labels_correct = correct_enc.input_ids.squeeze(0)
        attention_mask_translate = correct_enc.attention_mask.squeeze(0)

        labels_translate = latex_enc.input_ids.squeeze(0)

        # Replace pad token ids in labels with -100 so they are ignored by loss
        pad_id = self.tokenizer.pad_token_id
        labels_correct = labels_correct.masked_fill(labels_correct == pad_id, -100)
        labels_translate = labels_translate.masked_fill(labels_translate == pad_id, -100)

        return {
            "input_ids": input_ids.long(),
            "attention_mask_correct": attention_mask_correct.long(),
            "attention_mask_translate": attention_mask_translate.long(),
            "labels_correct": labels_correct.long(),
            "labels_translate": labels_translate.long(),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    model: MathASR,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    num_epochs: int,
    grad_clip: float | None = None,
) -> None:
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask_correct = batch["attention_mask_correct"].to(device)
            attention_mask_translate = batch["attention_mask_translate"].to(device)
            labels_correct = batch["labels_correct"].to(device)
            labels_translate = batch["labels_translate"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(
                input_ids=input_ids,
                attention_mask_correct=attention_mask_correct,
                attention_mask_translate=attention_mask_translate,
                labels_correct=labels_correct,
                labels_translate=labels_translate,
            )
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(1, len(dataloader))
        tqdm.write(f"Epoch {epoch} average loss: {avg_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MathASR (two-stage T5) with raw PyTorch")
    parser.add_argument("--csv_path", type=str, default="./result_ASR.csv", help="Path to CSV with columns: whisper_text, pronunciation, sentence")
    parser.add_argument("--tokenizer_path", type=str, default="google-t5/t5-small", help="Tokenizer name or path")
    parser.add_argument("--corrector_model", type=str, default="google-t5/t5-small", help="Stage-1 (correction) T5 model name or path")
    parser.add_argument("--translator_model", type=str, default="google-t5/t5-small", help="Stage-2 (translation) T5 model name or path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_input_len", type=int, default=MAX_LENGTH1)
    parser.add_argument("--max_correct_len", type=int, default=MAX_LENGTH2)
    parser.add_argument("--max_latex_len", type=int, default=MAX_LENGTH3)
    parser.add_argument("--output_path", type=str, default="ASRPostCorrection/mathasr_checkpoint.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)

    model_corrector = T5ForConditionalGeneration.from_pretrained(args.corrector_model)
    model_corrector.resize_token_embeddings(len(tokenizer))
    model_corrector.to(device)

    model_translator = T5ForConditionalGeneration.from_pretrained(args.translator_model)
    model_translator.resize_token_embeddings(len(tokenizer))
    model_translator.to(device)

    dataset = MathSpeechDataset(
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        max_correct_len=args.max_correct_len,
        max_latex_len=args.max_latex_len,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = MathASR(tokenizer=tokenizer, model_name1=model_corrector, model_name2=model_translator, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        grad_clip=args.grad_clip,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(
        {
            "model1_state_dict": model.model1.state_dict(),
            "model2_state_dict": model.model2.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "tokenizer_name_or_path": args.tokenizer_path,
            "args": vars(args),
        },
        args.output_path,
    )
    print(f"Saved final checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()

