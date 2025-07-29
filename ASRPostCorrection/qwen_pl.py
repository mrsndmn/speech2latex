import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
# from peft import LoraConfig, get_peft_model, TaskType
from pytorch_lightning.loggers import CSVLogger
from dataset import ASRDataset, get_collate_function, get_dataloader

import pandas as pd


from test_qwen import batched_model_generation

class Config:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, Config(**v))
            else:
                setattr(self, k, v)

    def __str__(self):
        return '\n'.join(f'{key}: {value}' for key, value in self.__dict__.items())

    def __repr__(self):
        return self.__str__()

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)


class Model_pl(pl.LightningModule):
    def __init__(self, cfg: dict[str, int|float|str], n_iters: int, model: nn.Module, tokenizer) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.n_embeddings = model.model.embed_tokens.weight.shape[0]
        # peft_cfg = LoraConfig(
        #     r=64,
        #     lora_alpha=128,
        #     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        #     lora_dropout=0,
        #     bias="none",
        #     task_type=TaskType.CAUSAL_LM
        # )
        # self.model = get_peft_model(model, peft_cfg)
        self.loss_fct = nn.CrossEntropyLoss()
        self.n_iters = n_iters
        self.save_hyperparameters('cfg')
        self.tokenizer = tokenizer

        self.train_wer = torchmetrics.text.WordErrorRate()
        self.train_cer = torchmetrics.text.CharErrorRate()

        self.validation_wer = torchmetrics.text.WordErrorRate()
        self.validation_cer = torchmetrics.text.CharErrorRate()


    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:

        optimizer = torch.optim.AdamW(list(self.model.parameters()), lr=self.cfg.learning_rate, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=20000)
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }}

    def on_train_epoch_end(self) -> None:

        self.model.save_pretrained(f"{self.logger.save_dir}/tuned-model")

        tokenizer_path = f"{self.logger.save_dir}/tokenizer"
        if not os.path.exists(tokenizer_path):
            self.tokenizer.save_pretrained(tokenizer_path)

        return

    def compute_loss_logits_labels(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask'].bool()
        assistant_masks = batch['assistant_masks'].bool()

        logits = self.model(input_ids, attention_mask=attention_mask).get("logits")
        # Optimize only assistant answers
        # assistant_logits = logits[]

        batch_size = input_ids.shape[0]

        # Create labels
        labels = input_ids.clone()
        labels[~attention_mask] = -100 # ignore padding
        labels[~assistant_masks] = -100 # ignore system prompt and user request

        # Shift tokens
        logits = logits[:, :-1]
        labels = labels[:, 1:]

        loss = self.loss_fct(logits.flatten(0, 1), labels.flatten())
        labels = labels.clone()

        return loss, logits, labels

    def decode_for_metric_compute(self, logits: torch.Tensor, labels: torch.Tensor):

        predicted_ids = torch.argmax(logits, dim=-1)

        predicted_ids[labels == -100] = self.tokenizer.pad_token_id
        labels[labels == -100] = self.tokenizer.pad_token_id

        predicted_text = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        target_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        return predicted_text, target_text

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:

        loss, logits, labels = self.compute_loss_logits_labels(batch)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        predicted_text, target_text = self.decode_for_metric_compute(logits, labels)

        self.train_wer.update(predicted_text, target_text)
        self.train_cer.update(predicted_text, target_text)

        self.log("train_wer", self.train_wer.compute().item(), prog_bar=True)
        self.log("train_cer", self.train_cer.compute().item(), prog_bar=True)

        self.train_wer.reset()
        self.train_cer.reset()

        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> None:
        loss, logits, labels = self.compute_loss_logits_labels(batch)

        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        device = batch['input_ids'].device
        generated_latex = batched_model_generation(self.model, self.tokenizer, batch, device=device)

        predicted_text = generated_latex['predicted_text']
        target_text = generated_latex['target_text']

        self.validation_wer.update(predicted_text, target_text)
        self.validation_cer.update(predicted_text, target_text)

        return

    def on_validation_epoch_end(self):
        self.log("val_wer", self.validation_wer.compute().item())
        self.log("val_cer", self.validation_cer.compute().item())

        self.validation_wer.reset()
        self.validation_cer.reset()

