import random
import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

import datasets
from datasets import Audio

from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import CSVLogger
from qwen_audio_data_collator import DataCollatorForQwen2Audio

import pandas as pd


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
    def __init__(self, cfg, model, train_dataset, collate_function, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.model = model

        self.n_embeddings = model.config.vocab_size
        # self.loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.train_dataset = train_dataset
        self.collate_function = collate_function
        self.n_iters = len(self.train_dataloader())
        self.save_hyperparameters('cfg')
        self.tokenizer = tokenizer

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(list(self.model.parameters()), lr=self.cfg.learning_rate, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=300, num_training_steps=self.n_iters)
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1

        }}

    def on_train_epoch_end(self):
        # torch.save(self.model.lm_head.state_dict() ,f"ckpts/{self.cfg.exp_name}/lm_head_state_dict.pth")
        self.model.save_pretrained(f"ckpts/{self.cfg.exp_name}/tuned-model")
        self.tokenizer.save_pretrained(f"ckpts/{self.cfg.exp_name}/tokenizer")

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(**batch)

        loss = outputs.loss

        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx % self.cfg.ckp_iterations_step == 0 and self.global_rank == 0:
            os.makedirs(f"ckpts/{self.cfg.exp_name}/{batch_idx}", exist_ok=True)
            self.model.save_pretrained(f"ckpts/{self.cfg.exp_name}/{batch_idx}/tuned-model")

        return loss


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.collate_function, num_workers = self.cfg.num_workers, shuffle = True)



# Example:
# python train.py --dataset_path /workspace-SR004.nfs2/d.tarasov/rsi-speech2latex/Data/trainable_split/equations_dev_new/ --config configs/config.json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.json')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config_dict = json.load(config_file)
    cfg = Config(**config_dict)
    # df = pd.read_csv(cfg.df_path, index_col=False)dummy_ex

    s2l_dataset = datasets.DatasetDict.load_from_disk(cfg.dataset_path)
    s2l_dataset = s2l_dataset[cfg.train_dataset_split]

    print("len dataset", len(s2l_dataset))

    columns_to_leave = ['audio_path', cfg.latex_column_name]
    s2l_dataset = s2l_dataset.remove_columns(list(set(s2l_dataset.column_names) - set(columns_to_leave)))

    torch.manual_seed(1234)

    processor = AutoProcessor.from_pretrained(cfg.model_ckpt, trust_remote_code=True)

    s2l_dataset = s2l_dataset.cast_column('audio_path', Audio(sampling_rate=processor.feature_extractor.sampling_rate))

    model = Qwen2AudioForConditionalGeneration.from_pretrained(cfg.model_ckpt, device_map="cpu", trust_remote_code=True, torch_dtype=torch.bfloat16)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'out_proj', 'fc_out', 'gate_proj', 'up_proj', 'down_proj'],
        target_modules=["c_attn", "c_proj", "w1", "w2", "wte", "lm_head", "query", "key", "value", "out", "proj", "audio_bos_eos_token"],
        inference_mode=False,
        r=8,          # Размер ранга
        lora_alpha=16, # Коэффициент увеличения
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    os.makedirs(f"ckpts/{cfg.exp_name}", exist_ok=True)
    logger = CSVLogger("ckpts", name=cfg.exp_name)
    cfg.exp_name = os.path.join(cfg.exp_name, f'version_{logger.version}')

    train_dataset = s2l_dataset
    collate_function = DataCollatorForQwen2Audio(processor, sampling_rate=processor.feature_extractor.sampling_rate, latex_column_name=cfg.latex_column_name)

    module = Model_pl(cfg, model, train_dataset, collate_function, processor.tokenizer)
    trainer = pl.Trainer(max_epochs=cfg.n_epochs, logger = logger, accumulate_grad_batches = cfg.grad_accum)
    trainer.fit(module)

