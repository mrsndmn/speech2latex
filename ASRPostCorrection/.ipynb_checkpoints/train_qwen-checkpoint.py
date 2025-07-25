import argparse
import json
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
# from peft import LoraConfig, get_peft_model, TaskType
from pytorch_lightning.loggers import CSVLogger
from dataset import get_dataset, get_collate_function, get_dataloader
        
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
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.n_iters = n_iters
        self.save_hyperparameters('cfg')
        self.tokenizer = tokenizer

        self.wer = torchmetrics.text.WordErrorRate()
        self.cer = torchmetrics.text.CharErrorRate()
        
    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        
        optimizer = torch.optim.AdamW(list(self.model.parameters()), lr=self.cfg.learning_rate, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=6000)
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
            
        }}
    

    def on_train_epoch_end(self) -> None:
        self.model.save_pretrained(f"ckpts/{self.cfg.exp_name}/tuned-model")
        self.tokenizer.save_pretrained(f"ckpts/{self.cfg.exp_name}/tokenizer")

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, masks = batch
        inputs = inputs.long()
        
        logits = self.model(inputs).get("logits")[:, :-1]
        
        labels = inputs[:, 1:]
        masks = masks[:, 1:]
          
        logits = logits[masks].contiguous()
        labels = labels[masks].contiguous()

        
        loss = self.loss_fct(logits.view(-1, self.n_embeddings), labels.view(-1)).mean()
            
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
        target_text = self.tokenizer.decode(labels, skip_special_tokens=True)

        self.wer(predicted_text, target_text)
        self.cer(predicted_text, target_text)
        self.log("train_wer", self.wer, prog_bar=True)
        self.log("train_cer", self.cer, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> None:
        inputs, masks = batch
        inputs = inputs.long()
        
        logits = self.model(inputs).get("logits")[:, :-1]
        
        labels = inputs[:, 1:]
        masks = masks[:, 1:]
          
        logits = logits[masks].contiguous()
        labels = labels[masks].contiguous()

        
        loss = self.loss_fct(logits.view(-1, self.n_embeddings), labels.view(-1)).mean()
            
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
        target_text = self.tokenizer.decode(labels, skip_special_tokens=True)

        self.wer(predicted_text, target_text)
        self.cer(predicted_text, target_text)
        self.log("val_wer", self.wer, prog_bar=True)
        self.log("val_cer", self.cer, prog_bar=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config-4.json')
    parser.add_argument('--train_df', type=str)
    parser.add_argument('--val_df', type=str)
    args = parser.parse_args()
    
    with open(args.config, 'r') as config_file:
        config_dict = json.load(config_file)
        
    cfg = Config(**config_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_ckpt)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_ckpt)
    
    os.makedirs(f"ckpts/{cfg.exp_name}", exist_ok=True)
    logger = CSVLogger("ckpts", name=cfg.exp_name)
    cfg.exp_name = os.path.join(cfg.exp_name, f'version_{logger.version}')
    
    
    ### Work with data
    collate_function = get_collate_function(tokenizer)
    train_df = pd.read_csv(args.train_df)
    train_df = train_df.fillna({"pron": "", "latex":""})
    train_df = train_df.iloc[:int(train_df.shape[0]*0.7)]
    val_df = pd.read_csv(args.val_df)
    val_df = val_df.fillna({"pron": "", "latex":""})
    val_df = val_df.iloc[:int(val_df.shape[0]*0.7)]


    train_dataset = get_dataset(train_df, tokenizer)
    val_dataset = get_dataset(val_df, tokenizer)

    train_loader = get_dataloader(train_dataset, cfg.batch_size, collate_function, cfg.num_workers, True)
    val_loader = get_dataloader(val_dataset, cfg.batch_size, collate_function, cfg.num_workers, False)

    module = Model_pl(cfg, len(train_loader), model, tokenizer)
    trainer = pl.Trainer(devices=2, max_epochs=cfg.n_epochs, logger=logger, accumulate_grad_batches=cfg.grad_accum, strategy='ddp')
    trainer.fit(model=module, 
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
    )
    