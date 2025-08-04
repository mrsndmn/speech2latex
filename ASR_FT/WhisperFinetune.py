import torchaudio
import torch
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

df = pd.read_csv("train_ENG.csv")
hf_dataset = Dataset.from_pandas(df)

model_name = "openai/whisper-large-v3"
max_memory_mapping = {0: "80GB"}
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map="auto", max_memory=max_memory_mapping)

def preprocess_data(batch):
    labels = processor.tokenizer(batch['latex']).input_ids
    batch["labels"] = labels

    batch["input_features"] = batch['audio_path']
    return batch

hf_dataset = hf_dataset.map(preprocess_data, remove_columns=hf_dataset.column_names, num_proc=24)


from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for feature in features:
            audio, sr = torchaudio.load(feature['input_features'])
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16_000)
            audio = resampler(audio)
            audio = feature_extractor(audio.squeeze(0), sampling_rate=16_000).input_features[0]

            input_features.append({
                "input_features": audio,
            })

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=hf_dataset,
    eval_dataset=hf_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

trainer.train(resume_from_checkpoint='./whisper-small-hi/checkpoint-4000')
