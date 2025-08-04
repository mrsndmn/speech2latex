from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset

from trl import SFTTrainer, SFTConfig

from gemma_utils import (
    format_intersection_data,
    get_collate_fn,
    HF_MODEL_ID,
    HF_CACHE_DIR,
    MAX_WORKERS,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    OUTPUT_DIR,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LR,
    NUM_EPOCHS,
)

torch.set_float32_matmul_precision('high')

model = Gemma3nForConditionalGeneration.from_pretrained(
    HF_MODEL_ID, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    cache_dir=HF_CACHE_DIR,
)

processor = AutoProcessor.from_pretrained(
    HF_MODEL_ID, 
    trust_remote_code=True,
    cache_dir=HF_CACHE_DIR,
)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",

        # Audio layers
        "post", "linear_start", "linear_end",
        "embedding_projection",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_rslora=False,
    use_dora=False,
)
model = get_peft_model(model, peft_config)

dataset = load_dataset("csv", data_files="train.csv")
dataset = dataset.filter(lambda example: example["language"] == "eng", num_proc=MAX_WORKERS)
dataset = dataset.map(format_intersection_data, batched=True, batch_size=BATCH_SIZE, num_proc=MAX_WORKERS)
dataset = dataset['train'].train_test_split(test_size=0.1)
collate_fn = get_collate_fn(processor)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=processor.tokenizer,
    data_collator=collate_fn,
    args = SFTConfig(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        warmup_ratio = 0.1,
        learning_rate = LR,
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",

        num_train_epochs = NUM_EPOCHS,
        logging_steps = 250,
        save_steps=250,
        save_total_limit=3,
        save_strategy="steps",

        seed = 42,
        output_dir = OUTPUT_DIR,
        report_to = "none",

        # items for audio fine-tuning
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 2,
        ddp_find_unused_parameters=False,

        # PeftCausalLM warning fix
        label_names=["labels"]
    )
)

trainer.train()

# save adapters
model.save_pretrained("gemma-3n")
processor.save_pretrained("gemma-3n")
