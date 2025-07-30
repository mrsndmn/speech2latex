from datasets import load_dataset, Audio
from transformers import AutoProcessor, Gemma3nForConditionalGeneration, TextStreamer
from peft import PeftModel
import torch

import random

from gemma_utils import (
    HF_MODEL_ID,
    HF_DATASET_ID,
    HF_CACHE_DIR,
    DATASET_SPLIT,
    MAX_WORKERS,
)

dataset = load_dataset(
    HF_DATASET_ID,
    split=DATASET_SPLIT + "[:10]",
    cache_dir=HF_CACHE_DIR,
    num_proc=MAX_WORKERS,
)
dataset = dataset.filter(lambda example: example["language"] == "eng", num_proc=MAX_WORKERS)
dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16_000))
test_audio = random.choice(dataset)

model = Gemma3nForConditionalGeneration.from_pretrained(
    HF_MODEL_ID, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    cache_dir=HF_CACHE_DIR,
).to("cuda")

processor = AutoProcessor.from_pretrained(
    HF_MODEL_ID, 
    trust_remote_code=True,
    cache_dir=HF_CACHE_DIR,
)

model = PeftModel.from_pretrained(model, "gemma-3n")
model.eval()

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an assistant that transcribes speech accurately.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": test_audio['audio_path']['array']},
            {"type": "text", "text": "Please transcribe this audio."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to(model.device, torch.bfloat16)

_ = model.generate(
    **inputs,
    max_new_tokens = 256,
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(processor, skip_prompt = True),
    use_cache=False
)
print(test_audio)