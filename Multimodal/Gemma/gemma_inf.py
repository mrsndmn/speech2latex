from datasets import load_dataset
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel
import torch
import soundfile as sf
from tqdm import tqdm
import pandas as pd

from gemma_utils import (
    HF_MODEL_ID,
    HF_CACHE_DIR,
    MAX_WORKERS
)

dataset = load_dataset("csv", data_files="test.csv", split="train")
dataset = dataset.filter(lambda example: example["language"] == "eng", num_proc=MAX_WORKERS)

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

# merge LoRA adapters for faster inference
model = model.merge_and_unload()
model = torch.compile(model, mode="reduce-overhead")

answers = []

for data in tqdm(dataset):
    audio, _ = sf.read(data["audio_path"])

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an assistant that transcribes speech with formulas accurately into latex code.\n\nExamples:\nSpeech: x plus y squared equal z.\nYour answer: x + y^{2} = z\n\nSpeech: e to the power of a equals b over two.\nYour answer: e^{a} = \\frac{b}{2}",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
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
    ).to(model.device, model.dtype)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 256,
            temperature = 1.0, top_p = 0.95, top_k = 64,
            use_cache=False
        )

    text = processor.batch_decode(
        outputs,
        skip_special_tokens=True,
        skip_prompt=True,
        clean_up_tokenization_spaces=False
    )[0][286:] # skip system prompt

    answers.append(text.strip())

df = pd.read_csv("test.csv")
df = df[df["language"] == "eng"]
df["preds"] = answers
df.to_csv("output_eng.csv", index=False)
