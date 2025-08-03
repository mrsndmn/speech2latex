from typing import Final
from transformers import TextStreamer

HF_MODEL_ID: Final = "google/gemma-3n-E2B-it"
HF_DATASET_ID: Final = "marsianin500/Speech2Latex"
HF_CACHE_DIR: Final = "./cache"
DATASET_SPLIT: Final = "equations_train"
MAX_WORKERS: Final = 8
LORA_R: Final = 16
LORA_ALPHA: Final = 32
LORA_DROPOUT: Final = 0.3
OUTPUT_DIR: Final = "./outputs/gemma-3n-E2B-it-trl-sft"
BATCH_SIZE: Final = 8
GRADIENT_ACCUMULATION_STEPS: Final = 8
LR: Final = 4e-05
NUM_EPOCHS: Final = 2

def format_intersection_data(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    for idx in range(len(samples["audio_path"])):
        audio = samples["audio_path"][idx]["array"]
        label = str(samples["sentence_normalized"][idx])

        message = [
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
            },
            {
                "role": "assistant",
                "content":[{"type": "text", "text": label}]
            }
        ]
        formatted_samples["messages"].append(message)
    return formatted_samples

def get_collate_fn(processor):
    def collate_fn(examples):
        texts = []
        audios = []

        for example in examples:
            # Apply chat template to get text
            text = processor.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            ).strip()
            texts.append(text)

            # Extract audios
            audios.append(example["audio_path"]["array"])

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, audio=audios, return_tensors="pt", padding=True
        )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Use Gemma3n specific token masking
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if hasattr(processor.tokenizer, 'image_token_id'):
            labels[labels == processor.tokenizer.image_token_id] = -100
        if hasattr(processor.tokenizer, 'audio_token_id'):
            labels[labels == processor.tokenizer.audio_token_id] = -100
        if hasattr(processor.tokenizer, 'boi_token_id'):
            labels[labels == processor.tokenizer.boi_token_id] = -100
        if hasattr(processor.tokenizer, 'eoi_token_id'):
            labels[labels == processor.tokenizer.eoi_token_id] = -100


        batch["labels"] = labels
        return batch
    return get_collate_fn

# Helper function for inference
def do_gemma_3n_inference(model, processor, messages, max_new_tokens = 128):
    _ = model.generate(
        **processor.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        do_sample=False,
        streamer = TextStreamer(processor, skip_prompt = True),
    )
