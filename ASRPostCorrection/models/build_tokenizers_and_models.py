import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)

    os.makedirs("init_checkpoints", exist_ok=True)

    for special_token in ["{", "}", "[", "]", "_", "^", "-", "\\", '~', '<', '>']:
        result = tokenizer.add_tokens([special_token])
        if result > 0:
            print(f"added {special_token}")
        else:
            print(f"token is already in the vocab: {special_token}")

    model.resize_token_embeddings(len(tokenizer))

    random_tokens_init_checkpoint_path = "init_checkpoints/qwen2.5-0.5B-Instruct-added-tokens"
    tokenizer.save_pretrained(random_tokens_init_checkpoint_path)
    model.save_pretrained(random_tokens_init_checkpoint_path)

    print(f"saved to {random_tokens_init_checkpoint_path}")
