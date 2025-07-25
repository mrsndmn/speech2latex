import torch

from transformers import AutoTokenizer
from chat_template_with_generation import CHAT_TEMPLATE_WITH_GENERATION



def test_chat_template_with_generation():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    chat = [
        {"role": "system", "content": "This is a dialog with AI assistant."},
        {"role": "user", "content": "Rewrite the text using latex formulas if needed.\\nalpha betta gamma shtrikh"},
        {"role": "assistant", "content": "\\alpha \\betta \\gamma \\shtrikh"},
    ]

    output = tokenizer.apply_chat_template(chat, tokenize=True, chat_template=CHAT_TEMPLATE_WITH_GENERATION, return_assistant_tokens_mask=True, return_dict=True, return_tensors='pt')
    output['assistant_masks'] = torch.tensor(output['assistant_masks']).unsqueeze(0)
    assert output['assistant_masks'].sum().item() > 0

    print("output")
    print(output)

    print("Assistant tokens decoded:")
    print(tokenizer.decode(output['input_ids'][output['assistant_masks'].bool()]))

    breakpoint()