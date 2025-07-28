import datasets
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from io import BytesIO
from urllib.request import urlopen
import librosa
from ASRPostCorrection.chat_template_with_generation import CHAT_TEMPLATE_WITH_GENERATION

def test_dataset_tokenizer():

    s2l_dataset = datasets.Dataset.load_from_disk("/workspace-SR004.nfs2/d.tarasov/rsi-speech2latex/Data/trainable_split/equations_dev_new/")
    # normalized_formulas = pd.read_csv("/workspace-SR004.nfs2/d.tarasov/rsi-speech2latex/Data/trainable_split/normalized_formulas.csv")

    s2l_dataset_item1 = s2l_dataset[0]
    s2l_dataset_item2 = s2l_dataset[1]

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

    conversation1 = [
        {"role": "system", "content": "You are a helpful assistant. Transcribe latex formula."},
        {"role": "user", "content": [
            { "type": "audio", "audio": s2l_dataset_item1['audio_path']['array'] },
        ]},
        {"role": "assistant", "content": s2l_dataset_item1['sentence']},
    ]

    conversation2 = [
        {"role": "system", "content": "You are a helpful assistant. Transcribe latex formula."},
        {"role": "user", "content": [
            { "type": "audio", "audio": s2l_dataset_item2['audio_path']['array'] },
        ]},
        {"role": "assistant", "content": s2l_dataset_item2['sentence']},
    ]

    conversations_for_tokenization = [ conversation1, conversation2 ]

    chat_template_no_system_custom_no_role = (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
                # "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
        )

    text = processor.apply_chat_template(
        conversations_for_tokenization,
        tokenize=False,
        add_generation_prompt=False,
        # return_assistant_mask=True,
    )

    labels_text = processor.apply_chat_template(
        [ [ x[-1] ] for x in conversations_for_tokenization ],
        tokenize=False,
        chat_template=chat_template_no_system_custom_no_role,
        add_generation_prompt=False,
    )

    labels_tensor = processor(text=labels_text, return_tensors="pt", padding=True)

    audios = [ s2l_dataset_item1['audio_path']['array'], s2l_dataset_item2['audio_path']['array'] ]

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs['labels'] = inputs['input_ids'].clone()

    labels_seq_len = labels_tensor['input_ids'].shape[1]

    inputs['labels'][:, :-labels_seq_len] = -100
    inputs['labels'][:, -labels_seq_len:][ labels_tensor['input_ids'] == processor.tokenizer.pad_token_id ] = -100

    assert (inputs['labels'] == -100).sum() == -100, 'labels count is ok'

    labels_to_decode = inputs['labels'].clone()
    labels_to_decode[ labels_to_decode < 0 ] = 0
    print(processor.tokenizer.batch_decode(labels_to_decode))

    inputs = inputs.to("cuda")

    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = model.to("cuda")
    outputs = model(**inputs)

    print("outputs.loss", outputs.loss)

    assert outputs.loss < 3.3, 'loss is too high'

    breakpoint()

if __name__ == "__main__":
    test_dataset_tokenizer()



