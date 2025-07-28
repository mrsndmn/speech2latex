
class DataCollatorForQwen2Audio():

    def __init__(self, processor, sampling_rate, latex_column_name):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.latex_column_name = latex_column_name

        self.chat_template_no_system_custom_no_role = (
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


    def __call__(self, items):

        conversations = []
        audios = []

        for item in items:
            audio = item['audio_path']['array']
            audios.append(audio)

            conversation = [
                {"role": "system", "content": "You are a helpful assistant. Transcribe latex formula."},
                {"role": "user", "content": [
                    { "type": "audio", "audio": audio },
                ]},
                {"role": "assistant", "content": item[self.latex_column_name]},
            ]

            conversations.append(conversation)

        text = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )

        labels_text = self.processor.apply_chat_template(
            [ [ x[-1] ] for x in conversations ],
            tokenize=False,
            chat_template=self.chat_template_no_system_custom_no_role,
            add_generation_prompt=False,
        )

        labels_tensor = self.processor(text=labels_text, return_tensors="pt", padding=True)

        model_inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=self.sampling_rate)
        model_inputs['labels'] = model_inputs['input_ids'].clone()

        labels_seq_len = labels_tensor['input_ids'].shape[1]

        model_inputs['labels'][:, :-labels_seq_len] = -100
        model_inputs['labels'][:, -labels_seq_len:][ labels_tensor['input_ids'] == self.processor.tokenizer.pad_token_id ] = -100

        return model_inputs


