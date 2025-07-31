import torch
from torch.utils.data import DataLoader, Dataset

from typing import Callable, List

import random

from chat_template_with_generation import CHAT_TEMPLATE_WITH_GENERATION
from process_formula import NormalizeFormula


class ASRDataset(Dataset):
    def __init__(self, dataset, pron_column_name='pron', latex_column_name='latex'):
        self.dataset = dataset

        self.pron_column_name = pron_column_name
        self.latex_column_name = latex_column_name

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "pron": item[self.pron_column_name],
            "latex": item[self.latex_column_name],
        }

def get_collate_function(tokenizer, process_formulas=None, latex_column='latex', whisper_column='pron'):

    user_instructions_prefix = [
        # 'Translate  transcribation to LaTex formula: '
        'Please, give me LaTeX representation of the following formula. Formula pronunciation: '
    ]
    
    def formulas_preprocessor(formulas_list):
        if process_formulas is not None:
            return process_formulas(formulas_list)
        return formulas_list
    
    def collate_fnc(dataset_items):
        
        all_chats = []
        all_chats_no_assistant_answer = []
        
        latex_processed = formulas_preprocessor([ item[latex_column] for item in dataset_items ])
        
        for i, dataset_item in enumerate(dataset_items):
            pronunciation = dataset_item[whisper_column]
            latex = latex_processed[i]

            user_instruction_prefix = random.choice(user_instructions_prefix)

            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{user_instruction_prefix}{pronunciation}"},
                {"role": "assistant", "content": f"{latex}"},
            ]
            
            all_chats.append(chat)
            all_chats_no_assistant_answer.append(chat[:2])

        output = tokenizer.apply_chat_template(
            all_chats,
            padding=True,
            tokenize=True,
            chat_template=CHAT_TEMPLATE_WITH_GENERATION,
            return_assistant_tokens_mask=True,
            return_dict=True,
            return_tensors='pt',
        )
        
        no_assistant_answer_output = tokenizer.apply_chat_template(
            all_chats_no_assistant_answer,
            padding=True,
            tokenize=True,
            chat_template=CHAT_TEMPLATE_WITH_GENERATION,
            return_assistant_tokens_mask=True,
            return_dict=True,
            return_tensors='pt',
            add_generation_prompt=True,
        )

        output['assistant_masks'] = torch.tensor(output['assistant_masks'])
        no_assistant_answer_output['assistant_masks'] = torch.tensor(no_assistant_answer_output['assistant_masks'])
        
        # merge generation dict with prefix
        for k, v in no_assistant_answer_output.items():
            output[f'generation_{k}'] = v

        return output

    return collate_fnc

def get_dataloader(dataset: Dataset,
                   batch_size: int,
                   collate_fn: Callable,
                   num_workers: int,
                   train: bool = False) -> DataLoader:
    
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size, 
                      collate_fn=collate_fn, 
                      num_workers=num_workers, 
                      shuffle=train, 
                      drop_last=train,
                      pin_memory=True,)
