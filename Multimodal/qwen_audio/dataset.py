import json
import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class ASREnhDataset(Dataset):
    def __init__(self, s2l_dataset, tokenizer):
        self.s2l_dataset = s2l_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.s2l_dataset)


    def __getitem__(self, idx):
        data = self.s2l_dataset.iloc[idx]
        audio_url =  data['audio_path']
        some_answer = data['latex']

        sp_prompt = "<|startofanalysis|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
        query = f"<audio>{audio_url}</audio>{sp_prompt}"
        audio_info = self.tokenizer.process_audio(query)

        eos_token_id = self.tokenizer('<|endoftext|>',return_tensors='pt').input_ids[0]
        query_tokens = self.tokenizer(query, return_tensors='pt',audio_info=audio_info)
        input_ids_query =      query_tokens.input_ids[0]
        token_type_ids_query = query_tokens.token_type_ids[0]
        attention_mask_query = query_tokens.attention_mask[0]
        some_answer_ids = self.tokenizer(some_answer, return_tensors='pt').input_ids[0]
        mask = torch.tensor([False] * len(input_ids_query) + [True] * (len(some_answer_ids) + 1))
        input_ids_query_with_answer =      torch.cat([input_ids_query, some_answer_ids, eos_token_id])
        token_type_ids_query_with_answer = torch.cat([token_type_ids_query, torch.zeros_like(some_answer_ids), torch.tensor([0])])
        attention_mask_query_with_answer = torch.cat([attention_mask_query, torch.ones_like(some_answer_ids), torch.tensor([1])])
        # print(idx)
        assert len(mask) == len(input_ids_query_with_answer) == len(token_type_ids_query_with_answer) == len(attention_mask_query_with_answer)

        return input_ids_query_with_answer, token_type_ids_query_with_answer, attention_mask_query_with_answer, mask, audio_info


def get_dataset(s2l_dataset, tokenizer):

    return ASREnhDataset(s2l_dataset, tokenizer)


def get_collate_function(eos_token_id):
    def collate_fnc(data):
        input_ids_query_with_answers, token_type_ids_query_with_answers, attention_mask_query_with_answers, masks, audio_infos = zip(*data)
        input_ids_query_with_answers = list(input_ids_query_with_answers)
        token_type_ids_query_with_answers = list(token_type_ids_query_with_answers)
        attention_mask_query_with_answers = list(attention_mask_query_with_answers)
        masks = list(masks)
        audio_infos = list(audio_infos)
        max_len = max([t.shape[-1] for t in input_ids_query_with_answers])

        for i in range(len(input_ids_query_with_answers)):
            pad_len = max_len - input_ids_query_with_answers[i].shape[-1]
            masks[i] = torch.cat([masks[i], torch.tensor(pad_len*[False], dtype=bool)], dim=0)
            input_ids_query_with_answers[i] =      torch.cat([input_ids_query_with_answers[i], torch.tensor(pad_len*[eos_token_id], dtype=int)], dim=0)
            attention_mask_query_with_answers[i] = torch.cat([attention_mask_query_with_answers[i], torch.tensor(pad_len*[0], dtype=int)], dim=0)
            token_type_ids_query_with_answers[i] = torch.cat([token_type_ids_query_with_answers[i], torch.tensor(pad_len*[0], dtype=int)], dim=0)

        input_ids_query_with_answers = torch.stack(input_ids_query_with_answers)
        masks = torch.stack(masks)
        attention_mask_query_with_answers = torch.stack(attention_mask_query_with_answers)
        token_type_ids_query_with_answers = torch.stack(token_type_ids_query_with_answers)

        # return input_ids_query_with_answers, token_type_ids_query_with_answers, attention_mask_query_with_answers, masks
        return {
            'input_ids': input_ids_query_with_answers,
            'token_type_ids': token_type_ids_query_with_answers,
            'attention_mask': attention_mask_query_with_answers
        }, masks, audio_infos

    return collate_fnc
