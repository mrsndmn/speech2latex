import pandas as pd
import os
import time
import tqdm
import logging

import torch.multiprocessing as mp
from functools import partial
from audio_worker import AudioWorker
from file_worker import write_json
import torch
import os
import re
import tqdm

logging.basicConfig(
    filename='fast-main.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_number(filename):

    match = re.search(r'\d+', filename)

    return int(match.group()) if match else None


def process_audios(audioWorker:AudioWorker,audio_paths_arr,df:pd.DataFrame, _result_path):


    df_iters = df.iterrows()


    print(df.head())
    print(audio_paths_arr[:5])

    assert len(audio_paths_arr) == df.shape[0]

    pbar = tqdm.tqdm(zip(audio_paths_arr,df_iters))
    for apa,(_,row) in pbar:
        file_id = row["FILENAME"]
        pronunciation = row["INPUT:text"]
        pbar.set_description(f"process audio {file_id}")

        file_name  = extract_extension(apa[1])
        audio_path = os.path.join(*apa)


        result_path = os.path.join(_result_path,f"{file_name}.json")

        if os.path.exists(result_path):
            continue

        words_timestamps = audioWorker.get_word_timestamps(audio_path)
        sentences_timestamps = audioWorker.get_sentences_with_timestamps(words_timestamps)

        dict_output = {
            "words_timestamps":words_timestamps,
            "sentences_timestamps":sentences_timestamps,
            "file_name": file_id,
            "pronunciation":pronunciation
        }

        write_json(result_path,dict_output)

        logging.info(f"audio_path {audio_path} filename {file_name} result_path {result_path} ")


def extract_extension(file_name):
    return os.path.splitext(file_name)[0]


def get_audios_paths(data_path):

    audios_paths = []
    for dirpath, dirnames, filenames in  os.walk(data_path):
        for filename in filenames:
            audios_paths.append((dirpath,filename))
    return sorted(audios_paths,key = lambda x:extract_number(x[1]))



def read_excel_to_df(file_path,sheet_name = 0):

    ending = file_path.split(".")[-1]
    if ending == "csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path,sheet_name)



def run(data_paths,excel_paths,sheet_names,result_path,WHISPER_MODEL,CUDA):

    for data_path,excel_path,sheet_name in zip(data_paths,excel_paths,sheet_names):
        audio_paths = get_audios_paths(data_path)

        df = read_excel_to_df(excel_path,sheet_name)

        output_path = os.path.join(result_path, os.path.basename(data_path))
        os.makedirs(output_path,exist_ok=True)

        audioWorker = AudioWorker(WHISPER_MODEL,CUDA)
        print(f"columns in df {excel_path}", df.columns)
        process_audios(audioWorker,audio_paths,df,output_path)


if __name__ == "__main__":

    data_paths = [
        "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/TagMePools"
    ]
    excel_paths = [
            "/home/jovyan/Nikita/SberSaluth/data/s2l_main_v0_final_fixed.xlsx",
    ]

    sheet_names = [
        "truly_checked_new_formulas"
    ]


    result_path = "../whisper_synthesized_audios_final"
    WHISPER_MODEL = "large-v3"
    CUDA = 2

    os.makedirs(result_path,exist_ok=True)
    run(data_paths,excel_paths,sheet_names,result_path,WHISPER_MODEL,CUDA)

