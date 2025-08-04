import pandas as pd
import os
import time
import tqdm
import logging

import torch.multiprocessing as mp
from functools import partial
from audio_worker import AudioWorker
from file_worker import write_json,read_json
import torch
import os
import re
import tqdm

logging.basicConfig(
    filename='whisper_pool.log',  
    level=logging.INFO,         
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_number(filename):

    match = re.search(r'\d+', filename)
    
    return int(match.group()) if match else None


def process_audios(audioWorker:AudioWorker,audio_paths_arr,df:pd.DataFrame, _result_path):

    ids = set(df["FILENAME"].astype(dtype=int).to_list())

    
    batch_size = 64
    pbar = tqdm.tqdm(range(0,len(audio_paths_arr),batch_size))

    for i in pbar:

        batch_paths = audio_paths_arr[i:min(len(audio_paths_arr),i+batch_size)]
        results = audioWorker.get_word_timestamps_batch(batch_paths,batch_size=batch_size)
        
        
        pbar.set_description(f"process batch {i}:{i+batch_size}")
        for result,apa in zip(results,batch_paths):
            audio_path = os.path.join(*apa)
            
            file_name  = extract_extension(apa[1])

            file_id = extract_number(apa[1])
            if not file_id in ids:
                logging.info(f"audio_path {audio_path} has no match with table ")
                continue

            pronunciation = df[df['FILENAME'] == file_id]["INPUT:text"].values[0]
            latex = df[df['FILENAME'] == file_id]["INPUT:text3"].values[0]

            output_path = os.path.join(_result_path, os.path.basename(apa[0]))
            os.makedirs(output_path,exist_ok=True)      
            result_path = os.path.join(output_path,f"{file_name}.json")

            if os.path.exists(result_path):
                print(f"{result_path} already exists")
                continue

            
                  
            words_timestamps = None
            sentences_timestamps = None

            words_timestamps = result["chunks"]
            sentences_timestamps = audioWorker.get_sentences_with_timestamps(words_timestamps)

            dict_output = {
                "words_timestamps": words_timestamps,
                "sentences_timestamps":sentences_timestamps,
                "file_name": file_id,
                "pronunciation":pronunciation,
                "audio_path": audio_path,
                "latex":latex
            }

            write_json(result_path,dict_output)
            
            logging.info(f"audio_path {audio_path} filename {file_name} result_path {result_path} ")


def extract_extension(file_name):
    return os.path.splitext(file_name)[0]


def get_audios_paths(data_path):

    audios_paths = []
    for dirpath, dirnames, filenames in  os.walk(data_path):
        for filename in filenames:
            audios_paths.append([dirpath,filename])
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



        audioWorker = AudioWorker(WHISPER_MODEL,CUDA) 
        print(f"columns in df {excel_path}", df.columns)
        process_audios(audioWorker,audio_paths,df,result_path)
        logging.info(f"Finish")


if __name__ == "__main__":
    
    data_paths = [
                "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/TagMePools/eng/pool4_eng"
                ]
    excel_paths = [
                "/home/jovyan/Nikita/speech2latex/SberSaluth/data/math_bridge.xlsx",
                    ]

    sheet_names = [
        "truly_checked_new_formulas"
    ]

    assert len(data_paths) == len(excel_paths)

    result_path = "/home/jovyan/Nikita/speech2latex/SberSaluth/whisper_synthesized_audios_final/real/eng"
    WHISPER_MODEL = "large-v3"
    CUDA = 1

    os.makedirs(result_path,exist_ok=True)
    run(data_paths,excel_paths,sheet_names,result_path,WHISPER_MODEL,CUDA)

