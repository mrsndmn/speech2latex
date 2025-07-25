import pandas as pd
import os
import time
import tqdm
import logging
# from multiprocessing import Process,Pool
import torch.multiprocessing as mp
from functools import partial
from audio_worker import AudioWorker
from file_worker import write_json
import torch
import os
import re
import tqdm

# Настройка логирования
logging.basicConfig(
    filename='fast-main.log',  # Лог сохраняется в файл csv_worker.log
    level=logging.INFO,         # Уровень логирования (INFO, DEBUG, ERROR и т.д.)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_number(filename):
    # # Регулярное выражение для поиска числа между "audio_" и ".wav"
    # match = re.search(r'audio_(\d+)\.wav', filename)
    # if match:
    #     return int(match.group(1))
    # else:
    #     return None
    # Используем регулярное выражение для поиска первого числа
    match = re.search(r'\d+', filename)
    # Если число найдено, возвращаем его как целое число, иначе возвращаем None
    return int(match.group()) if match else None


def process_audios(audioWorker:AudioWorker,audio_paths_arr,df:pd.DataFrame, _result_path):

    # #мега костыль. Потому что не обработано первое произношение в мат бридже.
    df_iters = df.iterrows()
    # next(df_iters)

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
        # print(apa)
        # print(extract_number(apa[1]))
        assert int(file_id) == extract_number(apa[1]), f"Айди не метчатся  {int(file_id)} { extract_number(apa[1])}"

        result_path = os.path.join(_result_path,f"{file_name}.json")

        if os.path.exists(result_path):
            print(f"{result_path} уже существует")
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
    # Чтение Excel файла
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
        logging.info(f"Программа завершила работу")


if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    data_paths = [
        # "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/gpt_dataset_ru/Bys_24000",
        #          "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/gpt_dataset_ru/May_24000",
        #          "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/gpt_dataset_ru/Nec_24000",
        #          "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/gpt_dataset_ru/Ost_24000",
        #          "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/gpt_dataset_ru/Pon_24000",
        #          "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/gpt_dataset_ru/Tur_24000",
                #  "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/saluth__mathbridge_eng",
                                # "/home/jovyan/Nikita/TTS_S2L_generation/math_bridge_dima_eng",
                                "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/TagMePools"
                ]
    excel_paths = [
        # "/home/jovyan/Nikita/SberSaluth/data/s2l_main_v0_final_fixed.xlsx",
        #            "/home/jovyan/Nikita/SberSaluth/data/s2l_main_v0_final_fixed.xlsx",
        #            "/home/jovyan/Nikita/SberSaluth/data/s2l_main_v0_final_fixed.xlsx",
        #            "/home/jovyan/Nikita/SberSaluth/data/s2l_main_v0_final_fixed.xlsx",
        #            "/home/jovyan/Nikita/SberSaluth/data/s2l_main_v0_final_fixed.xlsx",
                   "/home/jovyan/Nikita/SberSaluth/data/s2l_main_v0_final_fixed.xlsx",
                    # "/home/jovyan/Nikita/SberSaluth/data/math_bridge.xlsx"
                    ]

    sheet_names = [
        "truly_checked_new_formulas"
    ]

    assert len(data_paths) == len(excel_paths),f"Не соответствуют пути аудио и excel {len(data_paths)} {len(excel_paths)}"

    result_path = "../whisper_synthesized_audios_final"
    WHISPER_MODEL = "large-v3"
    CUDA = 2

    os.makedirs(result_path,exist_ok=True)
    run(data_paths,excel_paths,sheet_names,result_path,WHISPER_MODEL,CUDA)

