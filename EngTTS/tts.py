import pandas as pd
from TTS.api import TTS
import os
from tqdm.auto import tqdm
import json



def read_csv(path, output_dir):
    df = pd.read_csv(path)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        if 'id' in row:
            file_name = row['id']
        else:
            file_name = str(index)

        text = row['pronunciation']
        lang = "en" if row["language"] == "eng" else "ru"
        if lang == "ru":
            continue

        output_file = os.path.join(output_dir, f"audio_{file_name}.wav")

        if os.path.exists(output_file):
            print(f"{output_file} was skipped")
            continue
        
        yield text, output_file, lang

def read_json_by_row(path,output_dir, req_lang = "en"):
    with open(path) as file:
        lines = file.readlines()
        for index, line in tqdm(enumerate(lines), total = len(lines)):
            dict_line = json.loads(line)
            text = dict_line['pronunciation']

            if 'id' in dict_line:
                file_name = dict_line['id']
            else:
                file_name = str(index)

            lang = None
            if "language" in dict_line:
                lang = dict_line["language"]
                            
            if lang is None:
                lang = req_lang

            if lang != req_lang:
                continue

            output_file = os.path.join(output_dir, f"audio_{file_name}.wav")

            if os.path.exists(output_file):
                print(f"{output_file} was skipped")
                continue
            
            yield text, output_file, lang
            

path = "../sample_data/latex_in_context_15k.jsonl_with_transcriptions.jsonl"
assert os.path.exists(path)

ext = os.path.basename(path).split(".")[-1]

if ext in ["csv"] :
    reader_func = read_csv
elif ext in ["jsonl"]:
    reader_func = read_json_by_row

device = "cuda:0"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

output_dir = 'tts'
os.makedirs(output_dir, exist_ok=True)

for text, output_file, lang in reader_func(path, output_dir):
    try:
        tts.tts_to_file(text=text, file_path=output_file, speaker_wav='./g300.wav', language=lang)
    except Exception as ex:
        raise ex
        path_err = "./err.txt"
        with open(path_err, "a+") as file:
            file.write(f"{text}, {output_file}\n")
