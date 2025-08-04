import pandas as pd
import glob
import os
import time
import tqdm
import logging
# from multiprocessing import Process,Pool
import torch.multiprocessing as mp
from functools import partial
from audio_worker import AudioWorker
from file_worker import write_json,read_json
import torch
import os
import re
from tqdm import tqdm

from pydub import AudioSegment

def get_audio_duration(file_path):
    # Загружаем аудио файл
    audio = AudioSegment.from_file(file_path)
    # Длительность в миллисекундах, переводим в секунды
    duration_in_seconds = len(audio) / 1000
    return duration_in_seconds
def extract_number(filename):

    match = re.search(r'\d+', filename)

    return int(match.group()) if match else None


# audios_path = "/workspace-SR006.nfs2/shares/SR006.nfs2/Nikita/speech2latex/chatgpt_generated_new_eng_kin/Kin_24000"
# audio_paths_tts = glob.glob(f"{audios_path}/*.wav")


import pandas as pd
test_path = "/home/jovyan/Nikita/speech2latex/table_creator/ExtraASR/test_ru_eng_2000.csv"
df = pd.read_csv(test_path).iloc[:10]


path_output = "/home/jovyan/Nikita/speech2latex/table_creator/ExtraASR/whisper_cor_test_1.csv"

WHISPER_MODEL = "openai/whisper-large-v3"
whisper_path = "/home/jovyan/Nikita/speech2latex/audio_preprocessing/whisper-small-hi/checkpoint-5000"
device = "cuda"

# df = pd.read_csv("/home/jovyan/Nikita/speech2latex/table_creator/ExtraASR/chatgpt_generated_eng.csv")

audioWorker = AudioWorker(whisper_path,WHISPER_MODEL, device) 

batch_size = 1

dict_trans = {
    # "id":[],
    "whisper_new_transcription":[],
    "whisper_old":[],
    "pron":[],
    "audio_path":[]
}

# for audio_path in audio_paths_tts:
#     assert os.path.exists(audio_path), "No file"

audio_paths_tts = []
pron = []
old_whisper = []

_times = []
_tt = []

start_time = time.time()
for _,row in df.iterrows():
    
    dict_trans["pron"].append(row["pronunciation"])
    dict_trans["whisper_old"].append(row["whisper_transcription"])
    dict_trans["audio_path"].append(row["audio_path"])
    # dict_trans["whisper_transcription"].append(row["whisper_transcription"])
    
    audio_paths_tts.append(row["audio_path"])
    pron.append(row["pronunciation"])
    old_whisper.append(row["whisper_transcription"])


    batch_time = time.time()
    if batch_size == len(audio_paths_tts):
    
        # for i in tqdm(range(0, len(audio_paths_tts), batch_size)):

        #     # batch_paths = audio_paths_tts[i:min(len(audio_paths_tts), i+batch_size)]
        batch_paths = audio_paths_tts
        results = audioWorker.get_word_timestamps_batch(batch_paths,batch_size=batch_size)

        for result,audio_path,_pron,_old_whisper in zip(results,batch_paths,pron,old_whisper):
            whisper_trans = result["text"]
            # ext_name = os.path.splitext(os.path.basename(audio_path))[0]
            # id = extract_number(ext_name)
            # dict_trans["id"].append(id)
            dict_trans["whisper_new_transcription"].append(whisper_trans)

            _tt.append(get_audio_duration(audio_path))
            print("\n")
            print("whisper_trans",whisper_trans)
            print("old_whisper",_old_whisper)
            print("pron",_pron)

    _times.append(time.time() - batch_time)
    audio_paths_tts = []
    pron = []
    old_whisper = []


duration = time.time() - start_time
print(f"duration work {duration} for table shape{df.shape}")
print(f"среднее время работы одного батча {batch_size}",sum(_times)/len(_times))
print("для Темы чето",sum(_tt)/duration)

res = pd.DataFrame(dict_trans)
res.to_csv(path_output,index=False)


