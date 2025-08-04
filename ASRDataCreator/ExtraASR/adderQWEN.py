import pandas as pd
from QwenWorker import QwenAudioWorker
import tqdm

device = 1
model = QwenAudioWorker(device)

path = "/home/jovyan/Nikita/speech2latex/table_creator/ExtraASR/output/post_cor_5asr.csv"
path_output = "asr_qwen_edit.csv"

df = pd.read_csv(path)

dict_asr = {
    "qwen_transcription":[],
}

# qwen_eng_na = df[(df["language"] == "eng") & (df["qwen_transcription"].isna())]

for _,row in tqdm.tqdm(df.iterrows()):
    audio_path = row["audio_path"]
    lang = row["language"]
    qwen_transcription = row["qwen_transcription"]

    if (pd.isna(qwen_transcription) or len(qwen_transcription) == 1) and (lang == "eng"):
        transcription_walm = model.get_transcription(audio_path)
        dict_asr["qwen_transcription"].append(transcription_walm)
    # if (lang == "eng") and (len(qwen_transcription) == 0 or len(qwen_transcription) == 1):
    else:
        dict_asr["qwen_transcription"].append(qwen_transcription)


df_asr = pd.DataFrame(dict_asr)
df_asr.to_csv(path_output,index=False)

print(f"file saved {path_output}")

