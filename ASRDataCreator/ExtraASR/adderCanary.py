import pandas as pd
from CanaryWorker import CanaryWorker
import tqdm



device = 3
model = CanaryWorker(device)


path = "/home/jovyan/Nikita/speech2latex/table_creator/whisper_synthesized_audios_final_excel/dataset_match_3.xlsx"
path_output = "asr_canary.csv"

df = pd.read_excel(path)

dict_asr = {
    "canary_transcription":[]
}
print("размер таблицы",df.shape)

for _,row in tqdm.tqdm(df.iterrows()):
    audio_path = row["audio_path"]
    lang = row["language"]

    transcription_walm = model.get_transcription(audio_path) if lang == "eng" else ""
    dict_asr["canary_transcription"].append(transcription_walm)


df_asr = pd.DataFrame(dict_asr)
df_asr.to_csv(path_output,index=False)

print(f"file saved {path_output}")

