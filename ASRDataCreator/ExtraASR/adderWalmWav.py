import pandas as pd
from walmworker import WavLMWorker
from wav2vac_worker import Wav2Vec2Worker
import tqdm



device = 1
model_walm = WavLMWorker(device)
model_wav2vec = Wav2Vec2Worker(device)


path = "/home/jovyan/Nikita/speech2latex/table_creator/whisper_synthesized_audios_final_excel/dataset_match_3.xlsx"
path_output = "asr_walm_wav2vec.csv"

df = pd.read_excel(path)

dict_asr = {
    "walm_transcription":[],
    "wav2vec_transcription":[]
}

for _,row in tqdm.tqdm(df.iterrows()):
    audio_path = row["audio_path"]
    lang = row["language"]

    transcription_walm = model_walm.get_transcription(audio_path) if lang == "eng" else ""
    dict_asr["walm_transcription"].append(transcription_walm)

    transcription_wav2vec = model_wav2vec.get_transcription(audio_path) if lang == "eng" else ""
    dict_asr["wav2vec_transcription"].append(transcription_wav2vec)



df_asr = pd.DataFrame(dict_asr)
df_asr.to_csv(path_output,index=False)

print(f"file saved {path_output}")

