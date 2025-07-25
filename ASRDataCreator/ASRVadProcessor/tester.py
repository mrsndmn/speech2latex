# from wav2vac_worker import Wav2Vec2Worker
# from walmworker import WavLMWorker
# from QwenWorker import QwenAudioWorker
# from qwen_worker_batch import QwenAudioWorkerBatch
from CanaryWorker import CanaryWorker
import glob
import pandas as pd
from collections import defaultdict
import time
import  tqdm
def output_test(name,df,model):
    with open(f"test_model_{name}.txt", "w+") as f:
        # batch_buffer = defaultdict(list)
        start_time = time.time()
        for _, row in tqdm.tqdm(df.iterrows()):
            audio_path = row["audio_path"]
            pron = row["pronunciation"]
            whisper_pron = row["whisper_transcription"]

            f.write("\n")
            f.write(f"predict {name} {model.get_transcription(audio_path)}\n") 
            f.write(f"predict whisper {whisper_pron}\n")
            f.write(f"gt {pron}\n")

        duration = time.time() - start_time   
        return duration




df = pd.read_excel("dataset_match_3.xlsx")

device = 2

# model = WavLMWorker().to(f"cuda:{device}")
# name = "walm"

# name ="Qwen_eng"
# model = QwenAudioWorker(device)

# t = model.get_batched_transcriptions(audio_paths_en)
# for _t in t:
#     print(_t)


# model = Wav2Vec2Worker(device)
# name = "wav2vec"

df = df[df["language"] == "eng"]
count_rows = df.shape[0]
n = 100
samples = df.sample(n ,random_state=42)

models = [
    # Wav2Vec2Worker,
    # WavLMWorker,
    # QwenAudioWorker,
    CanaryWorker

]
names = [
    # "Wav2vec_eng",
    # "WavLM_eng",
    # "Qwen_eng",
    "Canary"
]

print(samples["language"].value_counts())
print(f"cont rows {count_rows}")
print(f"sample shape {samples.shape}")
print(samples.head())

for Model,name in zip(models,names):
    model = Model(device)
    print(f"start {name}")
    duration = output_test(name,samples,model)
    print(f"{name} finsihed for {duration} in sample size {n}")
    print(f"expected time for rows {count_rows} is {(count_rows * duration) / n } ")

    


