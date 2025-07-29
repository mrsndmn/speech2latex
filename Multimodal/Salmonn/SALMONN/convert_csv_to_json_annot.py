
import pandas as pd
import json
import os
from glob import glob
from pathlib import Path

mathbridge_audios = list(glob("Data/mathbridge/wavs/*.wav"))
df = pd.read_csv("Data/mathbridge/MathBridge_train_cleaned_normalized_train.csv")

def write( path,dict):
    with open(path, "w+") as f:
        f.write(json.dumps(dict,indent=4))


path = "./Data/mathbridge"

train_anno = {
    "annotation":[ {"path":audio_path, "text":df.iloc[int(Path(audio_path).stem)]['formula_normalized_2'], "task" : "asr" } for audio_path in mathbridge_audios]
}


write(os.path.join(path,"train_anno.json"),train_anno)