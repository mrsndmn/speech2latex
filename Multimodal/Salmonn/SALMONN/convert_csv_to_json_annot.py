
import pandas as pd
import json

df = pd.read_csv("train.csv")
df = df[df["language"] == "eng"]

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)


# Split the shuffled DataFrame
train_size = int(0.9 * len(df_shuffled))
train_df = df_shuffled[:train_size]
val_df = df_shuffled[train_size:]

def write( path,dict):
    with open(path, "w+") as f:
        f.write(json.dumps(dict,indent=4))

train_anno = {
    "annotation":[ {"path":row['audio_path'], "text":row['formula_normalized'], "task" : "asr" } for _, row in train_df.iterrows()]
}

val_anno = {
    "annotation":[ {"path":row['audio_path'], "text":row['formula_normalized'], "task" : "asr" } for _, row in val_df.iterrows()]
}


write("data/train_anno.json",train_anno)
write("data/val_anno.json",val_anno)

df = pd.read_csv("train.csv")
test_df = df[df["language"] == "eng"]

def write( path,dict):
    with open(path, "w+") as f:
        f.write(json.dumps(dict,indent=4))

test_anno = {
    "annotation":[ {"path":row['audio_path'], "text":row['formula_normalized'], "task" : "asr" } for _, row in test_df.iterrows()]
}

write("data/test_anno.json",test_anno)
