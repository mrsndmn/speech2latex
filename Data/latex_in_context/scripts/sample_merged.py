
from tqdm.auto import tqdm
import glob
import pandas as pd
import json
import math

millnames = ['',' k',' M',' B']

def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def merge_jsonl_files(files):
    rows = []
    
    for file in tqdm(files, desc='loaded jsonl files'):
        with open(file, 'r') as f:
            for line in f:
                row = json.loads(line)
                rows.append(row)
                
    return pd.DataFrame(rows)


if __name__ == "__main__":
    
    shard_files = list(glob.glob('data/*'))
    
    merged_prefix = "latex_in_context_since_2019"
    
    merged_df = merge_jsonl_files(shard_files)
    merged_df.to_json(f'./{merged_prefix}_{millify(len(merged_df))}.jsonl', orient='records', lines=True)
    
    df_100k = merged_df.sample(100000)
    df_100k.to_json(f'./{merged_prefix}_100k.jsonl', orient='records', lines=True)
    
    df_15k = df_100k.sample(15000)
    df_15k.to_json(f'./{merged_prefix}_15k.jsonl', orient='records', lines=True)
