from s2l.normalization import normalize_in_context_formulas_bulk

import pandas as pd
import sys

import multiprocessing as mp

from tqdm.auto import tqdm

def process_frame(df):
    try:
        df['sentence'] = normalize_in_context_formulas_bulk(df['sentence'], with_tqdm=False)
    except Exception as e:
        print("e", e)

    # df['whisper_text'] = normalize_in_context_formulas_bulk(df['whisper_text'], with_tqdm=False)

    return df

if __name__ == "__main__":

    df_reader = pd.read_csv(sys.argv[1], chunksize=10)

    output_csv = sys.argv[1]

    pool = mp.Pool(128)

    funclist = []
    for df in df_reader:
        # process each data frame
        f = pool.apply_async(process_frame,[df])
        funclist.append(f)

    result = []
    for f in tqdm(funclist):
        result.append(f.get(timeout=60)) # timeout in 10 seconds

    df_result = pd.concat(result)
    df_result.to_csv(sys.argv[2], index=False)

    breakpoint()