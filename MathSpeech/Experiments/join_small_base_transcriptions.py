import sys
import os
from glob import glob
import pandas as pd


if __name__ == '__main__':

    test_shard_base = f"./result_ASR_s2l_equations_test_base_base_shard_0_of_1.csv"
    test_shard_small = f"./result_ASR_s2l_equations_test_small_small_shard_0_of_1.csv"

    test_df_base  = pd.read_csv(test_shard_base)
    test_df_small = pd.read_csv(test_shard_small)

    test_df_full = test_df_base.copy()
    test_df_full['whisper_small_predSE'] = test_df_small['whisper_small_predSE']

    print("test dataset saved to ./MathSpeech_s2l_equations_test_full_normalized.csv", len(test_df_full))
    test_df_full.to_csv(f"./MathSpeech_s2l_equations_test_full_normalized.csv", index=False)

    sys.exit()

    total_shards = 6
    for i in range(total_shards):
        shard_base = f"./result_ASR_s2l_equations_base_shard_{i}_of_{total_shards}.csv"
        shard_small = f"./result_ASR_s2l_equations_small_shard_{i}_of_{total_shards}.csv"

        df_base  = pd.read_csv(shard_base)
        df_small = pd.read_csv(shard_small)

        df_full = df_base.copy()
        df_full['whisper_small_predSE'] = df_small['whisper_small_predSE']

        output_file = f"./MathSpeech_s2l_equations_full_shard_{i}_of_{total_shards}.csv"
        df_full.to_csv(output_file, index=False)
        print(f"shard {i} of {total_shards} saved to {output_file}")