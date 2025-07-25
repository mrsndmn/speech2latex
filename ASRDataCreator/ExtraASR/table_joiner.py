import pandas as pd 

table = pd.read_excel("/home/jovyan/Nikita/speech2latex/table_creator/whisper_synthesized_audios_final_excel/dataset_match_3.xlsx")
extra_colums_1 = pd.read_csv("./asr_walm_wav2vec.csv")
extra_colums_2 = pd.read_csv("./asr_qwen_edit.csv")
extra_colums_3 = pd.read_csv("./asr_canary.csv")

united_table = pd.concat(
    [table,extra_colums_1,extra_colums_2,extra_colums_3],
    axis = 1
)

path_output = "./output/post_cor_5asr_edit.csv"
path_excel_output =  "./output/post_cor_5asr_edit.xlsx"
united_table.to_csv(path_output,index = False)
united_table.to_excel(path_excel_output, index = False)