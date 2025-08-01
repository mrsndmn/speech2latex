from tqdm.auto import tqdm
from datasets import load_dataset
import os
import json
import argparse
import sys
import pandas as pd
from s2l.eval import LatexInContextMetrics
from process_formula import NormalizeFormula


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('file_path', type=str)
    parser.add_argument('--use_normalized', type=bool, default=False)
    parser.add_argument('--compile_latex', type=bool, default=False)
    parser.add_argument('--add_lang_splits_for_s2l_equations_and_sentences', action='store_true')
    args = parser.parse_args()
    # file_path = args.file_path
    use_normalized = args.use_normalized
    compile_latex = args.compile_latex
    # add_lang_splits_for_s2l_equations_and_sentences = args.add_lang_splits_for_s2l_equations_and_sentences
    add_lang_splits_for_s2l_equations_and_sentences = True

    # file_path = sys.argv[1]

    math_speech_bench = False

    in_context_metrics = LatexInContextMetrics()
    all_s2l = load_dataset("marsianin500/Speech2Latex")

    all_file_paths = [
        "./ckpts/pretrained_Qwen2.5-0.5B/asr-normalized-Qwen2.5-0.5B_equations_sentence_multilingual_mix_0BGM0x/evaluation_generations.csv",
        "./ckpts/pretrained_Qwen2.5-0.5B/asr-normalized-Qwen2.5-0.5B_equations_sentence_normalized_multilingual_human_5eywXJ/evaluation_generations.csv",
        "./ckpts/pretrained_Qwen2.5-0.5B/asr-normalized-Qwen2.5-0.5B_equations_sentence_normalized_multilingual_mix_morqcH/evaluation_generations.csv",
        "./ckpts/pretrained_Qwen2.5-0.5B/asr-normalized-Qwen2.5-0.5B_equations_sentence_normalized_multilingual_synthetic_small_7DwkTe/evaluation_generations.csv",
        # "./ckpts/pretrained_Qwen2.5-0.5B/asr-normalized-Qwen2.5-0.5B_sentences_sentence_normalized_multilingual_human_IJRhXr/evaluation_generations.csv",
        # "./ckpts/pretrained_Qwen2.5-0.5B/asr-normalized-Qwen2.5-0.5B_sentences_sentence_normalized_multilingual_mix_aWixjn/evaluation_generations.csv",
        # "./ckpts/pretrained_Qwen2.5-0.5B/asr-normalized-Qwen2.5-0.5B_sentences_sentence_normalized_multilingual_synthetic_small_QSNfNK/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-0.5B-instruct/equations_sentence_normalized_multilingual_synthetic_small_41g36z/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-0.5B-instruct/equations_sentence_normalized_multilingual_human_YbWUdh/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-0.5B-instruct/equations_sentence_normalized_multilingual_mix_tduwXj/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-0.5B-instruct/equations_sentence_normalized_multilingual_mix_full_nqVVnm/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct/equations_sentence_normalized_multilingual_mix_a1RmwL/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct/equations_sentence_normalized_multilingual_human_ILOnRi/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct/equations_sentence_normalized_multilingual_synthetic_small_yt3JPE/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct/equations_sentence_normalized_multilingual_mix_HHKfmZ/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct/equations_sentence_normalized_multilingual_human_ENpADw/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct/equations_sentence_normalized_multilingual_synthetic_small_IOy1KE/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct/equations_sentence_normalized_multilingual_mix_FvQUss/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct/equations_sentence_normalized_multilingual_mix_full_fMgSB2/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-1.5B-instruct/equations_sentence_normalized_multilingual_synthetic_small_B9kVzY/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-1.5B-instruct/equations_sentence_normalized_multilingual_mix_DzL6wo/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-1.5B-instruct/equations_sentence_normalized_multilingual_human_KGVdMs/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct-2e/equations_sentence_normalized_multilingual_human_5LnTkw/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct-2e/equations_sentence_normalized_multilingual_synthetic_small_XFaHHa/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-math-1.5B-instruct-2e/equations_sentence_normalized_multilingual_mix_6dIkTo/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-1.5B-instruct-test/equations_sentence_normalized_multilingual_mix_full_0xvAsv/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-7B-instruct/equations_sentence_normalized_multilingual_human_X9yn4M/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-7B-instruct/equations_sentence_normalized_multilingual_mix_full_VzfvEG/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-7B-instruct/equations_sentence_normalized_multilingual_synthetic_small_lTALQs/evaluation_generations.csv",
        "./ckpts/asr-normalized-Qwen2.5-7B-instruct/equations_sentence_normalized_multilingual_mix_2SxAhg/evaluation_generations.csv",
    ]

    for file_path in tqdm(all_file_paths):

        if not os.path.exists(file_path):
            print("File not exists: ", file_path)
            continue

        if 'MathSpeech_LaTeX_result' in file_path:
            math_speech_bench = True
            latex_true_column = 'LaTeX'
            latex_pred_column = 'MathSpeech_LaTeX_result'
        else:
            latex_true_column = 'latex_true'
            latex_pred_column = 'latex_pred'

        df = pd.read_csv(file_path)

        df[latex_pred_column] = df[latex_pred_column].apply(lambda x: x.removeprefix('$').removesuffix('$').strip())
        df[latex_true_column] = df[latex_true_column].apply(lambda x: x.removeprefix('$').removesuffix('$').strip())
        # df[latex_pred_column] = df[latex_pred_column].apply(lambda x: x.replace('\displaystyle', '').replace('\operatorname', ''))
        # df[latex_true_column] = df[latex_true_column].apply(lambda x: x.replace('\displaystyle', '').replace('\operatorname', ''))

        # Calc normalized
        # if True:
        if compile_latex:
            print("Normalizing all")
            norm = NormalizeFormula(check_node=False)

            normalized_or_default_true = []
            normalized_or_default_pred = []
            normalized_true = norm(df[latex_true_column].values.tolist())
            normalized_pred = norm(df[latex_pred_column].values.tolist())

            non_compiled_true = 0
            for item, normalized in zip(df[latex_true_column].values.tolist(), normalized_true):
                if normalized == "":
                    normalized_or_default_true.append(item)
                    non_compiled_true += 1
                else:
                    normalized_or_default_true.append(normalized)

            non_compiled_pred = 0
            for item, normalized in zip(df[latex_pred_column].values.tolist(), normalized_pred):
                if normalized == "":
                    normalized_or_default_pred.append(item)
                    print("item", item)
                    non_compiled_pred += 1
                else:
                    normalized_or_default_pred.append(normalized)

            print("Non compiled pred: ", non_compiled_pred)
            print("Non compiled true: ", non_compiled_true)

        if use_normalized:
            df[latex_true_column] = normalized_or_default_true
            df[latex_pred_column] = normalized_or_default_pred


        # Filter non empty
        print("Orig length: ", len(df))
        df = df[df[latex_pred_column] != ""]
        df = df[df[latex_true_column] != ""]
        print("Filtered length: ", len(df))

        # for x, y in zip(df[latex_pred_column], df[latex_true_column]):
        #     print(x)
        #     print(y)
        #     print("-"*30)

        if not add_lang_splits_for_s2l_equations_and_sentences:
            metrics_values = in_context_metrics.compute_all(df[latex_pred_column], df[latex_true_column], compute_text_only=False, compute_formulas_only=False)
            in_context_metrics.dump(metrics_values)
        else:

            # Add language column from datasets
            print("len", len(df))

            if len(df) == len(all_s2l['equations_test']):
                language_column = all_s2l['equations_test']['language']
                is_tts_column = all_s2l['equations_test']['is_tts']
            # elif len(df) == len(all_s2l['sentences_test']):
            #     language_column = all_s2l['sentences_test']['language']
            else:
                raise ValueError("df size not matches any s2l splits")

            df['language'] = language_column
            df['is_tts'] = is_tts_column

            # Split by language
            df_ru =  df[ df['language']  == 'ru' ]
            df_eng = df[ df['language']  == 'eng' ]

            df_ru_mix = df_ru
            df_eng_mix = df_eng

            df_ru_tts = df_ru[ df_ru['is_tts'] == True ]
            df_ru_human = df_ru[ df_ru['is_tts'] == False ]

            df_eng_tts = df_eng[ df_eng['is_tts'] == True ]
            df_eng_human = df_eng[ df_eng['is_tts'] == False ]

            print("len df_ru_mix", len(df_ru_mix))
            print("len df_eng_mix", len(df_eng_mix))

            print("len df_ru_tts", len(df_ru_tts))
            print("len df_ru_human", len(df_ru_human))

            print("len df_eng_tts", len(df_eng_tts))
            print("len df_eng_human", len(df_eng_human))

            print("Computing metrics")

            for lang in [ 'ru', 'eng' ]:
                for split in [ 'mix', 'tts', 'human' ]:
                    df_lang = eval(f"df_{lang}_{split}")
                    metrics_values = in_context_metrics.compute_all(df_lang[latex_pred_column].values.tolist(), df_lang[latex_true_column].values.tolist(), compute_text_only=False, compute_formulas_only=False)
                    in_context_metrics.dump(metrics_values)
                    out_file = os.path.join(os.path.dirname(file_path), f"{lang}_{split}_metrics.json")
                    with open(out_file, 'w') as f:
                        json.dump(metrics_values, f)
