from datasets import load_dataset
import sys
import pandas as pd
from s2l.eval import LatexInContextMetrics
from process_formula import NormalizeFormula


if __name__ == "__main__":

    file_path = sys.argv[1]

    math_speech_bench = False

    eq_test = load_dataset('marsianin500/Speech2Latex', split='equations_test')
    eq_test = eq_test.remove_columns(set(eq_test.column_names) - set(['is_tts', 'language', 'sentence_normalized']))
    eq_test_df = eq_test.to_pandas()

    eq_test_df_mix_multilang = eq_test_df.copy()
    eq_test_df_mix_ru = eq_test_df[eq_test_df['language'] == 'ru']
    eq_test_df_mix_eng = eq_test_df[eq_test_df['language'] == 'eng']

    eq_test_df_human_multilang = eq_test_df[eq_test_df['is_tts'] == 0]
    eq_test_df_tts_multilang = eq_test_df[(eq_test_df['is_tts'] == 1)]

    eq_test_df_human_ru = eq_test_df[(eq_test_df['is_tts'] == 0) & (eq_test_df['language'] == 'ru')]
    eq_test_df_tts_ru = eq_test_df[(eq_test_df['is_tts'] == 1) & (eq_test_df['language'] == 'ru')]

    eq_test_df_human_eng = eq_test_df[(eq_test_df['is_tts'] == 0) & (eq_test_df['language'] == 'eng')]
    eq_test_df_tts_eng = eq_test_df[(eq_test_df['is_tts'] == 1) & (eq_test_df['language'] == 'eng')]

    length_to_dataframe_mapping = {
        len(eq_test_df_mix_multilang): (eq_test_df_mix_multilang['is_tts'], 'eq_test_df_mix_multilang'),
        len(eq_test_df_mix_ru): (eq_test_df_mix_ru['is_tts'], 'eq_test_df_mix_ru'),
        len(eq_test_df_mix_eng): (eq_test_df_mix_eng['is_tts'], 'eq_test_df_mix_eng'),
        len(eq_test_df_human_multilang): (eq_test_df_human_multilang['is_tts'], 'eq_test_df_human_multilang'),
        len(eq_test_df_tts_multilang): (eq_test_df_tts_multilang['is_tts'], 'eq_test_df_tts_multilang'),
        len(eq_test_df_human_ru): (eq_test_df_human_ru['is_tts'], 'eq_test_df_human_ru'),
        len(eq_test_df_tts_ru): (eq_test_df_tts_ru['is_tts'], 'eq_test_df_tts_ru'),
        len(eq_test_df_human_eng): (eq_test_df_human_eng['is_tts'], 'eq_test_df_human_eng'),
        len(eq_test_df_tts_eng): (eq_test_df_tts_eng['is_tts'], 'eq_test_df_tts_eng'),
    }

    assert len(length_to_dataframe_mapping) == 9


    latex_true_column = 'latex_true'
    latex_pred_column = 'latex_pred'

    df = pd.read_csv(file_path)

    df[latex_pred_column] = df[latex_pred_column].apply(lambda x: x.removeprefix('$').removesuffix('$').strip())
    df[latex_true_column] = df[latex_true_column].apply(lambda x: x.removeprefix('$').removesuffix('$').strip())


    # Filter non empty
    print("Orig length: ", len(df))
    df = df[df[latex_pred_column] != ""]
    df = df[df[latex_true_column] != ""]
    print("Filtered length: ", len(df))


    is_tts, split_name = length_to_dataframe_mapping[len(df)]

    human_df = df[(is_tts == 0).values]
    tts_df = df[(is_tts == 1).values]

    in_context_metrics = LatexInContextMetrics()

    human_metrics = in_context_metrics.compute_all(human_df[latex_pred_column].values.tolist(), human_df[latex_true_column].values.tolist(), compute_text_only=False, compute_formulas_only=False)
    tts_metrics = in_context_metrics.compute_all(tts_df[latex_pred_column].values.tolist(), tts_df[latex_true_column].values.tolist(), compute_text_only=False, compute_formulas_only=False)
    mix_values = in_context_metrics.compute_all(df[latex_pred_column].values.tolist(), df[latex_true_column].values.tolist(), compute_text_only=False, compute_formulas_only=False)

    in_context_metrics.dump(human_metrics)
    in_context_metrics.dump(tts_metrics)
    in_context_metrics.dump(mix_values)

    print("Latex results:", split_name)
    print(f"Flamingo                   & TODO             & TODO             & TODO           &       {mix_values['cer_lower']*100:.2f} &     {mix_values['tex_bleu']*100:.2f} &     {human_metrics['cer_lower']*100:.2f} &      {human_metrics['tex_bleu']*100:.2f} &      {tts_metrics['cer_lower']*100:.2f} &        {tts_metrics['tex_bleu']*100:.2f}  \\")
    # print("    &   ".join(map(lambda x: f"{x*100*100:.2f}", [ , , , , ,  ])))

    breakpoint()

