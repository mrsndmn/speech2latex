import sys
import pandas as pd
from s2l.eval import LatexInContextMetrics


if __name__ == "__main__":

    file_path = sys.argv[1]

    math_speech_bench = False

    if 'MathSpeech_LaTeX_result' in file_path:
        math_speech_bench = True
        latex_true_column = 'LaTeX'
        latex_pred_column = 'MathSpeech_LaTeX_result'
    else:
        latex_true_column = 'latex_true'
        latex_pred_column = 'latex_pred'

    df = pd.read_csv(file_path)

    if not math_speech_bench:
        df[latex_pred_column] = df[latex_pred_column].apply(lambda x: x.replace(' ', '').replace('\displaystyle', '').replace('\operatorname', ''))
        df[latex_true_column] = df[latex_true_column].apply(lambda x: x.replace(' ', '').replace('\displaystyle', '').replace('\operatorname', ''))
    else:
        df[latex_true_column] = df[latex_true_column].apply(lambda x: f"${x}$")
        df[latex_pred_column] = df[latex_pred_column].apply(lambda x: x.replace(' ', '').replace('\displaystyle', '').replace('\operatorname', ''))
        df[latex_true_column] = df[latex_true_column].apply(lambda x: x.replace(' ', '').replace('\displaystyle', '').replace('\operatorname', ''))


    for x, y in zip(df[latex_pred_column], df[latex_true_column]):
        print(x)
        print(y)
        print("-"*30)

    in_context_metrics = LatexInContextMetrics()
    metrics_values = in_context_metrics.compute_all(df[latex_pred_column], df[latex_true_column], compute_text_only=False, compute_formulas_only=False)

    in_context_metrics.dump(metrics_values)

