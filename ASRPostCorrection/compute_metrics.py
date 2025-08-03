import sys
import pandas as pd
from s2l.eval import LatexInContextMetrics
from process_formula import NormalizeFormula


if __name__ == "__main__":

    file_path = sys.argv[1]

    math_speech_bench = False

    # use_normalized = True
    use_normalized = False

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
    if False:
    # if True:
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

    in_context_metrics = LatexInContextMetrics()
    metrics_values = in_context_metrics.compute_all(df[latex_pred_column], df[latex_true_column], compute_text_only=False, compute_formulas_only=False)

    in_context_metrics.dump(metrics_values)

    breakpoint()

