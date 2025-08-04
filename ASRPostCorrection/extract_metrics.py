import os
import pickle

from s2l.eval import LatexInContextMetrics

def read_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def dump_metrics(metrics):
    # print("CER\tCER(EQ)\t\tCER(txt)\tRouge1(EQ)")
    # Short Metrics

    # LatexInContextMetrics.dump(metrics)

    # print("{value:.2f}\t & {value_furmulas:.2f} & \t{value_text:.2f} &\t{rouge:.2f} \\\\".format(value=100*metrics['cer_lower'],              value_furmulas=100*metrics.get('formulas_only_cer_lower', 0.0),      value_text=100*metrics.get('text_only_cer_lower', 0.0), rouge=100*metrics.get('formulas_only_rouge1', 0.0)))
    # return

    metric_names = [
        'formulas_only_cer',
        'formulas_only_cer_lower',
        'formulas_only_rouge1',
        'formulas_only_sacrebleu',
        'formulas_only_chrfpp',
        'text_only_sacrebleu',
        'text_only_wer_lower',
        'text_only_meteor',
    ]
    metric_values = []
    for key in metric_names:
        metric_values.append("{value:.2f}".format(value=metrics[key] * 100))

    print(" & ".join(metric_values), end="")
    print("")

    # Full Metrics


def process_directories(directories):
    for directory in directories:
        print(f"\n\n===============\nProcessing directory: {directory}")
        artificial_path = os.path.join(directory, 'artificial_metrics.pickle')

        if os.path.exists(artificial_path):
            artificial_data = read_pickle_file(artificial_path)
            print("artificial_metrics:")
            dump_metrics(artificial_data)
        else:
            print("File artificial_metrics.pickle not found")

    for directory in directories:
        print(f"\n\n===============\nProcessing directory: {directory}")
        humans_path = os.path.join(directory, 'humans_metrics.pickle')

        if os.path.exists(humans_path):
            humans_data = read_pickle_file(humans_path)
            print("humans_metrics:")
            dump_metrics(humans_data)

        else:
            print("File humans_metrics.pickle not found")

    return

if __name__ == "__main__":
    # directories = ['qwen2.5_artificial_1epoch', 'qwen2.5_humans_1epoch', 'qwen2.5_mix_1epoch']
    directories = ['qwen2.5_artificial_2e', 'qwen2.5_humans_2e', 'qwen2.5_mix_2e']
    # directories = ['qwen2.5_artificial', 'qwen2.5_humans', 'qwen2.5_mix']
    # directories = ['qwen2.5_artificial_norm', 'qwen2.5_humans_norm', 'qwen2.5_mix_norm']
    process_directories(directories)
