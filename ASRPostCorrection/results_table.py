#!/usr/bin/env python3
"""
Script to aggregate experiment results from ASRPostCorrection/ckpts folder.
Sorts experiments by properties and generates LaTeX tables.
"""

import os
import json
import re
from collections import defaultdict
from typing import Dict, List, Any
from tabulate import tabulate


def parse_experiment_name(name: str) -> Dict[str, str]:
    """
    Parse experiment directory name to extract properties.

    Expected format: asr-normalized-Qwen2.5-0.5B_{dataset_split}_{column_type}_{language}_{data_type}_{hash}
    """

    name = name.replace('sentence_normalized', 'sentence-norm')
    name = name.replace('synthetic_small', 'synthetic-small')

    # Remove the base model name and hash
    parts = name.replace('asr-normalized-Qwen2.5-0.5B_', '').split('_')

    if len(parts) >= 4:
        dataset_split = parts[0]  # equations or sentences
        column_type = parts[1]    # sentence or sentence_normalized
        language = parts[2]       # eng, ru, multilingual
        data_type = parts[3]      # human, synthetic_small, mix
        hash_id = '_'.join(parts[4:]) if len(parts) > 4 else ''

        return {
            'dataset_split': dataset_split,
            'column_type': column_type,
            'language': language,
            'data_type': data_type,
            'hash_id': hash_id,
            'full_name': name
        }
    else:
        # Fallback for unexpected formats
        return {
            'dataset_split': 'unknown',
            'column_type': 'unknown',
            'language': 'unknown',
            'data_type': 'unknown',
            'hash_id': '',
            'full_name': name
        }


def load_metrics(ckpt_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load metrics from a checkpoint directory.
    Returns dict with keys: artificial, humans, mix
    """
    metrics = {}
    metric_files = ['artificial_metrics.json', 'humans_metrics.json', 'mix_metrics.json']

    for metric_file in metric_files:
        file_path = os.path.join(ckpt_dir, metric_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract the split name from filename
                    split_name = metric_file.replace('_metrics.json', '')
                    metrics[split_name] = data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {file_path}: {e}")

    return metrics


def get_metric_value(metrics: Dict[str, float], metric_name: str, default: float = 0.0) -> float:
    """Safely get metric value with default fallback."""
    return metrics.get(metric_name, default) * 100


def create_results_table(experiments: List[Dict[str, Any]], split_type: str = 'mix') -> str:
    """
    Create a LaTeX table from experiment results.

    Args:
        experiments: List of experiment data
        split_type: Which split to use ('artificial', 'humans', 'mix')
    """
    # Define the metrics we want to display
    # metrics_to_show = [
    #     'wer', 'cer', 'bleu', 'sacrebleu', 'meteor', 'rouge1', 'chrf', 'tex_bleu'
    # ]

    metric_split_types = ['artificial', 'humans', 'mix']

    metrics_to_show = [
        'cer', 'cer_lower', 'tex_bleu'
    ]

    metrics_columns = []
    for metric_split_type in metric_split_types:
        for metric in metrics_to_show:
            metrics_columns.append(f"{metric_split_type}_{metric}")

    # Prepare table data
    table_data = []
    headers = ['Dataset', 'Column', 'Language', 'Data Type', 'Hash'] + [m.upper() for m in metrics_columns]

    for exp in experiments:
        if exp['properties']['data_type'] != split_type:
            # breakpoint()
            continue

        for metric_split_type in metric_split_types:
            # if metric_split_type not in exp['metrics']:
            #     continue

            metrics = exp['metrics'][metric_split_type]
            row = [
                exp['properties']['dataset_split'],
                exp['properties']['column_type'],
                exp['properties']['language'],
                exp['properties']['data_type'],
                exp['properties']['hash_id'][:8] if exp['properties']['hash_id'] else ''
            ]

            # Add metric values
            for metric in metrics_to_show:
                value = get_metric_value(metrics, metric)
                row.append(f"{value:.4f}")

            table_data.append(row)

    # Sort by dataset_split, language, data_type
    table_data.sort(key=lambda x: (x[0], x[2], x[3]))

    # Generate LaTeX table
    breakpoint()
    latex_table = tabulate(
        table_data,
        headers=headers,
        tablefmt='latex',
        floatfmt='.4f',
        numalign='right'
    )

    return latex_table


def main():
    # Path to the checkpoints directory
    ckpts_dir = "./ckpts"

    if not os.path.exists(ckpts_dir):
        print(f"Error: Directory {ckpts_dir} not found!")
        return

    # Collect all experiment directories
    experiments = []

    for item in os.listdir(ckpts_dir):
        item_path = os.path.join(ckpts_dir, item)
        if os.path.isdir(item_path) and item.startswith('asr-normalized-Qwen2.5-0.5B_'):
            # Parse experiment properties
            properties = parse_experiment_name(item)

            # Load metrics
            metrics = load_metrics(item_path)

            if metrics:  # Only include if we found metrics
                experiments.append({
                    'properties': properties,
                    'metrics': metrics,
                    'path': item_path
                })

    print(f"Found {len(experiments)} experiments with metrics")

    # Group experiments by properties for better organization
    grouped_experiments = defaultdict(list)
    for exp in experiments:
        key = (exp['properties']['dataset_split'], exp['properties']['language'])
        grouped_experiments[key].append(exp)

    # Generate tables for each split type
    split_types = ['mix', 'human', 'synthetic-small']

    for split_type in split_types:
        print(f"\n{'='*80}")
        print(f"Results for {split_type.upper()} split")
        print(f"{'='*80}")

        # Create table for all experiments
        all_table = create_results_table(experiments, split_type)
        print(all_table)

        # Create tables for each group
        for (dataset_split, language), group_exps in grouped_experiments.items():
            if len(group_exps) > 1:  # Only show groups with multiple experiments
                print(f"\n{'-'*60}")
                print(f"Group: {dataset_split} - {language}")
                print(f"{'-'*60}")
                group_table = create_results_table(group_exps, split_type)
                print(group_table)

    # # Save results to file
    # output_file = "experiment_results.tex"
    # with open(output_file, 'w') as f:
    #     f.write("\\documentclass{article}\n")
    #     f.write("\\usepackage{booktabs}\n")
    #     f.write("\\usepackage{longtable}\n")
    #     f.write("\\begin{document}\n\n")

    #     for split_type in split_types:
    #         f.write(f"\\section*{{Results for {split_type.upper()} split}}\n")
    #         f.write(create_results_table(experiments, split_type))
    #         f.write("\n\n")

    #     f.write("\\end{document}\n")

    # print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
