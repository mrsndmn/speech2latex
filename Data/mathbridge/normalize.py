from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
import pandas as pd
import re
import os

from process_formula import NormalizeFormula


if __name__ == "__main__":

    df = pd.read_csv('MathBridge_train_cleaned.csv')

    # df = df.head(100000)
    # df = df.tail(50000)
    # df = df.tail(len(df) - 1583177)

    hf_ds = Dataset.from_pandas(df)
    original_length = len(hf_ds)

    def remove_unescaped_percent(item):
        if re.findall(r'(?<!\\)\s*%', item['equation']):
            return False
        return True

    hf_ds = hf_ds.filter(remove_unescaped_percent)
    print("Filtered out: ", original_length - len(hf_ds), " samples")
    original_length = len(hf_ds)


    def validate_formulas(batch):
        norm = NormalizeFormula(check_node=False)
        normalized = norm(batch['equation'])
        # for n in normalized:
        #     if n == '':
        #         print(f"Normalized formula is empty: {batch['equation']}")
        #         # raise ValueError(f"Normalized formula is empty: {batch['equation']}")

        return {
            # mark whether the formula successfully compiles (non-empty string)
            'formula_normalized': [f"$ {n} $" for n in normalized]
        }

    # number of parallel processes; default to all CPU cores if available
    num_proc = min(32, os.cpu_count() or 1)  # cap at 8 to avoid oversubscription
    batch_size = 100

    print(f"\nRunning LaTeX compilation check in parallel using {num_proc} workers …")

    hf_ds = hf_ds.map(
        validate_formulas,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Validate formulas double",
    )

    def cleanup_final(item):
        # Not compiled
        if item['formula_normalized'] in ['$  $', '']:
            return False
        # Too short item
        if len(item['formula_normalized']) < 9:
            return False

        if 'undefined' in item['formula_normalized']:
            return False

        return True

    orig_length = len(hf_ds)
    hf_ds = hf_ds.filter(cleanup_final)
    print("Filtered out 1: ", orig_length - len(hf_ds), " samples")

    # the second normalization

    def validate_formulas_2(batch):
        norm = NormalizeFormula(check_node=False)
        normalized = norm(batch['formula_normalized'])
        # for n in normalized:
        #     if n == '':
        #         print(f"Normalized formula is empty: {batch['formula_normalized']}")
        #         # raise ValueError(f"Normalized formula is empty: {batch['equation']}")

        return {
            # mark whether the formula successfully compiles (non-empty string)
            'formula_normalized_2': [f"$ {n} $" for n in normalized]
        }
    hf_ds = hf_ds.map(
        validate_formulas_2,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Validate formulas",
    )

    orig_length = len(hf_ds)

    def cleanup_final_2(item):
        # Not compiled
        if item['formula_normalized_2'] in ['$  $', '']:
            return False
        # Too short item
        if len(item['formula_normalized_2']) < 9:
            return False

        if 'undefined' in item['formula_normalized_2']:
            return False

        return True


    hf_ds = hf_ds.filter(cleanup_final_2)
    print("Filtered out 2: ", orig_length - len(hf_ds), " samples")

    df_normalized = hf_ds.to_pandas()
    df_normalized.to_csv('MathBridge_train_cleaned_normalized.csv', index=False)

    dfn = df_normalized[['equation', 'formula_normalized']]

    errors = 0
    undefined_in_normalized = 0

    for i, row in dfn.iterrows():

        # Extract all latex operators from the formula
        # Ignore doubly escaped backslashes \\
        formula_normalized_re = r'(?<!\\)\\(?:[a-zA-Z]+)'
        operators_normalized = re.findall(formula_normalized_re, row['formula_normalized'])
        operators = re.findall(formula_normalized_re, row['equation'])

        if '%' in row['equation']:
            continue

        if row['formula_normalized'] in ['$  $', '']:
            # not compiled
            continue

        if 'undefined' in row['formula_normalized']:
            undefined_in_normalized += 1
            print("undefined in normalized", row['equation'])
            print("undefined in normalized", row['formula_normalized'])
            continue

        operators_normalized_set = set(operators_normalized)
        operators_set = set(operators)

        if '\\prime' in operators_normalized_set and '\\prime' not in operators_set:
            operators_normalized_set.remove('\\prime')

        if '\\textbf' in operators_set and '\\text' in operators_normalized_set:
            operators_set.remove('\\textbf')
            operators_normalized_set.remove('\\text')
            if '\\text' in operators_set:
                operators_set.remove('\\text')
            if '\\textbf' in operators_normalized_set:
                operators_normalized_set.remove('\\textbf')

        if '\\rm' in operators_set and '\\mathrm' in operators_normalized_set:
            operators_set.remove('\\rm')
            operators_normalized_set.remove('\\mathrm')
            if '\\mathrm' in operators_set:
                operators_set.remove('\\mathrm')

        replacable_operators = {
            '\\cal': '\\mathcal',
            '\\rm': '\\mathrm',
            '\\bf': '\\mathbf',
            '\\it': '\\mathit',
            '\\sf': '\\mathsf',
            '\\tt': '\\mathtt',
            '\\Bbb': '\\mathbb',
            '\\dfrac': '\\frac',
            '\\tfrac': '\\frac',
            '\\bm': '\\boldsymbol',
            '\\choose': '\\binom',
            '\\over': '\\frac',
            '\\hskip': '\\hspace',
            '\\bold': '\\mathbf',
            '\\atop': '\\binom',
            '\\frak': '\\mathfrak',
        }

        ops_to_remove = ['\\mathrel', '\\limits', '\\big', '\\bigg', '\\underset', '\\overset', '\\stackrel', '\\mathord', '\\bigl', '\\bigr', '\\Bigl', '\\Bigr', '\\biggl', '\\biggr', '\\Bigl', '\\Bigr', '\\biggl', '\\biggr', '\\Big', '\\mathop', '\\nolimits', '\\texttt', '\\primer', '\\primee']
        for ops in ops_to_remove:
            if ops in operators_set:
                operators_set.remove(ops)
            if ops in operators_normalized_set:
                operators_normalized_set.remove(ops)

        for operator in list(operators_set):
            if operator in replacable_operators:
                operators_set.remove(operator)
                operators_set.add(replacable_operators[operator])

        if '\\nobreakspace' in operators_normalized_set and '~' in row['equation']:
            operators_normalized_set.remove('\\nobreakspace')

        if operators_normalized_set != operators_set:
            print("--------------------------------")
            print("error", errors)
            print("i", i)
            print(row['equation'])
            print(row['formula_normalized'])
            print("operators_normalized_set", operators_normalized_set)
            print("operators_set           ", operators_set)
            errors += 1
            # if errors > 10:
            #     breakpoint()

    print(dfn.sample(10))

    print("errors", errors)
    print("undefined_in_normalized", undefined_in_normalized)
    breakpoint()

