import re
from typing import List
from process_formula import NormalizeFormula

from tqdm.auto import tqdm

# 
# Нормализация работает костыльно и если обучаться
# или считать на нормализованных данных, метркики
# только проседают, поэтому по умолчанию нормализация нигде не используется
# 

def normalize_in_context_formulas(text_line: str):

    normalized_text_line = text_line

    if '$$' in text_line:
        formulas_content = text_line.split('$$')
    else:
        formulas_content = text_line.split('$')

    if len(formulas_content) < 2:
        return normalized_text_line

    normlizer = NormalizeFormula()

    # each even element of formulas_content is formula content
    # print("formulas_content[1::2]", formulas_content[1::2])
    normalized_formulas_reversed = list(reversed(normlizer(formulas_content[1::2])))

    print("normalized_formulas_reversed", normalized_formulas_reversed)

    if sum(1 for x in normalized_formulas_reversed if x == '') > 0:
        # if katex failed to normalize leave formula as is
        return text_line

    i_start = len(normalized_text_line)
    i_formula = 0
    for substring_i in range(len(formulas_content)-1, 0, -1):
        if substring_i % 2 == 1:
            # formula
            formula_start = i_start - (len(formulas_content[substring_i]))

            formula_content: str = normalized_formulas_reversed[i_formula]
            # Костыль от нормализатора. Нормализатор добавляет кучу пробелов
            # И вообще говоря, довольно сложно честно восстановить их.
            # Регулярки ниже удаляют большую часть пробелов, за исключением, когда это
            # изменяет формулу, например:
            # Исходная формула:        \sum weight*5
            # После нормализации:      \sum w e i g h t * 5
            # После удаления пробелов: \sum weight*5
            while True:
                text_prev = formula_content
                formula_content = re.sub(r"(\s[a-zA-Z]+)\s+([a-zA-Z])", r'\1\2', formula_content)
                if text_prev == formula_content:
                    break

            formula_content = re.sub(r"([^a-zA-Z])\s+(?=[^a-zA-Z])", r'\1', formula_content)
            formula_content = re.sub(r"([a-zA-Z])\s+(?=[^a-zA-Z])", r'\1', formula_content)
            formula_content = re.sub(r"([^a-zA-Z])\s+(?=[a-zA-Z])", r'\1', formula_content)
            i_formula += 1

            normalized_text_line = normalized_text_line[:formula_start] + formula_content + normalized_text_line[i_start:]

        i_start -= (len(formulas_content[substring_i]) + 1)

        # normalized_text_line = normalized_text_line[]

    return normalized_text_line


def normalize_in_context_formulas_bulk(formulas_list: List[str], with_tqdm=False):
    result = []
    
    iterator = formulas_list
    if with_tqdm:
        iterator = tqdm(formulas_list)

    for f in iterator:
        result.append(normalize_in_context_formulas(f))

    return result

