import pytest

from test_qwen import normalize_in_context_formulas

def test_extract_in_context_formulas():

    normalized_formula_not_changed = "$\\sum _ { 1 } ^ { 2 } test$, context"
    assert normalize_in_context_formulas(normalized_formula_not_changed) == "$\\sum_{1}^{2}t e s t$, context"

    unnormazlized_formula = "$ \\sum_1^2 test $, context"
    assert normalize_in_context_formulas(unnormazlized_formula) == "$\\sum_{1}^{2}t e s t$, context"

    multi_formulas = "$ \\sum_1^2 $ sdsdfsd $ test $, context"
    assert normalize_in_context_formulas(multi_formulas) == "$\\sum_{1}^{2}$ sdsdfsd $t e s t$, context"

    unnormazlized_formula_text_collate = "$ \\sum x + y $, context"
    assert normalize_in_context_formulas(unnormazlized_formula_text_collate) == "$\\sum x+y$, context"
