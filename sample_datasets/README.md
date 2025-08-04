This directory contains sample datasets for the submission.

Load datasets:
```python
from datasets import DatasetDict

# Speech2Latex Equations and Sentences
speech2latex_equations_sentences = DatasetDict.load_from_disk('./speech2latex_equations_sentences_10_samples')

print(speech2latex_equations_sentences)

# MathBridge Cleaned Subset Data
speech2latex_mathbridge = DatasetDict.load_from_disk('./speech2latex_mathbridge_10_samples')

print(speech2latex_mathbridge)
```
