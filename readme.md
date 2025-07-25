# RSI-Speech2Latex


# Install dependencies

```
pip install -r requirements.txt
```


# ASRPostCorrection

## Training

```
cd ASRPostCorrection
PYTHONPATH=. python -m pdb -c continue train_qwen.py --config ./config-qwen2.5-in_context_training.json --train_df ../Data/latex_in_context_tts/latex_in_context_tts_v2_train.csv --val_df ../Data/latex_in_context_tts/latex_in_context_tts_v2_test.csv
```

## Testing trained model

```
cd ASRPostCorrection
python test_qwen.py --cuda 0 --test_file_csv ../Data/latex_in_context_tts/latex_in_context_tts_v2_test.csv --batch_size 20 --ckpt ./ckpts/tts-in-context/version_9/
```

## Evaluation

Its also possible to compute metrics from file without model.

Example usage:
```python
from s2l.eval import LatexInContextMetrics
in_context_metrics = LatexInContextMetrics()
metrics_values = in_context_metrics.compute_all(outputs['latex_pred'], outputs['latex_true'])
in_context_metrics.dump(metrics_values)
```

Example CLI-usage:
```
python src/s2l/eval.py --csv-data ./Data/latex_in_context_tts/latex_in_context_tts_v2_train.csv --pred-column model_prediction --target-column target_text
```
