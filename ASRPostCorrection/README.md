

# Train 
```
PYTHONPATH=. python -m pdb -c continue train_qwen.py --config ./config-qwen2.5-debug.json --train_df ../Data/trainable_split/train_ENG_to_submit_max_700.csv --val_df ../Data/trainable_split/test_ENG_to_submit.csv
````

# Test
```
PYTHONPATH=. python -m pdb -c continue test_qwen.py --test_file_csv ../Data/trainable_split/test_ENG_to_submit.csv --ckpt ./ckpts/asr-sentence/version_42/
```
