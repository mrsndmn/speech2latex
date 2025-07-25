
import os
import sys

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    client, extra_options = training_job_api_from_profile('default')

    wandb_api_key = os.environ.get('WANDB_API_KEY')
    if wandb_api_key is None or wandb_api_key == '':
        raise ValueError('WANDB_API_KEY is not set')

    workdir = os.getcwd()
    python_bin = '/workspace-SR004.nfs2/d.tarasov/envs/dtarasov-speech2latex/bin/python'

    author_name = 'd.tarasov'


    exp_configs = [
        {
            'exp_name': 'tts-equations-my-norm-added-tokens',
            'experiment_dir': 'qwen2.5_equations_my_normalized_added_tokens',
            'train_df': '../Data/mathbridge/MathBridge_train_cleaned_normalized.csv',
            'config': './config-qwen2.5-equations-my-normalized-added-tokens.json',
        },
        {
            'exp_name': 'tts-equations-my-norm',
            'experiment_dir': 'qwen2.5_equations_my_normalized',
            'train_df': '../Data/mathbridge/MathBridge_train_cleaned_normalized.csv',
            'config': './config-qwen2.5-equations-my-normalized.json',
        },
    ]

    for exp_config in exp_configs:
        exp_name = exp_config['exp_name']
        experiment_dir = exp_config['experiment_dir']
        train_df = exp_config['train_df']
        config = exp_config['config']

        script_string = f"bash -c 'cd {workdir} && {python_bin} train_test_qwen.py --test_equations_my_normalized --test_equations_unnormalized --few_train_samples 256000 --few_test_samples 1000 --experiment_dir {experiment_dir} --train_df {train_df} --config {config}'"

        print("script_string", script_string)

        if len(sys.argv) > 1 and sys.argv[1] == 'dry':
            print("dry run")
            continue

        result = client.run_job(
            payload={
                'script': script_string,
                'job_desc': f'S2L: {exp_name} #{author_name} #rnd #multimodal @mrsndmn',
                'env_variables': {
                    'WANDB_API_KEY': wandb_api_key,
                    "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
                    'PYTHONPATH': './:../src:/workspace-SR004.nfs2/d.tarasov/ProcessLaTeXFormulaTools/',
                    'WANDB_MODE': 'disabled',
                },
                'instance_type': 'a100.1gpu',
                'region': extra_options['region'],
                'type': 'binary_exp',
                'shm_size_class': 'medium',
                'base_image': 'cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36',
                'n_workers': 1,              # Количество воркеров.
                'processes_per_worker': 1,   # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
            }
        )

        print(exp_name, result)

