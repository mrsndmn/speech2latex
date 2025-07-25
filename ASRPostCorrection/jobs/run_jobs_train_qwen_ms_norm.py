
import os

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
            'exp_name': 'tts-equations-ms-norm-added-tokens',
            'experiment_dir': 'qwen2.5_equations_math_speech_normalized_added_tokens',
            'train_df': '../MathSpeech/Experiments/MathSpeech_s2l_equations_full_generated_with_whisper_en.csv',
            'config': './config-qwen2.5-equations-math-speech-normalized-added-tokens.json',
        },
        {
            'exp_name': 'tts-equations-ms-norm',
            'experiment_dir': 'qwen2.5_equations_math_speech_normalized',
            'train_df': '../MathSpeech/Experiments/MathSpeech_s2l_equations_full_generated_with_whisper_en.csv',
            'config': './config-qwen2.5-equations-math-speech-normalized.json',
        },
    ]

    for exp_config in exp_configs:
        exp_name = exp_config['exp_name']
        experiment_dir = exp_config['experiment_dir']
        train_df = exp_config['train_df']
        config = exp_config['config']


        result = client.run_job(
            payload={
                'script': f"bash -c 'cd {workdir} && {python_bin} train_test_qwen.py --test_equations_math_speech_normalized --experiment_dir {experiment_dir} --train_df {train_df} --config {config}'",
                'job_desc': f'S2L: {exp_name} #{author_name} #rnd #multimodal @mrsndmn',
                'env_variables': {
                    'WANDB_API_KEY': wandb_api_key,
                    "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
                    'PYTHONPATH': './:../src:/workspace-SR004.nfs2/d.tarasov/ProcessLaTeXFormulaTools/'
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

