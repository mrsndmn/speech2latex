
import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    client, extra_options = training_job_api_from_profile('default')

    workdir = os.getcwd()

    env_prefix = '/workspace-SR004.nfs2/d.tarasov/envs/dtarasov-speech2latex/bin'

    author_name = 'd.tarasov'

    os.listdir(f'{workdir}/configs')

    for config_file in os.listdir(f'{workdir}/configs'):
        result = client.run_job(
            payload={
                'script': f"cd {workdir} && {env_prefix}/python train.py --config configs/{config_file}",
                'job_desc': f'S2L: Qwen Audio FT {config_file} #{author_name} #rnd #multimodal @mrsndmn',
                'env_variables': {
                    'PYTHONPATH': './:../src:/workspace-SR004.nfs2/d.tarasov/ProcessLaTeXFormulaTools/',
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

        print(config_file, result)

