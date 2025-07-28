
import os
import sys

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    dry_run = len(sys.argv) > 1 and sys.argv[1] == 'dry'

    client, extra_options = training_job_api_from_profile('default')

    workdir = os.getcwd()

    env_prefix = '/workspace-SR004.nfs2/d.tarasov/envs/dtarasov-speech2latex/bin'

    author_name = 'd.tarasov'

    os.listdir(f'{workdir}/configs')

    # for dataset_split in ['sentences', 'equations']:
    for dataset_split in ['equations']:
        for latex_column_name in ['sentence', 'sentence_normalized']:
            for language in ['multilingual']:
            # for language in ['eng', 'ru', 'multilingual']:
                for data_type in ['mix']:
                # for data_type in ['human', 'synthetic_small', 'mix']:
                # for data_type in ['human', 'synthetic_small', 'synthetic_full', 'mix']:

                    command = f"cd {workdir} && {env_prefix}/python train_test_qwen.py --dataset_split {dataset_split} --latex_column_name {latex_column_name} --language {language} --data_type {data_type} --config configs/config-qwen2.5.json"
                    print("\n\n", command)
                    if dry_run:
                        continue

                    result = client.run_job(
                        payload={
                            'script': command,
                            'job_desc': f'S2L: Qwen2.5 lang={language} data={data_type} #{author_name} #rnd #multimodal @mrsndmn',
                            'env_variables': {
                                'PYTHONPATH': './:../src:/workspace-SR004.nfs2/d.tarasov/ProcessLaTeXFormulaTools/:../TeXBLEU',
                                'HF_HOME': '/workspace-SR004.nfs2/.cache/huggingface',
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

                    print("language", language, "data_type", data_type, "result", result)

