
import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    client, extra_options = training_job_api_from_profile('default')

    workdir = os.getcwd()

    author_name = 'd.tarasov'

    total_shards = 6

    for shard_num in range(total_shards):
        input_file = f'./MathSpeech_s2l_equations_full_shard_{shard_num}_of_{total_shards}.csv'
        output_file = f'./MathSpeech_s2l_equations_full_shard_{shard_num}_of_{total_shards}_generated.csv'

        result = client.run_job(
            payload={
                'script': f"bash -c 'cd {workdir} && /workspace-SR004.nfs2/d.tarasov/envs/dtarasov-speech2latex/bin/python MathSpeech_eval.py --input_csv {input_file} --output_csv {output_file}'",
                'job_desc': f'MathSpeech Generate {shard_num} #{author_name} #rnd #multimodal',
                'instance_type': 'a100.1gpu',
                'region': extra_options['region'],
                'type': 'binary_exp',
                'shm_size_class': 'medium',
                'base_image': 'cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36',
                'n_workers': 1,
                'processes_per_worker': 1,
            }
        )

        print(shard_num, result)

