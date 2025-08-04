#!/bin/bash

export OMP_NUM_THREADS=6
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
# Set environment variables
export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="127.0.0.1"
export PORT=29501
export MID_RUN_NAME="my_run_name"

DEFAULT_RUN_NAME="finetune_on_speech2latex_equations_human_multilingual"

STAGE_PATH=${1:-"nvidia/audio-flamingo-3"}

# data_mixture_1 is the entry of the dataset in llava/data/datasets_mixture.py.
DATA_MIXTURE=${2:-"human_multilingual"}


if [ "$NNODES" = "1" ] || [ "$NNODES" = "2" ]; then
    echo "Detected on single machine. Automatically set batch size to 1 for debugging purpose."
    PER_DEVICE_TRAIN_BATCH_SIZE=1
fi
    
    
# torchrun --nnodes \$NUM_NODES --nproc_per_node \$SUBMIT_GPUS --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT --node_rank \$NODE_RANK \
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3_gradient_clipping.json \
    --model_name_or_path $STAGE_PATH \
    --chat_template qwen2 \
    --data_mixture $DATA_MIXTURE \
    --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
    --dynamic_s2 True \
    --s2_scales "448,896,1344" \
    --s2_max_split_size 448 \
    --s2_resize_output_to_scale_idx -1 \
    --speech_tower openai/whisper-large-v2 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --speech_mm_projector mlp \
    --sound_mm_projector mlp \
    --tune_vision_tower False \
    --tune_speech_tower False \
    --tune_sound_tower True \
    --tune_mm_projector False \
    --tune_speech_mm_projector False \
    --tune_sound_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio dynamic_s2 \
    --bf16 True \
    --audio_frames 20 \
    --output_dir runs/train/finetune_on_speech2latex_equations_human_multilingual \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 300 \
    --save_total_limit 4 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    # --sound_tower /path/to/AF-Whisper \
