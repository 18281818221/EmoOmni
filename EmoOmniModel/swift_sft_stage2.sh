#!/bin/bash  # Add shebang to specify shell interpreter  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
export PYTHONWARNINGS="ignore::DeprecationWarning:librosa.core.audio"
export PYTHONWARNINGS="ignore::FutureWarning:librosa.core.audio"
# 仅屏蔽 librosa 模块的 UserWarning（更精准）
export PYTHONWARNINGS="ignore::UserWarning:librosa"

echo "entry load complete"
pip -V
conda activate swift_sft
pip -V

export NCCL_CONNECT_TIMEOUT=60
export NCCL_IB_TIMEOUT=60
export OMP_NUM_THREADS=8
export MPI_NUM_THREADS=8
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 读取 CUDA_VISIBLE_DEVICES 环境变量
VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}
# 计算 GPU 数量（通过逗号分割后的元素个数）
if [ -z "$VISIBLE_DEVICES" ]; then
    # 如果未设置，默认使用所有可用 GPU（通过 nvidia-smi 获取）
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
else
    # 按逗号分割并统计数量
    GPU_COUNT=$(echo "$VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

export NPROC_PER_NODE=${GPU_COUNT}
export FPS_MAX_FRAMES=40
export MASTER_PORT=7845
export ENABLE_AUDIO_OUTPUT=0
export USE_AUDIO_IN_VIDEO=True
export ENABLE_VIDEO_OUTPUT=0
export VIDEO_MAX_PIXELS=50176
# export FORCE_QWENVL_VIDEO_READER=torchvision
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' 
# export FORCE_QWENVL_VIDEO_READER=decord
export FORCE_QWENVL_VIDEO_READER=torchvision
HOST_NODE_PORT=${HOST_NODE_PORT:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 5)}
export PYTHONWARNINGS="ignore"


python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${ARNOLD_ID} \
    --nnodes=${ARNOLD_WORKER_NUM} \
    --master_addr=${ARNOLD_WORKER_0_HOST} \
    --master_port=${HOST_NODE_PORT} \
    -m swift.cli.sft \
    --model "Qwen/Qwen2.5-Omni-7B" \
    --dataset \
        "./MultimodalEmotionalDialogueData/MEIJU25_CN_AnNo_merged.jsonl" \
        "./MultimodalEmotionalDialogueData/MEIJU25_CN_Anno.jsonl" \
        "./MultimodalEmotionalDialogueData/MEIJU25_EN_Anno_merged.jsonl" \
        "./MultimodalEmotionalDialogueData/MEIJU25_EN_AnNo_merged.jsonl" \
        "./MultimodalEmotionalDialogueData/MER25.jsonl" \
        "./MultimodalEmotionalDialogueData/MultiDialog.jsonl" \
        "./MultimodalEmotionalDialogueData/privcate_movie_data.jsonl" \
        "./MultimodalEmotionalDialogueData/Multitask_Stage1_Data-Random50k.jsonl" \
    --torch_dtype bfloat16 \
    --train_type lora   \
    --lora_rank 8   \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --learning_rate 1e-4  \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 10000 \
    --logging_steps 10 \
    --max_length 12000 \
    --output_dir "ckpt_output/lm_output_dialogure/exp1" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --gradient_checkpointing=True \
    --dataloader_pin_memory=True \
    --dataloader_prefetch_factor=2 \
    --ddp_find_unused_parameters true \
    --deepspeed zero2 \
    --split_dataset_ratio 0 \
    --attn_impl flash_attn \
    --save_only_model false 

    
