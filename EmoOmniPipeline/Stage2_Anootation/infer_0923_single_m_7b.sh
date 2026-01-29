##swift路径
cd /xxx/xxx/ms-swift


# 要推理的文件路径 (一个 .jsonl 文件)
DATA_JSONL=""

#模型和对应的保存的路径
MODEL_BASE_PATH=""
RESULT_BASE_DIR=""

# 在此处配置您要使用的GPU ID
GPU_ID=0

echo "Using GPU: $GPU_ID"
echo "Starting inference process..."
echo "Starting inference for $val_dataset on GPU $GPU_ID"

CUDA_VISIBLE_DEVICES=$GPU_ID \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=40 \
MAX_PIXELS=1003520 \
ENABLE_AUDIO_OUTPUT=0 \
swift infer \
    --model "$MODEL_PATH" \
    --stream true \
    --infer_backend pt \
    --write_batch_size -1 \
    --max_new_tokens 4096 \
    --val_dataset "$val_dataset" \
    --result_path "$final_result_path"

echo "Finished processing $val_dataset."
