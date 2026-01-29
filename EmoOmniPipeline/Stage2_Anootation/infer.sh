#!/bin/bash
# Usage: bash infer.sh <model_path> <input_jsonl> <result_path> [gpu_id]
# Example: bash infer.sh MODEL/SABEER-LLM data.jsonl result.jsonl 0

if [ "$#" -lt 3 ]; then
    echo "Usage: bash infer.sh <model_path> <input_jsonl> <result_path> [gpu_id]"
    exit 1
fi

MODEL_PATH=$1
INPUT_JSONL=$2
RESULT_PATH=$3
GPU_ID=${4:-0} # Default to GPU 0 if not provided

SWIFT_PATH="/xxx/xxx/ms-swift"
cd "$SWIFT_PATH" || exit

echo "Using GPU: $GPU_ID"
echo "Starting inference for $INPUT_JSONL..."

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
    --val_dataset "$INPUT_JSONL" \
    --result_path "$RESULT_PATH"

echo "Finished processing $INPUT_JSONL. Results saved to $RESULT_PATH."