#!/bin/bash
# LLaDA Model - General Evaluation Script
#
# This script evaluates LLaDA-8B-Instruct on any task
#
# Usage:
#   bash llada.sh [GPU_IDS] [TASK] [OUTPUT_PATH] [MODEL_PATH] [MASTER_PORT]
#
# Examples:
#   # ChartQA
#   bash /home/aiscuser/lmms-eval/g2u/llada.sh "0" "chartqa100" "./logs/llada_chartqa"
#
#   # MMLU
#   bash /home/aiscuser/lmms-eval/g2u/llada.sh "0" "mmlu" "./logs/llada_mmlu"
#
#   # Multiple tasks (comma-separated, no spaces)
#   bash /home/aiscuser/lmms-eval/g2u/llada.sh "0" "chartqa100,mmbench" "./logs/llada_multi"
#
#   # With custom port
#   bash /home/aiscuser/lmms-eval/g2u/llada.sh "0" "chartqa100" "./logs/llada_chartqa" "GSAI-ML/LLaDA-8B-Instruct" "29700"

# ============ Configuration ============
GPU_IDS=${1:-"0"}
TASK=${2:-"mmlu"}
OUTPUT_PATH=${3:-"./logs/llada_${TASK}"}
MODEL_PATH=${4:-"GSAI-ML/LLaDA-8B-Instruct"}
MASTER_PORT=${5:-"29700"}
HF_REPO=${6:-""}  # Optional: HF repo for uploading logs
LIMIT=${7:-""}  # Optional: limit number of samples
BATCH_SIZE=1

# Model args - use LLaDA default values
GEN_LENGTH=128
MAX_GEN_LENGTH=128  # Limit max generation length to prevent slow tasks
MC_NUM=128
STEPS=128

MODEL_ARGS="pretrained=${MODEL_PATH},gen_length=${GEN_LENGTH},max_gen_length=${MAX_GEN_LENGTH},steps=${STEPS},mc_num=${MC_NUM}"
if [ -n "$HF_REPO" ]; then
    MODEL_ARGS="${MODEL_ARGS},hf_repo=${HF_REPO},hf_upload=True"
    echo "📤 Hugging Face upload enabled: ${HF_REPO}"
fi

# ============ Environment Setup ============
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export MASTER_PORT=${MASTER_PORT}
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export GLOO_USE_IPV6=0

# ============ Print Configuration ============
echo "======================================"
echo "LLaDA - General Evaluation"
echo "======================================"
echo "GPU(s):        ${GPU_IDS}"
echo "Model Path:    ${MODEL_PATH}"
echo "Task(s):       ${TASK}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Gen Length:    ${GEN_LENGTH}"
echo "MC Samples:    ${MC_NUM}"
echo "Master Port:   ${MASTER_PORT}"
if [ -n "$HF_REPO" ]; then
    echo "HF Upload:     ${HF_REPO}"
fi
echo "======================================"
echo ""

# ============ Run Evaluation ============
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit ${LIMIT}"
    echo "Limiting to ${LIMIT} samples"
fi

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lmms_eval \
  --model llada \
  --model_args ${MODEL_ARGS} \
  --tasks ${TASK} \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_PATH} \
  --log_samples \
  ${LIMIT_ARG} \
  --verbosity INFO

echo ""
echo "======================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "======================================"
