#!/bin/bash
# Qwen-Image-Edit Visual CoT - General Evaluation Script
#
# This script evaluates Qwen-Image-Edit with Visual Chain-of-Thought on any task
# Two-stage inference:
# 1. Stage 1: Generate visualization using Qwen-Image-Edit
# 2. Stage 2: Answer question using Qwen2.5-VL
#
# Usage:
#   bash qwen_visual_cot.sh [GPU_IDS] [TASK] [OUTPUT_PATH] [MODEL_EDIT] [MODEL_UNDERSTAND] [MASTER_PORT] [RESERVED] [LIMIT]
#
# Examples:
#   # ChartQA Visual CoT
#   bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "0" "chartqa100_visual_cot" "./logs/qwen_cot/chartqa"
#
#   # MathVista Visual CoT
#   bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "0" "mathvista_visual_cot" "./logs/qwen_cot/mathvista"
#
#   # With limit (test only 100 samples)
#   bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "0" "auxsolidmath_easy_visual_cot" "./logs/qwen_cot/auxsolidmath" "Qwen/Qwen-Image-Edit" "Qwen/Qwen2-VL-7B-Instruct" "29700" "" "100"

# ============ Configuration ============
GPU_IDS=${1:-"0"}
TASK=${2:-"chartqa100_visual_cot"}
OUTPUT_PATH=${3:-"./logs/qwen_cot_${TASK}"}
MODEL_EDIT=${4:-"Qwen/Qwen-Image-Edit"}
MODEL_UNDERSTAND=${5:-"Qwen/Qwen2.5-VL-7B-Instruct"}
MASTER_PORT=${6:-"29700"}
RESERVED=${7:-""}
LIMIT=${8:-""}
BATCH_SIZE=1

# Model arguments for Qwen-Image-Edit Visual CoT
# Stage 1: Image editing/generation parameters
# Stage 2: Visual understanding parameters
MODEL_ARGS="pretrained_edit=${MODEL_EDIT},pretrained_understand=${MODEL_UNDERSTAND},stage1_num_inference_steps=50,stage1_guidance_scale=7.5,stage2_max_new_tokens=512,stage2_temperature=0.0,stage2_do_sample=false,save_intermediate=true"

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

# Fix libstdc++ version issue - use conda's libraries
export LD_LIBRARY_PATH=/opt/conda/envs/ptca/lib:$LD_LIBRARY_PATH

# ============ Print Configuration ============
echo "======================================"
echo "Qwen-Image-Edit Visual CoT"
echo "======================================"
echo "GPU(s):        ${GPU_IDS}"
echo "Edit Model:    ${MODEL_EDIT}"
echo "Understand:    ${MODEL_UNDERSTAND}"
echo "Task(s):       ${TASK}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Master Port:   ${MASTER_PORT}"
if [ -n "${LIMIT}" ]; then
  echo "Limit:         ${LIMIT}"
fi
echo "======================================"
echo ""

# ============ Run Evaluation ============
# Build limit argument if provided
LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then
  LIMIT_ARG="--limit ${LIMIT}"
fi

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lmms_eval \
  --model qwen_image_edit_visual_cot \
  --model_args ${MODEL_ARGS} \
  --tasks ${TASK} \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_PATH} \
  --log_samples \
  --verbosity INFO \
  ${LIMIT_ARG}

echo ""
echo "======================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "======================================"