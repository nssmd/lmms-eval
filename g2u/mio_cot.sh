#!/bin/bash
# MIO Visual Chain-of-Thought Model - Evaluation Script
#
# This script evaluates MIO-7B with Visual CoT (two-stage reasoning)
# Stage 1: Generate visualization image from text prompt (with original image)
# Stage 2: Answer question using both original and generated images
#
# Usage:
#   bash mio_cot.sh [GPU_IDS] [TASK] [OUTPUT_PATH] [MODEL_PATH] [MASTER_PORT] [HF_REPO] [LIMIT]
#
# Examples:
#   # ChartQA Visual CoT (100 samples)
#   bash mio_cot.sh "0" "chartqa100_visual_cot" "./logs/mio_cot_chartqa"
#
#   # MathVista Visual CoT
#   bash mio_cot.sh "0" "mathvista_visual_cot" "./logs/mio_cot_mathvista"
#
#   # Uni-MMMU Jigsaw Visual CoT
#   bash mio_cot.sh "0" "uni_mmmu_jigsaw100_visual_cot" "./logs/mio_cot_jigsaw"
#
#   # Multiple GPUs
#   bash mio_cot.sh "0,1" "chartqa100_visual_cot" "./logs/mio_cot"
#
#   # With custom model and port
#   bash mio_cot.sh "0" "chartqa100_visual_cot" "./logs/test" "m-a-p/MIO-7B-Instruct" "29605"
#
#   # With HuggingFace upload
#   bash mio_cot.sh "0" "chartqa100_visual_cot" "./logs/test" "m-a-p/MIO-7B-Instruct" "29603" "username/repo-name"
#
#   # With limit (test on 10 samples)
#   bash mio_cot.sh "0" "chartqa100_visual_cot" "./logs/test" "m-a-p/MIO-7B-Instruct" "29603" "" "10"

# ============ Configuration ============
GPU_IDS=${1:-"0"}
TASK=${2:-"chartqa100_visual_cot"}
OUTPUT_PATH=${3:-"./logs/mio_cot_${TASK}"}
MODEL_PATH=${4:-"m-a-p/MIO-7B-Instruct"}
MASTER_PORT_ARG=${5:-"29603"}
HF_REPO=${6:-""}  # Optional: HuggingFace repo to upload logs
LIMIT=${7:-""}    # Optional: Limit number of samples for testing
BATCH_SIZE=1

# Build limit argument if provided
LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then
  LIMIT_ARG="--limit ${LIMIT}"
fi

# ============ Check MIO Repository ============
if [ ! -d "../MIO" ] && [ ! -d "MIO" ]; then
    echo "❌ Error: MIO repository not found!"
    echo ""
    echo "Please clone MIO repository:"
    echo "  cd .."
    echo "  git clone https://github.com/MIO-Team/MIO.git"
    echo "  cd MIO"
    echo "  pip install -r requirements.txt"
    echo "  cd ../lmms-eval"
    exit 1
fi

echo "✅ MIO repository found"

# ============ Model Arguments ============
# MIO Visual CoT config
MODEL_ARGS="pretrained=${MODEL_PATH}"

# Stage 1: Image generation parameters
MODEL_ARGS="${MODEL_ARGS},stage1_cfg_scale=5.0"
MODEL_ARGS="${MODEL_ARGS},stage1_num_inference_steps=50"
MODEL_ARGS="${MODEL_ARGS},stage1_guidance_scale=7.5"
MODEL_ARGS="${MODEL_ARGS},stage1_image_size=512"

# Stage 2: Visual understanding parameters
MODEL_ARGS="${MODEL_ARGS},stage2_max_new_tokens=512"
MODEL_ARGS="${MODEL_ARGS},stage2_temperature=0.0"
MODEL_ARGS="${MODEL_ARGS},stage2_do_sample=False"
MODEL_ARGS="${MODEL_ARGS},stage2_num_beams=1"
MODEL_ARGS="${MODEL_ARGS},stage2_top_p=0.9"
MODEL_ARGS="${MODEL_ARGS},stage2_repetition_penalty=1.0"

# Output and debugging
MODEL_ARGS="${MODEL_ARGS},save_intermediate=true"
MODEL_ARGS="${MODEL_ARGS},intermediate_dir=${OUTPUT_PATH}/artifacts"
MODEL_ARGS="${MODEL_ARGS},fail_gracefully=true"

# ============ Environment Setup ============
# Fix libstdc++ version issue
export LD_PRELOAD=/opt/conda/envs/ptca/lib/libstdc++.so.6:${LD_PRELOAD}
export LD_LIBRARY_PATH=/home/aiscuser/cuda_compat:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export MASTER_PORT=${MASTER_PORT_ARG}
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1

# ============ Print Configuration ============
echo "======================================"
echo "MIO Visual CoT - Evaluation"
echo "======================================"
echo "GPU(s):        ${GPU_IDS}"
echo "Model Path:    ${MODEL_PATH}"
echo "Task(s):       ${TASK}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Model Args:    ${MODEL_ARGS}"
echo "Master Port:   ${MASTER_PORT}"
if [ -n "${HF_REPO}" ]; then
    echo "HF Upload:     ${HF_REPO}"
fi
if [ -n "${LIMIT}" ]; then
    echo "Limit:         ${LIMIT}"
fi
echo "======================================"
echo ""
echo "Two-Stage Visual CoT Process:"
echo "  Stage 1: Generate visualization from prompt + original image"
echo "  Stage 2: Answer with original image + generated visualization"
echo "======================================"
echo ""

# ============ Run Evaluation ============
python -m lmms_eval \
  --model mio_cot \
  --model_args ${MODEL_ARGS} \
  --tasks ${TASK} \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_PATH} \
  --log_samples \
  --log_samples_suffix mio_cot_${TASK} \
  --verbosity INFO \
  ${LIMIT_ARG}

echo ""
echo "======================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "Intermediate artifacts: ${OUTPUT_PATH}/artifacts"
echo "======================================"

# ============ Upload to HuggingFace (Optional) ============
if [ -n "${HF_REPO}" ]; then
    echo ""
    echo "======================================"
    echo "Uploading logs to HuggingFace..."
    echo "Repository: ${HF_REPO}"
    echo "======================================"

    # Check if huggingface_hub is installed
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo "⚠️  Warning: huggingface_hub not installed. Skipping upload."
        echo "Install with: pip install huggingface_hub"
    else
        # Upload the entire output directory to HuggingFace
        python -c "
from huggingface_hub import HfApi
import os

api = HfApi()
output_path = '${OUTPUT_PATH}'
repo_id = '${HF_REPO}'
task_name = '${TASK}'

# Upload all files in the output directory
try:
    api.upload_folder(
        folder_path=output_path,
        repo_id=repo_id,
        path_in_repo=f'logs/{task_name}',
        repo_type='dataset',
        commit_message=f'Upload MIO Visual CoT evaluation logs for {task_name}'
    )
    print(f'✅ Successfully uploaded logs to {repo_id}')
except Exception as e:
    print(f'❌ Failed to upload: {e}')
"
    fi
fi
