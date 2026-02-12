#!/bin/bash
# Qwen-Image-Edit Visual CoT - Complete Evaluation Script
# Runs multiple Visual CoT tasks with Qwen-Image-Edit model
#
# Two-stage inference for each task:
# 1. Stage 1: Generate visualization using Qwen-Image-Edit
# 2. Stage 2: Answer question using Qwen2.5-VL

echo "======================================"
echo "Qwen-Image-Edit Visual CoT"
echo "Complete Evaluation"
echo "Running multiple Visual CoT tasks..."
echo "======================================"
echo ""

# Model paths
MODEL_EDIT="Qwen/Qwen-Image-Edit"
MODEL_UNDERSTAND="Qwen/Qwen2.5-VL-7B-Instruct"

# ChartQA - Chart understanding with visual CoT
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "0" "chartqa100_visual_cot" "./logs/qwen_cot/chartqa" "Qwen/Qwen-Image-Edit" "Qwen/Qwen2.5-VL-7B-Instruct" "39892"
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "1" "mmsi_cot" "./logs/qwen_cot/mmsi_cot"  "Qwen/Qwen-Image-Edit" "Qwen/Qwen2.5-VL-7B-Instruct" "29699" 
# AuxSolidMath - Solid geometry with visual CoT (limited to 100 samples)
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "2" "auxsolidmath_easy_visual_cot" "./logs/qwen_cot/auxsolidmath" "Qwen/Qwen-Image-Edit" "Qwen/Qwen2.5-VL-7B-Instruct" "29702" "" "100"
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "2" "illusionbench_arshia_visual_cot_split" "./logs/qwen_cot/illusionbench_cot" "Qwen/Qwen-Image-Edit" "Qwen/Qwen2.5-VL-7B-Instruct" "29741" 
# VisualPuzzles - Visual reasoning puzzles with CoT
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "3" "VisualPuzzles_visual_cot" "./logs/qwen_cot/visualpuzzles" "Qwen/Qwen-Image-Edit" "Qwen/Qwen2.5-VL-7B-Instruct" "29703"
  cd /home/aiscuser/lmms-eval && \
  bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh \"0" \
    "chartqa100_visual_cot" \
    "./logs/qwen_cot/chartqa" \
    "Qwen/Qwen-Image-Edit" \
    "Qwen/Qwen2.5-VL-7B-Instruct" \
    "39890"
# Uni-MMMU tasks with visual CoT
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "4" "maze100_visual_cot" "./logs/qwen_cot/maze" "Qwen/Qwen-Image-Edit" "Qwen/Qwen2.5-VL-7B-Instruct" "29704"
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "3" "uni_mmmu_jigsaw100_visual_cot" "./logs/qwen_cot/jigsaw" "Qwen/Qwen-Image-Edit" "Qwen/Qwen2.5-VL-7B-Instruct" "29705"
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "3" "uni_mmmu_sliding54_visual_cot" "./logs/qwen_cot/sliding" "Qwen/Qwen-Image-Edit" "Qwen/Qwen2.5-VL-7B-Instruct" "29706"

# VSP - Visual spatial planning with CoT
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "7" "collision_visual_cot" "./logs/qwen_cot/collision" "${MODEL_EDIT}" "${MODEL_UNDERSTAND}" "29707"
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "0" "google_map_visual_cot" "./logs/qwen_cot/google_map" "${MODEL_EDIT}" "${MODEL_UNDERSTAND}" "29708"

# Phyx - Physics reasoning with visual CoT (limited to 100 samples each)
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "1" "phyx_mechanics100_visual_cot" "./logs/qwen_cot/phyx_mechanics" "${MODEL_EDIT}" "${MODEL_UNDERSTAND}" "29709" "" "100"
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "2" "phyx_optics100_visual_cot" "./logs/qwen_cot/phyx_optics" "${MODEL_EDIT}" "${MODEL_UNDERSTAND}" "29710" "" "100"

# RealUnify - Real-world understanding with CoT
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "3" "mental_reconstruction_visual_cot" "./logs/qwen_cot/mental_reconstruction" "${MODEL_EDIT}" "${MODEL_UNDERSTAND}" "29711"
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "4" "mental_tracking_visual_cot" "./logs/qwen_cot/mental_tracking" "${MODEL_EDIT}" "${MODEL_UNDERSTAND}" "29712"

# BabyVision - Visual perception with CoT
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "5" "fine_grained_discrimination_visual_cot" "./logs/qwen_cot/fine_grained" "${MODEL_EDIT}" "${MODEL_UNDERSTAND}" "29713"
bash /home/aiscuser/lmms-eval/g2u/qwen_visual_cot.sh "6" "visual_tracking_visual_cot" "./logs/qwen_cot/visual_tracking" "${MODEL_EDIT}" "${MODEL_UNDERSTAND}" "29714"

echo ""
echo "======================================"
echo "All Qwen Visual CoT evaluations completed!"
echo "Results saved to: ./logs/qwen_cot/"
echo "======================================"