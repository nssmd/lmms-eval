#!/bin/bash
# LLaDA Complete Evaluation Script
# Runs multiple evaluation tasks sequentially
#
# Usage:
#   bash complete_llada.sh [LIMIT]
#
# Examples:
#   bash complete_llada.sh 100  # Test only first 100 samples per task
#   bash complete_llada.sh      # Test all samples

LIMIT=${1:-""}  # Optional: limit number of samples per task

# ChartQA
bash /home/aiscuser/lmms-eval/g2u/llada.sh "2" "chartqa100" "./logs/llada_chartqa" "GSAI-ML/LLaDA-8B-Instruct" "29700" "" "${LIMIT}"

# IllusionBench
bash /home/aiscuser/lmms-eval/g2u/llada.sh "2" "illusionbench_arshia_test" "./logs/llada_illusionbench" "GSAI-ML/LLaDA-8B-Instruct" "29701" "" "${LIMIT}"

# VisualPuzzles
bash /home/aiscuser/lmms-eval/g2u/llada.sh "3" "VisualPuzzles" "./logs/llada_visualpuzzles" "GSAI-ML/LLaDA-8B-Instruct" "29702" "" "${LIMIT}"

# RealUnify
bash /home/aiscuser/lmms-eval/g2u/llada.sh "2" "realunify" "./logs/llada_realunify" "GSAI-ML/LLaDA-8B-Instruct" "29703" "" "${LIMIT}"

# MMSI
bash /home/aiscuser/lmms-eval/g2u/llada.sh "3" "mmsi" "./logs/llada_mmsi" "GSAI-ML/LLaDA-8B-Instruct" "29704" "" "${LIMIT}"

# Uni-MMMU
bash /home/aiscuser/lmms-eval/g2u/llada.sh "2" "uni_mmmu" "./logs/llada_uni_mmmu" "GSAI-ML/LLaDA-8B-Instruct" "29705" "" "${LIMIT}"

# VSP
bash /home/aiscuser/lmms-eval/g2u/llada.sh "3" "vsp" "./logs/llada_vsp" "GSAI-ML/LLaDA-8B-Instruct" "29706" "" "${LIMIT}"
bash /home/aiscuser/lmms-eval/g2u/llada.sh "1" "vsp" "./logs/llada_vsp" "GSAI-ML/LLaDA-8B-Instruct" "29706" "" "${LIMIT}"
bash /home/aiscuser/lmms-eval/g2u/llada.sh "0" "babyvision" "./logs/babyvision" "GSAI-ML/LLaDA-8B-Instruct" "29706" ""
bash /home/aiscuser/lmms-eval/g2u/llada.sh "1" "phyx_simple" "./logs/phyx_simple" "GSAI-ML/LLaDA-8B-Instruct" "29707" ""
bash /home/aiscuser/lmms-eval/g2u/llada.sh "2" "geometry3k" "./logs/geometry3k" "GSAI-ML/LLaDA-8B-Instruct" "29708" "" "100"
bash /home/aiscuser/lmms-eval/g2u/llada.sh "3" "auxsolidmath_easy" "./logs/auxsolidmath_easy" "GSAI-ML/LLaDA-8B-Instruct" "29709" "" "100"


echo ""
echo "======================================"
echo "All LLaDA evaluations completed!"
echo "======================================"