#!/bin/bash
# UniWorld Visual CoT - Complete Evaluation
# Runs all Visual CoT tasks with HF upload

bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0" "chartqa100_visual_cot" "./logs/chartqa_cot" "LanguageBind/UniWorld-V1" "29700" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "1" "illusionbench_arshia_visual_cot_split" "./logs/illusionbench_cot" "LanguageBind/UniWorld-V1" "29701" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "2" "VisualPuzzles_visual_cot" "./logs/visualpuzzles_cot" "LanguageBind/UniWorld-V1" "29702" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "3" "realunify_cot" "./logs/realunify_cot" "LanguageBind/UniWorld-V1" "29703" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0" "mmsi_cot" "./logs/mmsi_cot" "LanguageBind/UniWorld-V1" "29709" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "3" "uni_mmmu_cot" "./logs/uni_mmmu_cot" "LanguageBind/UniWorld-V1" "29705" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0" "vsp_cot" "./logs/vsp_cot" "LanguageBind/UniWorld-V1" "29706" "caes0r/uniworld-results"

echo ""
echo "======================================"
echo "All Visual CoT evaluations completed!"
echo "Results uploaded to: caes0r/uniworld-results"
echo "======================================"