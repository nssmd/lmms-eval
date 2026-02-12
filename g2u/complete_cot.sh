#!/bin/bash
# UniWorld Visual CoT - Complete Evaluation
# Runs all Visual CoT tasks with HF upload

b
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "1" "illusionbench_arshia_visual_cot_split" "./logs/illusionbench_cot" "LanguageBind/UniWorld-V1" "29701" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "2" "VisualPuzzles_visual_cot" "./logs/visualpuzzles_cot" "LanguageBind/UniWorld-V1" "29702" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "3" "realunify_cot" "./logs/realunify_cot" "LanguageBind/UniWorld-V1" "29703" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0" "mmsi_cot" "./logs/mmsi_cot" "LanguageBind/UniWorld-V1" "29709" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "3" "uni_mmmu_cot" "./logs/uni_mmmu_cot" "LanguageBind/UniWorld-V1" "29705" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0" "vsp_cot" "./logs/vsp_cot" "LanguageBind/UniWorld-V1" "29706" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0" "phyx_cot" "./logs/phyx_cot" "LanguageBind/UniWorld-V1" "29712" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "2" "babyvision_cot" "./logs/babyvision_cot" "LanguageBind/UniWorld-V1" "29739" "caes0r/uniworld-results"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "1" "geometry3k_visual_cot" "./logs/geometry3k_visual_cot" "LanguageBind/UniWorld-V1" "29786" "caes0r/uniworld-results" "100"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "3" "auxsolidmath_easy_visual_cot" "./logs/auxsolidmath_easy_visual_cot" "LanguageBind/UniWorld-V1" "29796" "caes0r/uniworld-results" "100"
  bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "1" "uni_mmmu_cot" "./logs/uni_mmmu_cot_full"
  "LanguageBind/UniWorld-V1" "29739" "caes0r/uniworld-results"
  bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "1" "uni_mmmu_maze100_visual_cot"
  "./logs/maze_cot_test" "LanguageBind/UniWorld-V1" "29737"
  bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "1" "uni_mmmu_sliding54_visual_cot"
  "./logs/sliding_cot_test" "LanguageBind/UniWorld-V1" "29738" "" "1"
echo ""
echo "======================================"
echo "All Visual CoT evaluations completed!"
echo "Results uploaded to: caes0r/uniworld-results"
echo "======================================"