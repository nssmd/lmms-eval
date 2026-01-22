#!/bin/bash
# Complete MIO Visual CoT evaluation script for all tasks
# Each task runs on a different GPU with a unique port
#
# Usage:
#   bash complete_mio_cot.sh
#
# Tasks:
#   - chartqa100_visual_cot: Chart question answering with visual CoT
#   - mathvista_visual_cot: Mathematical visual reasoning
#   - uni_mmmu_jigsaw100_visual_cot: Jigsaw puzzle solving
#   - Add more visual CoT tasks as needed

echo "======================================"
echo "MIO Visual CoT - Complete Evaluation"
echo "======================================"
echo "Starting evaluation on all Visual CoT tasks..."
echo ""

# ChartQA Visual CoT - GPU 0
echo "Task 1/7: ChartQA Visual CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "0" "chartqa100_visual_cot" "./logs/mio_cot/chartqa" "m-a-p/MIO-7B-Instruct" "29700" "caes0r/mio"

# IllusionBench Visual CoT - GPU 1
echo "Task 2/7: IllusionBench Visual CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "3" "illusionbench_arshia_visual_cot_split" "./logs/mio_cot/illusionbench" "m-a-p/MIO-7B-Instruct" "29701" "caes0r/mio"

# VisualPuzzles Visual CoT - GPU 2
echo "Task 3/7: VisualPuzzles Visual CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "4" "VisualPuzzles_visual_cot" "./logs/mio_cot/visualpuzzles" "m-a-p/MIO-7B-Instruct" "29702" "caes0r/mio"

# RealUnify CoT - GPU 3
echo "Task 4/7: RealUnify CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "2" "realunify_cot" "./logs/mio_cot/realunify" "m-a-p/MIO-7B-Instruct" "29703" "caes0r/mio"

# MMSI CoT - GPU 0
echo "Task 5/7: MMSI CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "5" "mmsi_cot" "./logs/mio_cot/mmsi" "m-a-p/MIO-7B-Instruct" "29709" "caes0r/mio"

# Uni-MMMU CoT - GPU 3
echo "Task 6/7: Uni-MMMU CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "6" "uni_mmmu_cot" "./logs/mio_cot/uni_mmmu" "m-a-p/MIO-7B-Instruct" "29705" "caes0r/mio"

# VSP CoT - GPU 0
echo "Task 7/7: VSP CoT"
bash /home/aiscuser/lmms-eval/g2u/mio_cot.sh "7" "vsp_cot" "./logs/mio_cot/vsp" "m-a-p/MIO-7B-Instruct" "29706" "caes0r/mio"

echo ""
echo "======================================"
echo "All evaluations completed!"
echo "Results saved to: ./logs/mio_cot/"
echo "======================================"
