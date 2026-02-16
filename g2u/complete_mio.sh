#!/bin/bash
# Complete evaluation script for all tasks
# Each task runs on a different GPU with a unique port

bash /home/aiscuser/lmms-eval/g2u/mio.sh "0" "chartqa100" "./logs/chartqa" "m-a-p/MIO-7B-Instruct" "29602"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "0" "illusionbench_arshia_test" "./logs/illusionbench" "m-a-p/MIO-7B-Instruct" "29603"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "4" "VisualPuzzles" "./logs/VisualPuzzles" "m-a-p/MIO-7B-Instruct" "29604"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "5" "realunify" "./logs/realunify" "m-a-p/MIO-7B-Instruct" "29605"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "4" "mmsi" "./logs/mmsi" "m-a-p/MIO-7B-Instruct" "29606"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "7" "uni_mmmu" "./logs/uni_mmmu" "m-a-p/MIO-7B-Instruct" "29607"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "7" "vsp" "./logs/vsp" "m-a-p/MIO-7B-Instruct" "29608"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "7" "babyvision" "./logs/babyvision" "m-a-p/MIO-7B-Instruct" "29609"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "6" "phyx" "./logs/phyx" "m-a-p/MIO-7B-Instruct" "29610"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "4" "auxsolidmath_easy" "./logs/auxsolidmath_easy" "m-a-p/MIO-7B-Instruct" "29611"
bash /home/aiscuser/lmms-eval/g2u/mio.sh "4" "geometry3k" "./logs/geometry3k" "m-a-p/MIO-7B-Instruct" "29612 " "" "100"