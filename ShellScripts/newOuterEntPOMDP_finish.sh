#!/bin/bash

fullDir="/scratch/slayback.d/GDICE/Entropy/"
for jobInd in {1..100..1}
do
    sbatch /scratch/slayback.d/GDICE/G-DICE/ShellScripts/newInner.sh "--save_path" $fullDir "--env_type" "POMDP" "--set_list" "EntPOMDPsToEval.txt" "--unfinished" "1"
done
