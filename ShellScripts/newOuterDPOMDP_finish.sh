#!/bin/bash

fullDir="/scratch/slayback.d/GDICE/"
for jobInd in {1..100..1}
do
    sbatch /scratch/slayback.d/GDICE/G-DICE/ShellScripts/newInner.sh "--save_path" $fullDir "--env_type" "DPOMDP" "--set_list" "DPOMDPsToEval.txt" "--unfinished" "1"
done
