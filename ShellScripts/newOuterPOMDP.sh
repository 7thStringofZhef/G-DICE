#!/bin/bash

fullDir="/scratch/slayback.d/GDICE/"
for jobInd in {1..100..1}
do
    sbatch /scratch/slayback.d/GDICE/G-DICE/ShellScripts/newInnerPhi.sh "--save_path" $fullDir "--env_type" "POMDP" "--set_list" "POMDPsToEval.txt"
done
