#!/bin/bash

fullDir="/scratch/slayback.d/GDICE/"
for runInd in {1..4..1}
do
    for envInd in {0..2..1}
    do
        for paramInd in {0..5..1}
        do
            sbatch /scratch/slayback.d/GDICE/G-DICE/ShellScripts/aptimaInner.sh "--save_path" $fullDir "--env_num" $envInd "--param_num" $paramInd "--run_num" $runInd
        done
    done
done