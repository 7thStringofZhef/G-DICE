#!/bin/bash

for runDir in "1" "2" "3" "4" "5" "6" "7" "8" "9" "10"
do
    fullDir="/scratch/slayback.d/GDICE/$runDir"
    for envName in "POMDP-1d-v0" "POMDP-4x3-v0" "POMDP-cheese-v0" "POMDP-concert-v0" "POMDP-hallway-v0" "POMDP-heavenhell-v0" "POMDP-loadunload-v0" "POMDP-network-v0" "POMDP-shopping_5-v0" "POMDP-tiger-v0" "POMDP-voicemail-v0"
    do
        sbatch /scratch/slayback.d/GDICE/G-DICE/ShellScripts/main.sh "--save_path" $fullDir "--env_name" $envName
    done
done
