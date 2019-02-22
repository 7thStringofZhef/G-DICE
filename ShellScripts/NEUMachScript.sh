#!/bin/bash

fullDir="/home/david/GDICERes/Entropy/"
python /home/david/NortheasternStuff/Research/G-DICE/GDICEPython/generalGDICE.py "--save_path" $fullDir "--env_type" "POMDP" "--set_list" "EntPOMDPsToEval.txt"