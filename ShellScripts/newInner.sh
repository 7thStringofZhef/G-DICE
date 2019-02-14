#!/bin/bash

#SBATCH --job-name=GDICE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32Gb
#SBATCH --partition=general

srun python /scratch/slayback.d/GDICE/G-DICE/GDICEPython/generalGDICE.py $1 $2 $3 $4 $5 $6