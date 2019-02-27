#!/bin/bash

#SBATCH --job-name=GDICE
#SBATCH --cpus-per-task=10
#SBATCH --mem=32Gb
#SBATCH --partition=phi

srun python /scratch/slayback.d/GDICE/G-DICE/GDICEPython/generalGDICE.py $1 $2 $3 $4 $5 $6 $7 $8
