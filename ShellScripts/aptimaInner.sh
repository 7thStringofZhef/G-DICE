#!/bin/bash

#SBATCH --job-name=GDICE
#SBATCH --cpus-per-task=10
#SBATCH --mem=32Gb
#SBATCH --partition=general

srun python /scratch/slayback.d/GDICE/G-DICE/GDICEPython/aptimaGDICE.py