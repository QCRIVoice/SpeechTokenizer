#!/bin/bash -l

#SBATCH -J repcodec
#SBATCH -o job.%J.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:A100_80GB:2
#SBATCH --mem 160GB
eval "$(conda shell.bash hook)"
conda activate rep_env
srun run.sh

