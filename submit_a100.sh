#!/bin/bash -l

#SBATCH -J ST
#SBATCH -o job.%J.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:A100_80GB:3
#SBATCH --mem 240GB
eval "$(conda shell.bash hook)"
. /export/home/vsukhadia/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
bash run_v2.sh


