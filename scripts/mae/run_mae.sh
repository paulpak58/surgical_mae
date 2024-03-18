#!/bin/bash
#SBATCH --job-name=RUN_MAE_CIFAR10
#SBATCH --output=/home/ppak/outputs/MAE/RUN_MAE.txt
#SBATCH --error=/home/ppak/outputs/MAE/RUN_MAE_ERROR.txt
#SBATCH --time=00:30:00
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem-per-cpu=4500
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022b
conda activate surgical_adventure

home=/home/ppak
python ${home}/surgical_adventure/src/run_mae.py