#!/bin/bash
#SBATCH --job-name=LFB
#SBATCH --output=/home/ppak/outputs/LFB_L30.txt
#SBATCH --error=/home/ppak/outputs/LFB_L30_ERROR.txt
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --mem-per-cpu=6400
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022b
conda activate surgical_adventure

home=/home/ppak
python3 ${home}/surgical_adventure/src/Trans-SVNet/generate_LFB.py -s 10
