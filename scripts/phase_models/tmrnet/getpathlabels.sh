#!/bin/bash
#SBATCH --job-name=TMRNet_PATHLABELS
#SBATCH --output=/home/ppak/outputs/TMRNet_1.txt
#SBATCH --error=/home/ppak/outputs/TMRNet_1_ERROR.txt
#SBATCH --time=00:10:00
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem-per-cpu=2500
#SBATCH --gres=gpu:volta:1


source /etc/profile
module load anaconda/2022b
conda activate py_sdtm_spc

home=/home/ppak
data_dir=${home}/Dataset/cholec80/Video
annotation_dir=${home}/Data/annotations/cholec80_protobuf
log_dir=${home}/surgical_adventure/final_code/lightning_logs/EVAL_MAE_VIT_B/
ckpt_dir=${home}/surgical_adventure/final_code/lightning_logs/MAE_VIT_B/checkpoints/epoch=119-step=17556.ckpt
cache_dir=${home}/cache_mgh/img_cache

python3 ${home}/surgical_adventure/final_code/src/TMRNet/code/Training\ memory\ bank\ model/get_paths_labels.py
