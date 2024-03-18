#!/bin/bash
#SBATCH --job-name=TransSV
#SBATCH --output=/home/ppak/outputs/EVAL_TransSV.txt
#SBATCH --error=/home/ppak/outputs/EVAL_TransSV_ERROR.txt
#SBATCH --time=24:00:00
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022b
conda activate surgical_adventure

home=/home/ppak
data_dir=${home}/Dataset/cholec80/Video
annotation_dir=${home}/Data/annotations/cholec80_protobuf
log_dir=${home}/surgical_adventure/final_code/lightning_logs/EVAL_MAE_VIT_B/
ckpt_dir=${home}/surgical_adventure/final_code/lightning_logs/MAE_VIT_B/checkpoints/epoch=119-step=17556.ckpt
cache_dir=${home}/cache_mgh/img_cache

python3 ${home}/surgical_adventure/src/Trans-SVNet/eval/python/test_trans_svnet.py --name trans_svnet
