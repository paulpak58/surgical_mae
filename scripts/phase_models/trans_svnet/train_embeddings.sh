#!/bin/bash
#SBATCH --job-name=TRAIN_EMBEDDINGS
#SBATCH --output=/home/ppak/outputs/TRAIN_EMBEDDINGS.txt
#SBATCH --error=/home/ppak/outputs/TRAIN_EMBEDDINGS_ERROR.txt
#SBATCH --time=04:00:00
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem-per-cpu=2500
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

sampling_rate=1.0
training_ratio=1.0
temporal_length=16
num_dataloader_workers_trainer=8
batch_size=2
num_epochs=30
cuda_device=1

python3 ${home}/surgical_adventure/src/Trans-SVNet/train_embedding.py
