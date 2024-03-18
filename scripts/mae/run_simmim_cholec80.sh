#!/bin/bash
#SBATCH --job-name=RUN_SIMMIM_CHOLEC80
#SBATCH --output=/home/ppak/outputs/MAE/RUN_SIMMIM.txt
#SBATCH --error=/home/ppak/outputs/MAE/RUN_SIMMIM_ERROR.txt
#SBATCH --time=06:00:00
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:volta:2

source /etc/profile
# module load anaconda/2022b
# conda activate surgical_adventure

home=/home/ppak
# home=/saiil2/paulpak
python ${home}/surgical_adventure/src/run_simmim_cholec80.py \
    --output_dir ./logs/pretrain/simmim-cholec80 \
    --remove_unused_columns False \
    --label_names bool_masked_pos \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --model_name_or_path 'google/vit-base-patch16-224-in21k' \
    --label_path ${home}/surgical_ncp/train_val_paths_labels1.pkl \
    --resume_from_checkpoint ./logs/pretrain/simmim-cholec80/checkpoint-172688/ \
    --overwrite_output_dir
    # --model_name_or_path 'microsoft/swin-base-simmim-window6-192' \