#!/bin/bash
#SBATCH --job-name=RUN_VIDEOMAE_CHOLEC80
#SBATCH --output=/home/ppak/outputs/TEMPORAL_MAE/RUN_VIDEOMAE.txt
#SBATCH --error=/home/ppak/outputs/TEMPORAL_MAE/RUN_VIDEOMAE_ERROR.txt
#SBATCH --time=10:00:00
#SBATCH -n 128
#SBATCH -N 4
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:volta:2

source /etc/profile
# module load anaconda/2022b
# conda activate mae

home=/home/ppak
# home=/saiil2/paulpak
python3 ${home}/surgical_adventure/src/run_videomae_cholec80.py \
    --output_dir ./logs/pretrain/videomae-cholec80 \
    --remove_unused_columns False \
    --label_names bool_masked_pos \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 128 \
    --eval_accumulation_steps 128 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --warmup_ratio 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --num_train_epochs 30 \
    --model_name_or_path 'MCG-NJU/videomae-base' \
    --label_path ${home}/surgical_ncp/train_val_paths_labels1.pkl \
    --overwrite_output_dir
    # --base_learning_rate 1.5e-4 \
    # --seed 1337 \

    # --mask_ratio 0.90 \
    #--resume_from_checkpoint ./logs/pretrain/vitmae-cholec80/checkpoint-129516/ \
    #----------------------------------------------------------------
    #--norm_pix_loss \