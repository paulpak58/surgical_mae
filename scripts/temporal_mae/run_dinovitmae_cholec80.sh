#!/bin/bash
#SBATCH --job-name=RUN_DINOMAE_CHOLEC80
#SBATCH --output=/home/ppak/outputs/MAE/RUN_DINOMAE.txt
#SBATCH --error=/home/ppak/outputs/MAE/RUN_DINOMAE_ERROR.txt
#SBATCH --time=06:00:00
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:volta:2

source /etc/profile
# module load anaconda/2022b
# conda activate mae

home=/home/ppak
# home=/saiil2/paulpak
python3 ${home}/surgical_adventure/src/run_dinovitmae_cholec80.py \
    --output_dir ./logs/pretrain/dinovitmae-cholec80 \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 20 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --model_name_or_path 'facebook/vit-mae-base' \
    --label_path ${home}/surgical_ncp/train_val_paths_labels1.pkl \
    # --metric_for_best_model test_loss \
    # --resume_from_checkpoint ./logs/pretrain/vitmae-cholec80/checkpoint-129516/ \
    # --overwrite_output_dir 