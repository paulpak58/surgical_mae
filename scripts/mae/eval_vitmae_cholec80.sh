#!/bin/bash
#SBATCH --job-name=EVAL_MAE_CHOLEC80
#SBATCH --output=/home/ppak/outputs/MAE/EVAL_MAE.txt
#SBATCH --error=/home/ppak/outputs/MAE/EVAL_MAE_ERROR.txt
#SBATCH --time=00:45:00
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:volta:1

source /etc/profile
# module load anaconda/2022b
# conda activate surgical_adventure

home=/home/ppak
# home=/saiil2/paulpak
python ${home}/surgical_adventure/src/eval_vitmae_cholec80.py \
    --remove_unused_columns False \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 50 \
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
    --model_name_or_path ./logs/finetune/vitmae-cholec80/checkpoint-269825/ \
    --label_path ${home}/surgical_ncp/train_val_paths_labels1.pkl \
    --overwrite_output_dir \
    --output_dir ./eval/vitmae-cholec80-predictions \
    # --metric_for_best_model accuracy \
    # --label_names pixel_values \
    # --metric_for_best_model test_loss \ 