#!/bin/bash
#SBATCH --job-name=EVAL_SIMMIM_CHOLEC80
#SBATCH --output=/home/ppak/outputs/MAE/EVAL_SIMMIM.txt
#SBATCH --error=/home/ppak/outputs/MAE/EVAL_SIMMIM_ERROR.txt
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
python ${home}/surgical_adventure/src/eval_simmim_cholec80.py \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --label_path ${home}/surgical_ncp/train_val_paths_labels1.pkl \
    --overwrite_output_dir \
    --model_name_or_path ./logs/finetune/simmim-cholec80/checkpoint-269825/ \
    --output_dir ./eval/simmim-cholec80-predictions \
    # --label_names bool_masked_pos \
    # --metric_for_best_model eval_loss
    # --label_names pixel_values \
    # --metric_for_best_model test_loss \ 