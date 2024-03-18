#!/bin/bash
#SBATCH --job-name=MATLAB_EVAL_TRANS_SVNET
#SBATCH --output=/home/ppak/outputs/MATLAB_TRANSV.txt
#SBATCH --error=/home/ppak/outputs/MATLAB_TRANSV_ERROR.txt
#SBATCH --time=00:30:00
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem-per-cpu=2500
#SBATCH --gres=gpu:volta:1

script_dir=/home/ppak/surgical_adventure/final_code/src/matlab-eval/Main.m
matlab -nodisplay -nosplash -nodesktop -r "run('$script_dir');exit;" | tail -n +11