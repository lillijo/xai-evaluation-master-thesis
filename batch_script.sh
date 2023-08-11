#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=bd1083
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --exclusive

source ~/.bashrc
conda activate mt-cause-XAI
python3 train_model.py