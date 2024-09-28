#!/bin/bash
#SBATCH --job-name=melchior_train
#SBATCH --output=logs/melchior_train_%j.out
#SBATCH --error=logs/melchior_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Load any necessary modules
module load python/3.8
module load cuda/11.3

./setup.sh

# Run your training script
python train.py --model melchior --epochs 20 --batch_size 64 --num_gpus 4