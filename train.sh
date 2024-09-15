#/usr/bin/env bash

#SBATCH --job-name=melchior_train
#SBATCH --account-name=elonlit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=0-01:00:00

# Load the required modules
module load python/3.8
module load cuda/11.1
module load cudnn/8.0.5

# Create and activate the virtual environment
source setup.sh

# Run the training script
python -m utils.train


