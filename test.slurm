#! /bin/bash
# ====================================
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# ====================================

source ~/software/init-conda
conda activate pytorch

python text.py