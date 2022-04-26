#!/bin/bash
#SBATCH --job-name="wiki-top-50-predicitions"
#SBATCH --time=00:59:00
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preempt
#SBATCH --gres=gpu:rtx3090
#SBATCH --mail-user=alex.nyffenegger@outlook.com

# Your code below this line
module load Anaconda3
module load Workspace
module load CUDA
eval "$(conda shell.bash hook)"
conda activate standard-nlp

srun python3 wiki-demasking.py
