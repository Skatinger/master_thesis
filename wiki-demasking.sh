#!/bin/bash
#SBATCH --job-name="demask-wiki-dataset"
#SBATCH --time 10:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preempt
#SBATCH --gres=gpu:rtx3090
#SBATCH --mail-user=alex.nyffenegger@outlook.com
#SBATCH --mail-type=end,fail

# Your code below this line
module load Anaconda3
module load Workspace
module load CUDA
eval "$(conda shell.bash hook)"
conda activate standard-nlp

srun python3 wiki-demasking.py 0
