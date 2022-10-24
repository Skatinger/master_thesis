#!/bin/bash
#SBATCH --job-name="build-unparaphrased-large-wiki-dataset"
#SBATCH --time=00:03:00
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu
#SBATCH --qos=job_epyc2
#SBATCH --mail-user=alex.nyffenegger@outlook.com

# Your code below this line
module load Anaconda3
module load Workspace
eval "$(conda shell.bash hook)"
conda activate standard-nlp

srun python3 build-unparaphrased-large.py
