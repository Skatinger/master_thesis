#!/bin/bash
#SBATCH --job-name="paraphrase-wiki-dataset"
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preempt
#SBATCH --gres=gpu:rtx3090
#SBATCH --mail-user=alex.nyffenegger@outlook.com
#SBATCH --mail-type=end,fail
#SBATCH --array=0-3

# Your code below this line
module load Anaconda3
module load Workspace
module load CUDA
eval "$(conda shell.bash hook)"
conda activate standard-nlp

# start 4 jobs in parallel, each with a different shard to process
srun python build-paraphrased-large.py ${SLURM_ARRAY_TASK_ID}
