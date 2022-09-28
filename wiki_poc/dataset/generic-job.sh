#!/bin/bash
#SBATCH --job-name="generic-job"
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preempt
#SBATCH --gres=gpu:rtx3090
#SBATCH --mail-user=alex.nyffenegger@outlook.com

# generic bash script to quickly run jobs on ubelix
# without creating a new script every time

script=""

if [ -z $script ] || [ ! -f $script ]; then
    echo "Script $script not found!"
    exit
fi

module load Anaconda3
module load Workspace
module load CUDA
eval "$(conda shell.bash hook)"
conda activate standard-nlp


srun python3 $script
