#!/bin/bash
#SBATCH --job-name=IMN
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -p rtx4090

# List of dataset IDs as arguments to this script

nvidia-smi -L
echo "Started"

singularity exec --nv -B ~/counterfactuals:/counterfactuals /shared/sets/singularity/miniconda_pytorch_py311.sif ./run.sh

