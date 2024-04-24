#!/bin/bash
#SBATCH --job-name=ppcef-mlp-digits
#SBATCH --time=0-3:00:00 # dni-godziny:minuty:sekundy
#SBATCH --nodes=1 # ilosc nodow - duzy ruch sprawia ze lepiej czasem brac kilka nodeow po 1-2 gpu bo sa kolejki
#SBATCH --ntasks-per-node=1
#SBATCH --account=plgprobregtabdata-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --gres=gpu:0 # ilość gpu per node ( jak wezmiesz 4 gpu i 2 nodes to bedzie 8 gpu)

cd /net/tscratch/people/plgofurman/counterfactuals
# potencjalny wybor venva
source venv1/bin/activate
srun python run_train_and_experiment_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.AuditDataset \
    disc_model=node \
    gen_model=large_maf \
    counterfactuals.disc_loss._target_=counterfactuals.losses.MulticlassDiscLoss

# BinaryDiscLoss
# MulticlassDiscLoss


# srun python run_train_and_experiment_cv.py --multirun \
#         dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
# counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AuditDataset,\
# counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \
#     gen_model=medium_maf \
#     disc_model=mlr,mlp

# srun --job-name=interctive_nodeflow\
#     --nodes=1\
#     --ntasks-per-node=1\
#     --account=plgprobregtabdata-gpu-a100\
#     --partition=plgrid-gpu-a100\
#     --cpus-per-task=32\
#     --mem=30G\
#     --time=0-1:00:00\
#     --gres=gpu:0\
#     --ntasks=1 --pty /bin/bash
