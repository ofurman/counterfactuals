#!/bin/bash
#SBATCH --job-name=cegp-mlp-digits
#SBATCH --time=0-48:00:00 # dni-godziny:minuty:sekundy
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
srun python alternative_methods/cegp_exp_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    disc_model=mlr \
    gen_model=medium_maf

# counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
# counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AuditDataset,\
# counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \

# srun --job-name=interctive_nodeflow\
#     --nodes=1\
#     --ntasks-per-node=1\
#     --account=plgprobregtabdata-gpu-a100\
#     --partition=plgrid-gpu-a100\
#     --cpus-per-task=32\
#     --mem=30G\
#     --time=0-1:00:00\
#     --gres=gpu:1\
#     --ntasks=1 --pty /bin/bash