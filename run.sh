#!/bin/bash


# Uruchomienie eksperymentu
#./setup_env.sh
source venv/bin/activate
#python -m papermill notebooks/p_diversity/1.1.with_changed_loss_0.2_adult.ipynb outputs/output_adult_2.ipynb
#python -m papermill notebooks/p_diversity/1.1.with_changed_loss_0.2_german_credit.ipynb outputs/output_german_credt_cuda.ipynb
#python -m papermill notebooks/p_diversity/german_credit_cuda.ipynb outputs/cred_cuda.ipynb
#python -m papermill notebooks/p_diversity/adult_cuda.ipynb outputs/adult_cuda.ipynb
#python -m papermill notebooks/p_diversity/give_credit_cpu.ipynb outputs/give_credit.ipynb

#python notebooks/p_diversity/give_credit.py
python -m papermill notebooks/p_diversity/adult_cuda.ipynb outputs/adult_cuda.ipynb


