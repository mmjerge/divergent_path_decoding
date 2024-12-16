#!/bin/bash
 
#SBATCH --job-name="counterfactual_analysis"
#SBATCH --output=job_%x-%j.out
#SBATCH --error=job_%x-%j.err
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpupod
#SBATCH --mem=512GB

python3 selective_mmlu.py