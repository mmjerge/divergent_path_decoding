#!/bin/bash

#SBATCH --job-name="counterfactual_analysis"
#SBATCH --output=job_%x-%j.out
#SBATCH --error=job_%x-%j.err
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpupod
#SBATCH --mem=256GB
#SBATCH --time=12:00:00 

# Run the script
python3 selective_greedy_counterfactual_analysis.py --dataset gsm8k --num_samples 10 --save_prefix gsm8k_analysis