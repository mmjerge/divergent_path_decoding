#!/bin/bash

#SBATCH --job-name="counterfactual_analysis"
#SBATCH --output=job_%x-%j.out
#SBATCH --error=job_%x-%j.err
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpupod
#SBATCH --mem=512GB
#SBATCH --time=3:00:00

# python3 main.py --dataset mmlu --num_samples 150
python3 main.py --dataset gsm8k --num_samples 150