# Divergent Path Decoding

This repository contains the implementation of Divergent Path Decoding (DPD), an inference-time approach that modifies the autoregressive token generation process to improve output reliability in large language models.

## Project Structure

```
├── charts/               # Visualization tools for analysis plots
├── gsm8k/               # Output directory for GSM8K evaluation results
├── mmlu/                # Output directory for MMLU evaluation results
├── scripts/             # Utility scripts for experiments
├── utils/               # Common utility functions
├── custom_greedy_counterfactual.py
├── environment.yaml     # Environment configuration
├── main.py             # Main execution script
├── multi_causal_counterfactual_analysis.py
├── selective_distributed_multi_mmlu.py
├── selective_distributed_random_mmlu.py
├── selective_greedy_counterfactual.py
└── selective_greedy_counterfactual_analysis.py
```

## Overview

This project implements a novel decoding method that:
- Modifies the autoregressive token generation process at inference time
- Explores alternative generation paths through controlled token interventions
- Requires no model fine-tuning or architectural changes
- Improves output reliability while maintaining model capabilities

## Requirements

The system was tested on a high-performance computing environment with:
- NVIDIA A100 GPUs (8 per node, 40GB VRAM each)
- Dual 64-core AMD EPYC processors
- 512GB system RAM
- SLURM workload manager
- PyTorch with DistributedDataParallel (DDP)
- Hugging Face Transformers library

## Installation

```bash
# Clone the repository
git clone https://github.com/mmjerge/counterfactual_ensemble_decoding.git

# Create and configure environment from yaml
conda env create -f environment.yaml
conda activate divergent_decoding
```

## Key Components

- `main.py`: Primary script for running experiments and generating results
- `custom_greedy_counterfactual.py`: Implementation of basic greedy counterfactual generation
- `multi_causal_counterfactual_analysis.py`: Multi-causal analysis framework
- `selective_distributed_multi_mmlu.py`: Distributed MMLU evaluation
- `selective_distributed_random_mmlu.py`: Random sampling for distributed evaluation
- `selective_greedy_counterfactual.py`: Selective greedy counterfactual generation
- `selective_greedy_counterfactual_analysis.py`: Analysis tools for selective approaches

## Usage

Run experiments using the main.py script:

```bash
# Run evaluation on GSM8K dataset with 150 samples
python3 main.py --dataset gsm8k --num_samples 150

# Run evaluation on MMLU dataset with 150 samples
python3 main.py --dataset mmlu --num_samples 150
```

Results will be generated in the respective output directories:
- `mmlu/`: Contains MMLU evaluation results
- `gsm8k/`: Contains GSM8K evaluation results

## Method

The implementation follows these key steps:
1. Generates multiple decoding paths for each input prompt
2. Introduces controlled deviations at specific token positions
3. Explores alternative reasoning trajectories
4. Aggregates results using majority voting

## Results

Performance on benchmark datasets:
- MMLU: 72.4% (0-shot)
- GSM8K: 78.6% (0-shot)

## Citation

```bibtex
@article{jerge2024divergent,
  title={Divergent Path Decoding},
  author={Jerge, Michael},
  institution={University of Virginia},
  year={2024}
}
```

## License

See LICENSE file for details.

## Contact

- Author: Michael Jerge
- Institution: University of Virginia
- Computing ID: mj6ux
- Advisor: David Evans
- Repository: https://github.com/mmjerge/counterfactual_ensemble_decoding
