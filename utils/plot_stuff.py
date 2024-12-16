import matplotlib.pyplot as plt
import numpy as np

def plot_analysis_results(results, output_file='analysis_plots.pdf'):
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Extract data for the first example
    data = results[0]
    steps = list(range(len(data['analysis']['step_by_step']['regular']['entropy'])))
    deviation_step = 5  # Based on your code
    
    # Plot 1: Entropy Over Time
    ax1.set_title('Entropy Over Time')
    ax1.plot(steps, data['analysis']['step_by_step']['regular']['entropy'], label='Regular', color='blue')
    for path in ['strategic_5th', 'strategic_8th', 'strategic_9th', 'strategic_10th']:
        ax1.plot(steps, data['analysis']['step_by_step'][path]['entropy'], label=path, alpha=0.5)
    ax1.axvline(x=deviation_step, color='g', linestyle='--', label='Deviation Point')
    ax1.set_xlabel('Generation Step')
    ax1.set_ylabel('Entropy')
    ax1.legend()
    
    # Plot 2: Model Confidence Over Time
    ax2.set_title('Model Confidence Over Time')
    ax2.plot(steps, data['analysis']['step_by_step']['regular']['confidence'], label='Regular', color='blue')
    for path in ['strategic_5th', 'strategic_8th', 'strategic_9th', 'strategic_10th']:
        ax2.plot(steps, data['analysis']['step_by_step'][path]['confidence'], label=path, alpha=0.5)
    ax2.axvline(x=deviation_step, color='g', linestyle='--', label='Deviation Point')
    ax2.set_xlabel('Generation Step')
    ax2.set_ylabel('Confidence (Max Probability)')
    ax2.legend()
    
    # Plot 3: Distribution Spread Over Time
    ax3.set_title('Distribution Spread Over Time')
    ax3.plot(steps, data['analysis']['step_by_step']['regular']['distribution_spread'], label='Regular', color='blue')
    for path in ['strategic_5th', 'strategic_8th', 'strategic_9th', 'strategic_10th']:
        ax3.plot(steps, data['analysis']['step_by_step'][path]['distribution_spread'], label=path, alpha=0.5)
    ax3.axvline(x=deviation_step, color='g', linestyle='--', label='Deviation Point')
    ax3.set_xlabel('Generation Step')
    ax3.set_ylabel('Spread (Top-K Range)')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Load data and create plots
import json

with open('gsm8k_analysis_final.json', 'r') as f:
    results = json.load(f)

plot_analysis_results(results, 'analysis_plots.pdf')