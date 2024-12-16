import torch
import numpy as np
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.tree import Tree
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datasets import load_dataset
import argparse
import json
import gc
import os
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run analysis on examples from MMLU or GSM8K dataset')
    parser.add_argument('--dataset', type=str, choices=['mmlu', 'gsm8k'], required=True,
                      help='Which dataset to use (mmlu or gsm8k)')
    parser.add_argument('--num_samples', type=int, default=1,
                      help='Number of samples to process (default: 1)')
    parser.add_argument('--save_prefix', type=str, default='analysis',
                      help='Prefix for saved files')
    return parser.parse_args()

class SequenceAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def analyze_distributions(self, input_text, deviation_step=5, max_steps=256):
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        sequences = {
            'regular': input_ids.clone(),
            'strategic_5th': input_ids.clone(),
            'strategic_8th': input_ids.clone(),
            'strategic_9th': input_ids.clone(),
            'strategic_10th': input_ids.clone()
        }
        
        analysis_results = {path: {'entropy': [], 'confidence': [], 'distribution_spread': []} 
                          for path in sequences.keys()}
        analysis_results['divergence_metrics'] = []
        
        for step in range(max_steps):
            torch.cuda.empty_cache()  
            
            distributions = {}
            for path_name, sequence in sequences.items():
                dist = self._get_distribution(sequence)
                distributions[path_name] = dist
                
                analysis_results[path_name]['entropy'].append(self._calculate_entropy(dist))
                analysis_results[path_name]['confidence'].append(torch.max(dist).item())
                analysis_results[path_name]['distribution_spread'].append(self._calculate_spread(dist))
            
            if step >= deviation_step:
                for path_name in ['strategic_5th', 'strategic_8th', 'strategic_9th', 'strategic_10th']:
                    divergence = self._calculate_kl_divergence(distributions['regular'], distributions[path_name])
                    analysis_results['divergence_metrics'].append({
                        'step': step,
                        'path': path_name,
                        'kl_divergence': divergence,
                        'probability_ratio': distributions[path_name].max().item() / distributions['regular'].max().item()
                    })
            
            for path_name, sequence in sequences.items():
                if path_name == 'regular':
                    next_token = torch.argmax(distributions[path_name])
                else:
                    if step == deviation_step-1:
                        token_index = {
                            'strategic_5th': 4,
                            'strategic_8th': 7,
                            'strategic_9th': 8,
                            'strategic_10th': 9
                        }[path_name]
                        next_token = torch.topk(distributions[path_name], 10)[1][token_index]
                    else:
                        next_token = torch.argmax(distributions[path_name])
                
                sequences[path_name] = torch.cat([sequence, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
        
        return self._summarize_analysis(analysis_results)
    
    def _get_distribution(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            return torch.nn.functional.softmax(outputs.logits[:, -1, :], dim=-1)[0]
    
    def _calculate_entropy(self, dist):
        probs = dist.cpu().detach().numpy()
        return entropy(probs[probs > 0])
    
    def _calculate_spread(self, dist):
        top_k_probs = torch.topk(dist, 10)[0]
        return (top_k_probs[0] - top_k_probs[-1]).item()
    
    def _calculate_kl_divergence(self, p, q):
        p = p.cpu().detach().numpy()
        q = q.cpu().detach().numpy()
        return entropy(p, q)
    
    def _summarize_analysis(self, results):
        summaries = {}
        for path in ['strategic_5th', 'strategic_8th', 'strategic_9th', 'strategic_10th']:
            pre_deviation_entropy = np.mean(results[path]['entropy'][:5])
            post_deviation_entropy = np.mean(results[path]['entropy'][5:])
            entropy_change = (post_deviation_entropy - pre_deviation_entropy) / pre_deviation_entropy
            
            path_divergences = [d['kl_divergence'] for d in results['divergence_metrics'] 
                              if d['path'] == path]
            avg_divergence = np.mean(path_divergences)
            
            summaries[path] = {
                'entropy_change_percentage': entropy_change * 100,
                'average_distribution_divergence': avg_divergence,
                'confidence_impact': np.mean([d['probability_ratio'] for d in results['divergence_metrics']
                                           if d['path'] == path])
            }
        
        return {
            'metrics': summaries,
            'step_by_step': results,
            'interpretation': self._generate_interpretation(summaries)
        }
    
    def _generate_interpretation(self, summaries):
        interpretations = []
        for path, metrics in summaries.items():
            path_interpretations = []
            entropy_change = metrics['entropy_change_percentage']
            avg_divergence = metrics['average_distribution_divergence']
            
            if entropy_change > 20:
                path_interpretations.append(f"{path}: Led to significantly more uncertain predictions")
            elif entropy_change < -20:
                path_interpretations.append(f"{path}: Led to more confident predictions")
                
            if avg_divergence > 1.0:
                path_interpretations.append(f"{path}: Distributions diverged substantially from regular path")
            elif avg_divergence < 0.5:
                path_interpretations.append(f"{path}: Remained relatively similar despite deviation")
                
            interpretations.extend(path_interpretations)
            
        return interpretations

def format_prompt(question, dataset_type='gsm8k'):
    if dataset_type == 'mmlu':
        choices = question['choices']
        prompt = (
            f"Question: {question['question']}\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}\n\n"
            f"Please provide a concise explanation in 2-3 sentences, followed by your final answer (A, B, C, or D)."
        )
    else:  
        prompt = (
            f"Question: {question['question']}\n\n"
            f"Please solve this step by step and provide your final numerical answer."
        )
    return prompt

def analyze_example(model, tokenizer, question, dataset_type):
    try:
        input_text = format_prompt(question, dataset_type)
        
        device = next(model.parameters()).device
        
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        sequences = {
            'regular': input_ids.clone(),
            'strategic_5th': input_ids.clone(),
            'strategic_8th': input_ids.clone(),
            'strategic_9th': input_ids.clone(),
            'strategic_10th': input_ids.clone()
        }
        
        max_steps = 256
        deviation_step = 5
        top_k = 10
        
        for step in range(max_steps):
            torch.cuda.empty_cache()
            
            for path_name, sequence in sequences.items():
                with torch.no_grad():
                    outputs = model(input_ids=sequence)
                    probabilities = torch.nn.functional.softmax(outputs.logits[:, -1, :], dim=-1)
                    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)
                    
                    if path_name == 'regular':
                        next_token = top_indices[:, 0]
                    else:
                        if step == deviation_step-1:
                            token_index = {
                                'strategic_5th': 4,
                                'strategic_8th': 7,
                                'strategic_9th': 8,
                                'strategic_10th': 9
                            }[path_name]
                            next_token = top_indices[:, token_index]
                        else:
                            next_token = top_indices[:, 0]
                    
                    next_token = next_token.unsqueeze(0)
                    sequences[path_name] = torch.cat([sequences[path_name], next_token], dim=-1)
        
        decoded_sequences = {
            path: tokenizer.decode(sequence[0], skip_special_tokens=True)
            for path, sequence in sequences.items()
        }
        
        analyzer = SequenceAnalyzer(model, tokenizer)
        analysis_results = analyzer.analyze_distributions(input_text, deviation_step, max_steps)
        
        return {
            'question': question,
            'prompt': input_text,
            'sequences': decoded_sequences,
            'analysis': analysis_results
        }
        
    except Exception as e:
        print(f"Error in analyze_example: {str(e)}")
        return None

def main():
    args = parse_arguments()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "40GiB"},  
        low_cpu_mem_usage=True,
        offload_state_dict=True,  
    )
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        use_fast=False
    )
    
    model.gradient_checkpointing_enable()
    
    if args.dataset == 'mmlu':
        dataset = load_dataset("cais/mmlu", "all")
        split = 'test'
    else:  
        dataset = load_dataset("openai/gsm8k", "main")
        split = 'train'
    
    results = []
    for i in range(min(args.num_samples, len(dataset[split]))):
        try:
            print(f"\nProcessing example {i+1}/{args.num_samples}")
            example = dataset[split][i]
            result = analyze_example(model, tokenizer, example, args.dataset)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                with open(f'{args.save_prefix}_intermediate_{i+1}.json', 'w') as f:
                    json.dump(results, f, indent=2)
                    
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing example {i+1}: {str(e)}")
            continue
    
    with open(f'{args.save_prefix}_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.save_prefix}_final.json")

if __name__ == "__main__":
    main()