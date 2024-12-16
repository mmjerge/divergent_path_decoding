import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.tree import Tree

def visualize_token_distributions(input_text, max_steps=10, top_k=50):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    console = Console()
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    regular_sequence = input_ids.clone()
    custom_sequence = input_ids.clone()
    
    regular_tree = Tree(f"[bold blue]Regular Greedy Search - Starting text: '{input_text}'")
    custom_tree = Tree(f"[bold green]Custom Greedy Search (5th token) - Starting text: '{input_text}'")
    
    for step in range(max_steps):
        outputs_regular = model(input_ids=regular_sequence)
        probs_regular = torch.nn.functional.softmax(outputs_regular.logits[:, -1, :], dim=-1)
        top_probs_regular, top_indices_regular = torch.topk(probs_regular, k=top_k, dim=-1)
        
        outputs_custom = model(input_ids=custom_sequence)
        probs_custom = torch.nn.functional.softmax(outputs_custom.logits[:, -1, :], dim=-1)
        top_probs_custom, top_indices_custom = torch.topk(probs_custom, k=top_k, dim=-1)
        
        regular_step = regular_tree.add(f"[yellow]Step {step + 1}")
        custom_step = custom_tree.add(f"[yellow]Step {step + 1}")
        
        for i, (prob, idx) in enumerate(zip(top_probs_regular[0], top_indices_regular[0])):
            token = tokenizer.decode(idx)
            style = "[bold blue]" if i == 0 else ""
            regular_step.add(f"{style}Token: '{token}' - Prob: {prob:.3f}")
            
        for i, (prob, idx) in enumerate(zip(top_probs_custom[0], top_indices_custom[0])):
            token = tokenizer.decode(idx)
            style = "[bold green]" if i == 4 else ""
            custom_step.add(f"{style}Token: '{token}' - Prob: {prob:.3f}")
        
        regular_sequence = torch.cat([regular_sequence, top_indices_regular[:, 0].unsqueeze(-1)], dim=-1)
        custom_sequence = torch.cat([custom_sequence, top_indices_custom[:, 4].unsqueeze(-1)], dim=-1)
    
    regular_text = tokenizer.decode(regular_sequence[0], skip_special_tokens=True)
    custom_text = tokenizer.decode(custom_sequence[0], skip_special_tokens=True)
    
    print("\n==== Token Distribution Trees ====")
    console.print(regular_tree)
    print("\n")
    console.print(custom_tree)
    print("\n==== Final Generated Sequences ====")
    print(f"Regular: {regular_text}")
    print(f"Custom:  {custom_text}")

input_text = "The capital of France"
visualize_token_distributions(input_text)