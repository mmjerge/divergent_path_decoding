import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm 

def compare_sequences(input_text, model, tokenizer, deviation_step=5, max_steps=10, top_k=5):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    regular_sequence = input_ids.clone()
    strategic_sequence = input_ids.clone()
    
    for step in range(max_steps):
        outputs_regular = model(input_ids=regular_sequence)
        probs_regular = torch.nn.functional.softmax(outputs_regular.logits[:, -1, :], dim=-1)
        top_probs_regular, top_indices_regular = torch.topk(probs_regular, k=top_k, dim=-1)
        
        outputs_strategic = model(input_ids=strategic_sequence)
        probs_strategic = torch.nn.functional.softmax(outputs_strategic.logits[:, -1, :], dim=-1)
        top_probs_strategic, top_indices_strategic = torch.topk(probs_strategic, k=top_k, dim=-1)
        
        regular_sequence = torch.cat([regular_sequence, top_indices_regular[:, 0].unsqueeze(-1)], dim=-1)
        
        if step == deviation_step - 1:
            chosen_index = top_indices_strategic[:, 4]
        else:
            chosen_index = top_indices_strategic[:, 0]
        strategic_sequence = torch.cat([strategic_sequence, chosen_index.unsqueeze(-1)], dim=-1)
        
    regular_text = tokenizer.decode(regular_sequence[0], skip_special_tokens=True)
    strategic_text = tokenizer.decode(strategic_sequence[0], skip_special_tokens=True)
    
    return regular_text, strategic_text

dataset = load_dataset("gsm8k", "main")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

outputs = []

for sample_index in tqdm(range(10)):
    sample = dataset['train'][sample_index]
    input_text = sample['question']
    answer = sample['answer']
    
    regular_text, strategic_text = compare_sequences(input_text, model, tokenizer, deviation_step=5)
    
    output_data = {
        'question': input_text,
        'regular_text': regular_text,
        'strategic_text': strategic_text,
        'answer': answer
    }
    outputs.append(output_data)
    
with open('outputs.json', 'w') as f:
    json.dump(outputs, f, indent=4)
    
print("\nResults saved to outputs.json")
