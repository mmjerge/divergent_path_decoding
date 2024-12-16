import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist
import gc
import traceback

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def compare_sequences(input_text, model, tokenizer, rank, deviation_step=5, max_steps=512, top_k=10):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(f'cuda:{rank}')
    
    sequences = {
        'regular': input_ids.clone(),
        'strategic_5th': input_ids.clone(),
        'strategic_8th': input_ids.clone(),
        'strategic_9th': input_ids.clone(),
        'strategic_10th': input_ids.clone()
    }
    
    has_deviated = {path: False for path in sequences.keys() if path != 'regular'}
    
    token_changes = {path: None for path in sequences.keys() if path != 'regular'}
    
    for step in range(max_steps):
        for path_name, sequence in sequences.items():
            outputs = model(input_ids=sequence)
            probs = torch.nn.functional.softmax(outputs.logits[:, -1, :], dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            if path_name == 'regular':
                chosen_index = top_indices[:, 0].to(f'cuda:{rank}')
            else:
                if step == deviation_step - 1 and not has_deviated[path_name]:
                    token_index = {
                        'strategic_5th': 4,
                        'strategic_8th': 7,
                        'strategic_9th': 8,
                        'strategic_10th': 9
                    }[path_name]
                    chosen_index = top_indices[:, token_index].to(f'cuda:{rank}')
                    has_deviated[path_name] = True
                    
                    regular_token_text = tokenizer.decode(top_indices[:, 0][0])
                    strategic_token_text = tokenizer.decode(chosen_index[0])
                    token_changes[path_name] = {
                        'step': step + 1,
                        'regular_token': regular_token_text,
                        'strategic_token': strategic_token_text,
                        'position_in_text': len(tokenizer.decode(sequence[0])),
                        'token_rank': token_index + 1
                    }
                else:
                    chosen_index = top_indices[:, 0].to(f'cuda:{rank}')
            
            sequences[path_name] = torch.cat([sequence, chosen_index.unsqueeze(-1)], dim=-1)
    
    decoded_sequences = {}
    token_counts = {}
    for path_name, sequence in sequences.items():
        decoded_sequences[path_name] = tokenizer.decode(sequence[0], skip_special_tokens=True)
        token_counts[path_name] = len(sequence[0])
    
    return decoded_sequences, token_changes, token_counts

def format_prompt(question, choices):
    prompt = (
        f"Question: {question}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        f"Please provide a concise explanation in 2-3 sentences, followed by your final answer (A, B, C, or D). Complete your response within approximately 200 words."
    )
    return prompt

def process_subset(rank, world_size, dataset, model, tokenizer, start_idx, end_idx, indices=None):
    outputs = []
    
    pbar_questions = tqdm(indices if indices is not None else range(start_idx, end_idx), 
                         desc=f"GPU {rank} Processing Questions",
                         position=rank,
                         leave=True)
    
    for sample_index in pbar_questions:
        sample = dataset['test'][sample_index]
        
        pbar_questions.set_description(f"GPU {rank} - Processing {sample['subject']} (Question {sample_index})")
        
        question = sample['question']
        choices = [sample['choices'][i] for i in range(4)]
        input_text = format_prompt(question, choices)
        
        decoded_sequences, token_changes, token_counts = compare_sequences(
            input_text, 
            model, 
            tokenizer,
            rank,
            deviation_step=5,
            max_steps=256,
            top_k=10
        )
        
        output_data = {
            'subject': sample['subject'],
            'question': question,
            'choices': choices,
            'correct_answer': sample['answer'],
            'prompt_used': input_text,
            'sequences': decoded_sequences,
            'token_changes': token_changes,
            'token_counts': token_counts,
            'question_index': sample_index
        }
        outputs.append(output_data)
        
        if len(outputs) % 10 == 0:
            with open(f'mmlu_outputs_gpu{rank}_intermediate.json', 'w', encoding='utf-8') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    pbar_questions.close()
    return outputs

def run_parallel(rank, world_size, num_samples=None):
    try:
        setup(rank, world_size)
        
        dataset = load_dataset("cais/mmlu", "all")
        
        if num_samples is not None:
            all_indices = list(range(num_samples))
        else:
            all_indices = list(range(len(dataset['test'])))
        
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map={'': f'cuda:{rank}'}
        )
        
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        
        samples_per_gpu = len(all_indices) // world_size
        start_idx = rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if rank != world_size - 1 else len(all_indices)
        
        gpu_indices = all_indices[start_idx:end_idx]
        
        print(f"GPU {rank} processing questions {gpu_indices[0]} to {gpu_indices[-1]}")
        outputs = process_subset(rank, world_size, dataset, model, tokenizer, 
                               gpu_indices[0], gpu_indices[-1], indices=gpu_indices)
        
        with open(f'mmlu_outputs_gpu{rank}_final.json', 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False, separators=(',', ': '))
        
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e
    
    finally:
        cleanup()
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()

def main():
    world_size = torch.cuda.device_count()
    
    try:
        mp.spawn(
            run_parallel,
            args=(world_size, None),
            nprocs=world_size,
            join=True
        )
        
        all_outputs = []
        for rank in range(world_size):
            try:
                with open(f'mmlu_outputs_gpu{rank}_final.json', 'r', encoding='utf-8') as f:
                    outputs = json.load(f)
                    all_outputs.extend(outputs)
            except Exception as e:
                print(f"Error loading results from GPU {rank}: {str(e)}")
        
        with open('mmlu_outputs_combined.json', 'w', encoding='utf-8') as f:
            json.dump(all_outputs, f, indent=2, ensure_ascii=False, separators=(',', ': '))
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    main()