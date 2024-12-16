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

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def compare_sequences(input_text, model, tokenizer, rank, deviation_step=5, max_steps=256, top_k=5):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(f'cuda:{rank}')
    regular_sequence = input_ids.clone()
    strategic_sequence = input_ids.clone()
    
    has_deviated = False
    token_change_info = None
    
    for step in range(max_steps):
        # Regular sequence processing
        outputs_regular = model(input_ids=regular_sequence)
        probs_regular = torch.nn.functional.softmax(outputs_regular.logits[:, -1, :], dim=-1)
        top_probs_regular, top_indices_regular = torch.topk(probs_regular, k=top_k, dim=-1)
        
        # Strategic sequence processing
        outputs_strategic = model(input_ids=strategic_sequence)
        probs_strategic = torch.nn.functional.softmax(outputs_strategic.logits[:, -1, :], dim=-1)
        top_probs_strategic, top_indices_strategic = torch.topk(probs_strategic, k=top_k, dim=-1)
        
        # Regular sequence always takes top token
        regular_token = top_indices_regular[:, 0].to(f'cuda:{rank}')
        regular_sequence = torch.cat([regular_sequence, regular_token.unsqueeze(-1)], dim=-1)
        
        # Strategic sequence takes 5th highest token only at deviation_step if not deviated yet
        if step == deviation_step - 1 and not has_deviated:
            chosen_index = top_indices_strategic[:, 4].to(f'cuda:{rank}')
            has_deviated = True
            
            regular_token_text = tokenizer.decode(regular_token[0])
            strategic_token_text = tokenizer.decode(chosen_index[0])
            token_change_info = {
                'step': step + 1,
                'regular_token': regular_token_text,
                'strategic_token': strategic_token_text,
                'position_in_text': len(tokenizer.decode(regular_sequence[0]))
            }
        else:
            chosen_index = top_indices_strategic[:, 0].to(f'cuda:{rank}')
            
        strategic_sequence = torch.cat([strategic_sequence, chosen_index.unsqueeze(-1)], dim=-1)
    
    regular_text = tokenizer.decode(regular_sequence[0], skip_special_tokens=True)
    strategic_text = tokenizer.decode(strategic_sequence[0], skip_special_tokens=True)
    
    token_info = {
        'regular_token_count': len(regular_sequence[0]),
        'strategic_token_count': len(strategic_sequence[0])
    }
    
    return regular_text, strategic_text, token_change_info, token_info

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
    
    # Progress bar for questions
    pbar_questions = tqdm(indices if indices is not None else range(start_idx, end_idx), 
                         desc=f"GPU {rank} Processing Questions",
                         position=rank,
                         leave=True)
    
    for sample_index in pbar_questions:
        sample = dataset['test'][sample_index]
        
        # Update progress bar with current subject
        pbar_questions.set_description(f"GPU {rank} - Processing {sample['subject']} (Question {sample_index})")
        
        question = sample['question']
        choices = [sample['choices'][i] for i in range(4)]
        input_text = format_prompt(question, choices)
        
        regular_text, strategic_text, token_change, token_info = compare_sequences(
            input_text, 
            model, 
            tokenizer,
            rank,
            deviation_step=5,
            max_steps=256
        )
        
        output_data = {
            'subject': sample['subject'],
            'question': question,
            'choices': choices,
            'correct_answer': sample['answer'],
            'prompt_used': input_text,
            'regular_text': regular_text,
            'strategic_text': strategic_text,
            'token_change': token_change,
            'token_counts': token_info,
            'question_index': sample_index
        }
        outputs.append(output_data)
        
        # Save intermediate results every 10 questions
        if len(outputs) % 10 == 0:
            with open(f'mmlu_outputs_gpu{rank}_intermediate.json', 'w', encoding='utf-8') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    pbar_questions.close()
    return outputs

def run_parallel(rank, world_size, num_samples=None):
    try:
        setup(rank, world_size)
        
        # Load dataset
        dataset = load_dataset("cais/mmlu", "all")
        
        # If num_samples is specified, create random indices
        if num_samples is not None:
            # Set random seed for reproducibility
            torch.manual_seed(42)
            # Generate random indices
            all_indices = torch.randperm(len(dataset['test']))[:num_samples].tolist()
            # Sort indices to maintain order
            all_indices.sort()
        else:
            all_indices = list(range(len(dataset['test'])))
        
        # Load model and ensure it's on the correct device
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map={'': f'cuda:{rank}'}
        )
        
        # DDP wrapper
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        
        # Calculate samples per GPU based on random indices
        samples_per_gpu = len(all_indices) // world_size
        start_idx = rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if rank != world_size - 1 else len(all_indices)
        
        # Get indices for this GPU
        gpu_indices = all_indices[start_idx:end_idx]
        
        print(f"GPU {rank} processing questions {gpu_indices[0]} to {gpu_indices[-1]}")
        outputs = process_subset(rank, world_size, dataset, model, tokenizer, 
                               gpu_indices[0], gpu_indices[-1], indices=gpu_indices)
        
        # Save final results for this GPU
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
            args=(world_size, None),  # Process 150 random questions
            nprocs=world_size,
            join=True
        )
        
        # Combine results from all GPUs
        all_outputs = []
        for rank in range(world_size):
            try:
                with open(f'mmlu_outputs_gpu{rank}_final.json', 'r', encoding='utf-8') as f:
                    outputs = json.load(f)
                    all_outputs.extend(outputs)
            except Exception as e:
                print(f"Error loading results from GPU {rank}: {str(e)}")
        
        # Save combined results
        with open('mmlu_outputs_combined.json', 'w', encoding='utf-8') as f:
            json.dump(all_outputs, f, indent=2, ensure_ascii=False, separators=(',', ': '))
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    main()