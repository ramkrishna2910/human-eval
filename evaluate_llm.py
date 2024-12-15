#!/usr/bin/env python3
from typing import List, Dict
import json
import os
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def setup_model():
    """
    Initialize the Phi-3-mini model and tokenizer.
    Returns the model and tokenizer objects.
    """
    print("Loading Phi-3-mini model and tokenizer...")
    model_name = "microsoft/Phi-3.5-mini-instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with lower precision for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto",  # Automatically choose best device (CPU/GPU)
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_completion(prompt: str, model, tokenizer) -> str:
    """
    Generate code completion using Phi-3-mini model.
    """
    try:
        # Prepare the prompt
        messages = [
            "You are a helpful coding assistant. Complete the following Python function.",
            "Only provide the function body, not the function signature.",
            f"Here's the function to complete:\n{prompt}"
        ]
        full_prompt = "\n".join(messages)
        
        # Tokenize input
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = inputs.to(model.device)
        
        # Generate completion
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=500,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the completion
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up the completion
        # If the completion includes the function signature, remove it
        if completion.startswith("def"):
            import re
            match = re.search(r'\n\s+', completion)
            if match:
                completion = completion[match.start():]
        
        # Ensure proper indentation (4 spaces)
        completion_lines = completion.split('\n')
        cleaned_lines = []
        for line in completion_lines:
            # Remove any existing indentation
            stripped = line.lstrip()
            if stripped:  # If line is not empty
                # Add 4 spaces indentation
                cleaned_lines.append("    " + stripped)
        
        completion = '\n'.join(cleaned_lines)
        
        return completion
    
    except Exception as e:
        print(f"Error generating completion: {e}")
        return "    return 1"  # Fallback completion

def main():
    # Setup model and tokenizer
    # model, tokenizer = setup_model()
    
    # # Read all problems from the HumanEval dataset
    # problems = read_problems()
    
    # # Generate samples - one completion per problem
    # samples = []
    # for task_id, problem in problems.items():
    #     print(f"\nGenerating completion for {task_id}")
        
    #     # Get the prompt from the problem
    #     prompt = problem["prompt"]
        
    #     # Generate code completion using Phi-3-mini
    #     completion = generate_completion(prompt, model, tokenizer)
    #     print(f"Generated completion:\n{completion}")
        
    #     # Add to samples
    #     sample = {
    #         "task_id": task_id,
    #         "completion": completion
    #     }
    #     samples.append(sample)
    
    # # Write samples to a jsonl file
    output_file = "samples.jsonl"
    # write_jsonl(output_file, samples)
    # print(f"\nSaved {len(samples)} samples to {output_file}")
    
    # Evaluate the samples
    print("\nEvaluating samples...")
    results = evaluate_functional_correctness(
        sample_file=output_file,
        k=[1],  # Evaluate pass@k for k=1
        n_workers=1,  # Number of parallel workers
        timeout=3.0  # Timeout for each evaluation in seconds
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 