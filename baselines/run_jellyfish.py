"""
Jellyfish-7B LLM Baseline for Data Repair
Uses Jellyfish-7B model for error detection and correction
"""
import pandas as pd
import os
import time
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# Global model and tokenizer
model = None
tokenizer = None
device = None
example_records = None  # Store clean examples

def prepare_examples(clean_df, num_examples=20):
    """
    Prepare example records from clean data to guide the model
    
    Args:
        clean_df: Clean dataframe
        num_examples: Number of example rows to use
    
    Returns:
        String containing formatted examples
    """
    global example_records
    
    examples = clean_df.head(num_examples)
    example_records = examples
    
    # Format examples as readable text
    example_text = "Here are some examples of correct records from this dataset:\n\n"
    
    for idx in range(min(num_examples, len(examples))):
        record = examples.iloc[idx].to_dict()
        record_str = ", ".join([f"{k}: {v}" for k, v in record.items()])
        example_text += f"Example {idx+1}: [{record_str}]\n"
    
    example_text += "\nBased on these examples, you can understand the correct format and patterns of the data.\n"
    
    return example_text

def initialize_model(use_8bit=False):
    """Initialize Jellyfish-7B model with memory optimization"""
    global model, tokenizer, device
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Clear cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print("Loading Jellyfish-7B model...")
    try:
        # Load with memory optimizations
        load_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,  # Reduce CPU memory usage
        }
        
        # Optional 8-bit quantization for lower memory
        if use_8bit and torch.cuda.is_available():
            load_kwargs["load_in_8bit"] = True
            print("Using 8-bit quantization to save memory")
        
        model = AutoModelForCausalLM.from_pretrained(
            "NECOUDBFM/Jellyfish-7B",
            **load_kwargs
        )
    except Exception as e:
        print(f"Failed to load with safetensors: {e}")
        print("Trying with trust_remote_code...")
        model = AutoModelForCausalLM.from_pretrained(
            "NECOUDBFM/Jellyfish-7B",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained("NECOUDBFM/Jellyfish-7B")
    
    if torch.cuda.is_available():
        print(f"Model loaded! GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print("Model loaded successfully!")

def correct_value(record, attribute_to_correct, examples_text):
    """Correct an erroneous value for a specific attribute"""
    system_message = "You are an AI assistant that follows instruction extremely well. Help as much as you can."
    
    # Format record
    record_str = ", ".join([f"{k}: {v}" for k, v in record.items()])
    
    user_message = f"""{examples_text}

Now, you are presented with a record that has an error in a specific attribute: {attribute_to_correct}.
Your task is to correct the value of {attribute_to_correct} using the available information in the record and the patterns you learned from the examples above.

Record with Error: [{record_str}]
Attribute with Error: [{attribute_to_correct}: {record[attribute_to_correct]}]

Based on the examples and the provided record, what should be the corrected value for {attribute_to_correct}?
Answer only the corrected value, nothing else."""
    
    prompt = f"{system_message}\n\n[INST]:\n\n{user_message}\n\n[\\INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=0.35,
                top_p=0.9,
                max_new_tokens=32,  # Reduced from 64 to save memory
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15,
            )
        
        response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        # Clean up to free memory
        del inputs, input_ids, output
        if device == "cuda:0" or device == "cuda":
            torch.cuda.empty_cache()
        
        return response
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM error: {e}")
            # Emergency cleanup
            if device == "cuda:0" or device == "cuda":
                torch.cuda.empty_cache()
            return None
        raise e

def repair_with_jellyfish(dirty_path, clean_path, output_path, max_rows=None, num_examples=20, use_8bit=False):
    """
    Repair dirty data using Jellyfish-7B LLM
    
    Args:
        dirty_path: Path to dirty CSV
        clean_path: Path to clean CSV (for error detection)
        output_path: Path to save repaired CSV
        max_rows: Maximum number of rows to process (for testing)
        num_examples: Number of clean examples for few-shot learning
        use_8bit: Use 8-bit quantization to save memory
    """
    start_time = time.time()
    
    # Initialize model
    initialize_model(use_8bit=use_8bit)
    
    # Load datasets - convert all to string for comparison
    dirty_df = pd.read_csv(dirty_path, dtype=str)
    clean_df = pd.read_csv(clean_path, dtype=str)
    
    # Create a copy for repairs
    repaired_df = dirty_df.copy()
    
    # Limit rows for testing
    if max_rows:
        dirty_df = dirty_df.head(max_rows)
        clean_df = clean_df.head(max_rows)
        repaired_df = repaired_df.head(max_rows)
        print(f"Processing first {max_rows} rows only")
    
    # Prepare examples from clean data
    print("Preparing examples from clean data...")
    examples_text = prepare_examples(clean_df, num_examples=num_examples)
    print(f"Using {min(num_examples, len(clean_df))} example records to guide the model")
    
    # Show first example for verification
    first_example = clean_df.iloc[0].to_dict()
    print(f"First example: {first_example}")
    print()
    
    # Detect errors by comparing dirty and clean (as strings)
    dirty_str = dirty_df.astype(str)
    clean_str = clean_df.astype(str)
    error_mask = (dirty_str != clean_str)
    
    # Count total errors
    total_errors = error_mask.sum().sum()
    print(f"Detected {total_errors} errors in {len(dirty_df)} rows x {len(dirty_df.columns)} columns")
    
    # Process only cells with errors
    error_count = 0
    for idx in tqdm(range(len(dirty_df)), desc="Processing rows"):
        # Convert row to dictionary
        record = dirty_df.iloc[idx].to_dict()
        
        # Check each attribute
        for col in dirty_df.columns:
            # Skip if no error in this cell
            if not error_mask.at[idx, col]:
                continue
            
            error_count += 1
            value = str(record[col])
            
            # Skip nan values
            if value == 'nan' or len(value) == 0:
                continue
            
            # Use LLM to correct the value (with examples)
            try:
                corrected = correct_value(record, col, examples_text)
                
                if corrected is None:
                    print(f"[{error_count}/{total_errors}] Row {idx}, Col '{col}': OOM, skipping")
                    continue
                
                repaired_df.at[idx, col] = corrected
                print(f"[{error_count}/{total_errors}] Row {idx}, Col '{col}': '{value}' -> '{corrected}'")
                
                # Periodic memory cleanup every 10 corrections
                if error_count % 10 == 0 and (device == "cuda:0" or device == "cuda"):
                    torch.cuda.empty_cache()
                    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"  [Memory check] GPU: {mem_allocated:.2f} GB")
                
            except Exception as e:
                print(f"Error processing row {idx}, col {col}: {e}")
                # Emergency cleanup on error
                if device == "cuda:0" or device == "cuda":
                    torch.cuda.empty_cache()
                continue
    
    # Save repaired data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    repaired_df.to_csv(output_path, index=False)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate metrics
    dirty_str = dirty_df.astype(str)
    clean_str = clean_df.head(len(dirty_df)).astype(str)
    repaired_str = repaired_df.astype(str)
    
    tp = ((dirty_str != clean_str) & (repaired_str == clean_str)).sum().sum()
    fp = ((dirty_str == clean_str) & (repaired_str != clean_str)).sum().sum()
    fn = ((dirty_str != clean_str) & (repaired_str != clean_str)).sum().sum()
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'runtime': runtime,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Jellyfish-7B LLM baseline')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/data1/qianc/EMCL/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='/data1/qianc/EMCL/baselines/results',
                        help='Path to output directory')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Maximum number of rows to process (for testing)')
    parser.add_argument('--num_examples', type=int, default=20,
                        help='Number of clean examples to show the model (default: 20)')
    parser.add_argument('--use_8bit', action='store_true',
                        help='Use 8-bit quantization to reduce memory usage')
    
    args = parser.parse_args()
    
    # Setup paths
    dirty_path = os.path.join(args.data_dir, args.dataset, 'dirty.csv')
    clean_path = os.path.join(args.data_dir, args.dataset, 'clean.csv')
    output_path = os.path.join(args.output_dir, 'jellyfish', args.dataset, 'repaired.csv')
    
    print(f"Running Jellyfish-7B on {args.dataset}...")
    print(f"Dirty data: {dirty_path}")
    print(f"Clean data: {clean_path}")
    print(f"Output: {output_path}")
    
    try:
        # Run repair
        results = repair_with_jellyfish(dirty_path, clean_path, output_path, 
                                        args.max_rows, args.num_examples, args.use_8bit)
        
        print(f"\nResults:")
        print(f"  TP: {results['tp']}, FP: {results['fp']}, FN: {results['fn']}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        
        # Save runtime info
        runtime_path = os.path.join(args.output_dir, 'jellyfish', args.dataset, 'runtime.json')
        with open(runtime_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRuntime info saved to: {runtime_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
