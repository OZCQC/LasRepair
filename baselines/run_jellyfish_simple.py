"""
Jellyfish Simple Baseline for Data Repair
Uses string similarity (jellyfish library) for simple typo correction
Fast and lightweight - no LLM required
"""
import pandas as pd
import jellyfish
import os
import time
import argparse
import json
from collections import Counter

def repair_with_jellyfish(dirty_path, clean_path, output_path):
    """
    Repair dirty data using simple string similarity
    """
    start_time = time.time()
    
    # Load datasets
    dirty_df = pd.read_csv(dirty_path, dtype=str)
    clean_df = pd.read_csv(clean_path, dtype=str)
    repaired_df = dirty_df.copy()
    
    print(f"Processing {len(dirty_df)} rows x {len(dirty_df.columns)} columns...")
    
    # Process each column
    for col in dirty_df.columns:
        # Get most common values as candidates (likely correct values)
        value_counts = dirty_df[col].value_counts()
        candidates = value_counts.head(20).index.tolist()  # Top 20 most frequent
        
        # For each cell in this column
        for idx in range(len(dirty_df)):
            value = str(dirty_df.at[idx, col])
            
            if value == 'nan' or len(value) == 0:
                continue
            
            # If value is already common, keep it
            if value in candidates[:5]:  # Top 5 are likely correct
                continue
            
            # Find best matching candidate
            best_match = value
            best_score = 0
            
            for candidate in candidates:
                if candidate == 'nan':
                    continue
                
                # Use Jaro-Winkler similarity
                score = jellyfish.jaro_winkler_similarity(value, candidate)
                
                # Only replace if very similar (likely typo)
                if score > best_score and score >= 0.92:
                    best_score = score
                    best_match = candidate
            
            if best_match != value:
                repaired_df.at[idx, col] = best_match
    
    # Save repaired data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    repaired_df.to_csv(output_path, index=False)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate metrics
    dirty_str = dirty_df.astype(str)
    clean_str = clean_df.astype(str)
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
    parser = argparse.ArgumentParser(description='Run Jellyfish Simple baseline')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/data1/qianc/EMCL/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='/data1/qianc/EMCL/baselines/results',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    dirty_path = os.path.join(args.data_dir, args.dataset, 'dirty.csv')
    clean_path = os.path.join(args.data_dir, args.dataset, 'clean.csv')
    output_path = os.path.join(args.output_dir, 'jellyfish_simple', args.dataset, 'repaired.csv')
    
    print(f"Running Jellyfish Simple on {args.dataset}...")
    print(f"Dirty data: {dirty_path}")
    print(f"Clean data: {clean_path}")
    print(f"Output: {output_path}")
    
    try:
        # Run repair
        results = repair_with_jellyfish(dirty_path, clean_path, output_path)
        
        print(f"\nResults:")
        print(f"  TP: {results['tp']}, FP: {results['fp']}, FN: {results['fn']}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        
        # Save runtime info
        runtime_path = os.path.join(args.output_dir, 'jellyfish_simple', args.dataset, 'runtime.json')
        os.makedirs(os.path.dirname(runtime_path), exist_ok=True)
        with open(runtime_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRuntime info saved to: {runtime_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

