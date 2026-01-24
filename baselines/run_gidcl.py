"""
GIDCL Baseline for Data Repair
Graph-Enhanced Interpretable Data Cleaning with LLMs
Note: This requires pre-trained models and is computationally expensive
"""
import os
import sys
import time
import argparse
import json
import pandas as pd
import shutil

# Add GIDCL to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GIDCL'))


def repair_with_gidcl(dataset_name, data_dir, output_dir):
    """
    Repair dirty data using GIDCL
    Note: This assumes GIDCL correction results are pre-computed
    """
    start_time = time.time()
    
    # Setup paths
    dirty_path = os.path.join(data_dir, dataset_name, 'dirty.csv')
    clean_path = os.path.join(data_dir, dataset_name, 'clean.csv')
    
    # GIDCL expects data in GEIL_Data format
    gidcl_data_dir = os.path.join(os.path.dirname(__file__), 'GIDCL', 'GEIL_Data', dataset_name)
    gidcl_correction_path = os.path.join(gidcl_data_dir, 'correction', 'result', 'correction.csv')
    
    output_path = os.path.join(output_dir, 'gidcl', dataset_name, 'repaired.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if pre-computed results exist
    if os.path.exists(gidcl_correction_path):
        print(f"Using pre-computed GIDCL results from {gidcl_correction_path}")
        shutil.copy(gidcl_correction_path, output_path)
    else:
        print(f"Warning: GIDCL pre-computed results not found at {gidcl_correction_path}")
        print("GIDCL requires:")
        print("  1. Error detection model training")
        print("  2. LLM fine-tuning for correction")
        print("  3. Graph embedding and refinement")
        print("\nFor full GIDCL pipeline, please refer to GIDCL/README.md")
        print("\nFalling back to copying dirty data (no repair)")
        shutil.copy(dirty_path, output_path)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate metrics
    try:
        clean_df = pd.read_csv(clean_path)
        dirty_df = pd.read_csv(dirty_path)
        repaired_df = pd.read_csv(output_path)
        
        # Align columns
        common_cols = list(set(clean_df.columns) & set(dirty_df.columns) & set(repaired_df.columns))
        clean_df = clean_df[common_cols]
        dirty_df = dirty_df[common_cols]
        repaired_df = repaired_df[common_cols]
        
        tp = ((dirty_df != clean_df) & (repaired_df == clean_df)).sum().sum()
        fp = ((dirty_df == clean_df) & (repaired_df != clean_df)).sum().sum()
        fn = ((dirty_df != clean_df) & (repaired_df != clean_df)).sum().sum()
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        precision = recall = f1 = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'runtime': runtime,
        'note': 'Uses pre-computed results or requires full GIDCL setup'
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GIDCL baseline')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/data1/qianc/EMCL/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='/data1/qianc/EMCL/baselines/results',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    print(f"Running GIDCL on {args.dataset}...")
    
    try:
        results = repair_with_gidcl(args.dataset, args.data_dir, args.output_dir)
        
        print(f"\nResults:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        print(f"  Note: {results['note']}")
        
        # Save runtime info
        runtime_path = os.path.join(args.output_dir, 'gidcl', args.dataset, 'runtime.json')
        with open(runtime_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRuntime info saved to: {runtime_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

