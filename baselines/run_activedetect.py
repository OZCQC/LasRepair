"""
ActiveDetect Baseline for Data Repair
Error detection + simple repair strategy
"""
import os
import sys
import time
import argparse
import json
import pandas as pd
import numpy as np

# Add activedetect to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'activedetect'))

from activedetect.loaders.csv_loader import CSVLoader
from activedetect.error_detectors.ErrorDetector import ErrorDetector


def repair_with_activedetect(dirty_path, clean_path, output_path):
    """
    Repair dirty data using ActiveDetect for detection + simple mode-based repair
    """
    start_time = time.time()
    
    # Load dirty data
    c = CSVLoader()
    loaded_data = c.loadFile(dirty_path)
    
    # Run error detection
    detector = ErrorDetector(loaded_data)
    detector.fit()
    
    # Collect detected errors
    detected_errors = {}
    for error in detector:
        cell = error['cell']
        detected_errors[cell] = error
    
    # Load data as DataFrame for repair
    dirty_df = pd.read_csv(dirty_path)
    repaired_df = dirty_df.copy()
    
    # Simple repair: replace with column mode (most frequent value)
    for (row_idx, col_idx) in detected_errors.keys():
        if col_idx < len(dirty_df.columns):
            col_name = dirty_df.columns[col_idx]
            # Use mode (most frequent value) for this column
            mode_val = dirty_df[col_name].mode()
            if len(mode_val) > 0:
                repaired_df.at[row_idx, col_name] = mode_val[0]
    
    # Save repaired data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    repaired_df.to_csv(output_path, index=False)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate metrics
    clean_df = pd.read_csv(clean_path)
    
    tp = ((dirty_df != clean_df) & (repaired_df == clean_df)).sum().sum()
    fp = ((dirty_df == clean_df) & (repaired_df != clean_df)).sum().sum()
    fn = ((dirty_df != clean_df) & (repaired_df != clean_df)).sum().sum()
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'runtime': runtime,
        'errors_detected': len(detected_errors)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ActiveDetect baseline')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/data1/qianc/EMCL/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='/data1/qianc/EMCL/baselines/results',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    dirty_path = os.path.join(args.data_dir, args.dataset, 'dirty.csv')
    clean_path = os.path.join(args.data_dir, args.dataset, 'clean.csv')
    output_path = os.path.join(args.output_dir, 'activedetect', args.dataset, 'repaired.csv')
    
    print(f"Running ActiveDetect on {args.dataset}...")
    print(f"Dirty data: {dirty_path}")
    print(f"Clean data: {clean_path}")
    print(f"Output: {output_path}")
    
    try:
        results = repair_with_activedetect(dirty_path, clean_path, output_path)
        
        print(f"\nResults:")
        print(f"  Errors detected: {results['errors_detected']}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        
        # Save runtime info
        runtime_path = os.path.join(args.output_dir, 'activedetect', args.dataset, 'runtime.json')
        with open(runtime_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRuntime info saved to: {runtime_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

