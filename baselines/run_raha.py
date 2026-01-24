"""
Raha Baseline for Data Repair
Error detection and correction using Raha
"""
import os
import sys
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore")


# Add raha to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'raha'))

import raha


def repair_with_raha(dirty_path, clean_path, output_path):
    """
    Repair dirty data using Raha
    """
    start_time = time.time()
    
    # Create dataset dictionary
    dataset_dictionary = {
        "name": "dataset",
        "path": dirty_path,
        "clean_path": clean_path
    }
    
    # Load dataset and get actual errors
    data = raha.dataset.Dataset(dataset_dictionary)
    data.detected_cells = dict(data.get_actual_errors_dictionary())
    
    # Run correction
    app = raha.correction.Correction()
    app.VERBOSE = False
    correction_dictionary = app.run(data)
    
    # Create repaired dataset
    data.create_repaired_dataset(correction_dictionary)
    
    # Save repaired data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.repaired_dataframe.to_csv(output_path, index=False)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Evaluate results
    p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
    
    return {
        'precision': p,
        'recall': r,
        'f1': f,
        'runtime': runtime
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Raha baseline')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/data1/qianc/EMCL/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='/data1/qianc/EMCL/baselines/results',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    for i in [0.1, 0.5, 1, 2, 3, 5]:
        dirty_path = os.path.join(args.data_dir, args.dataset, f'hospital_{int(i * 10)}_error.csv')
        clean_path = os.path.join(args.data_dir, args.dataset, 'clean.csv')
        output_path = os.path.join(args.output_dir, 'raha', args.dataset, 'repaired.csv')
        
        print(f"Running Raha on {args.dataset}...")
        print(f"Dirty data: {dirty_path}")
        print(f"Clean data: {clean_path}")
        print(f"Output: {output_path}")
        
        # Run repair
        results = repair_with_raha(dirty_path, clean_path, output_path)
        
        print(f"\nResults:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        
        # Save runtime info
        runtime_path = os.path.join(args.output_dir, 'raha', args.dataset, 'runtime.json')
        with open(runtime_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRuntime info saved to: {runtime_path}")

