"""
MLNClean Baseline for Data Repair
Uses Markov Logic Networks for data cleaning
Note: This is a Java-based project
"""
import os
import sys
import time
import argparse
import json
import pandas as pd
import subprocess
import shutil


def repair_with_mlnclean(dataset_name, data_dir, output_dir):
    """
    Repair dirty data using MLNClean
    Note: MLNClean is Java-based and requires compilation
    """
    start_time = time.time()
    
    # Setup paths
    dirty_path = os.path.join(data_dir, dataset_name, 'dirty.csv')
    clean_path = os.path.join(data_dir, dataset_name, 'clean.csv')
    output_path = os.path.join(output_dir, 'mlnclean', dataset_name, 'repaired.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mlnclean_dir = os.path.join(os.path.dirname(__file__), 'MLNClean')
    
    # Check if MLNClean is compiled
    jar_file = os.path.join(mlnclean_dir, 'target', 'my-app.jar')
    
    if not os.path.exists(jar_file):
        print(f"MLNClean JAR not found at {jar_file}")
        print("To compile MLNClean, run:")
        print(f"  cd {mlnclean_dir}")
        print("  mvn clean package")
        print("\nFalling back to copying dirty data (no repair)")
        shutil.copy(dirty_path, output_path)
        
        end_time = time.time()
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'runtime': end_time - start_time,
            'note': 'MLNClean requires compilation - mvn clean package'
        }
    
    # Check if dataset-specific files exist in MLNClean structure
    mlnclean_dataset_dir = os.path.join(mlnclean_dir, dataset_name, 'dataset')
    if not os.path.exists(mlnclean_dataset_dir):
        print(f"MLNClean dataset directory not found: {mlnclean_dataset_dir}")
        print("MLNClean requires dataset-specific setup in its directory structure")
        shutil.copy(dirty_path, output_path)
        
        end_time = time.time()
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'runtime': end_time - start_time,
            'note': 'MLNClean requires dataset-specific setup'
        }
    
    try:
        # Run MLNClean (this is a simplified version - actual usage may vary)
        cmd = [
            'java', '-jar', jar_file,
            '-dataset', dataset_name,
            '-dirty', dirty_path,
            '-clean', clean_path
        ]
        
        result = subprocess.run(
            cmd,
            cwd=mlnclean_dir,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        print(f"MLNClean stdout: {result.stdout}")
        if result.stderr:
            print(f"MLNClean stderr: {result.stderr}")
        
        # Look for output in MLNClean's typical output location
        mlnclean_output = os.path.join(mlnclean_dir, dataset_name, 'result', 'repaired.csv')
        if os.path.exists(mlnclean_output):
            shutil.copy(mlnclean_output, output_path)
        else:
            print(f"MLNClean output not found, copying dirty data")
            shutil.copy(dirty_path, output_path)
            
    except Exception as e:
        print(f"Error running MLNClean: {e}")
        shutil.copy(dirty_path, output_path)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate metrics
    try:
        clean_df = pd.read_csv(clean_path)
        dirty_df = pd.read_csv(dirty_path)
        repaired_df = pd.read_csv(output_path)
        
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
        'note': 'Java-based baseline, requires Maven compilation'
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MLNClean baseline')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/data1/qianc/EMCL/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='/data1/qianc/EMCL/baselines/results',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    print(f"Running MLNClean on {args.dataset}...")
    
    try:
        results = repair_with_mlnclean(args.dataset, args.data_dir, args.output_dir)
        
        print(f"\nResults:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        print(f"  Note: {results['note']}")
        
        # Save runtime info
        runtime_path = os.path.join(args.output_dir, 'mlnclean', args.dataset, 'runtime.json')
        with open(runtime_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRuntime info saved to: {runtime_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

