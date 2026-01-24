"""
BigDansing Baseline for Data Repair
Uses denial constraints and graph-based repair
"""
import os
import sys
import time
import argparse
import json
import subprocess

# Add bigdansing to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bigdansing_holistic'))


def repair_with_bigdansing(dataset_name, data_dir, output_dir):
    """
    Repair dirty data using BigDansing
    """
    start_time = time.time()
    
    # Setup paths
    dirty_path = os.path.join(data_dir, dataset_name, 'dirty.csv')
    clean_path = os.path.join(data_dir, dataset_name, 'clean.csv')
    
    # Find constraints file
    constraints_files = [
        os.path.join(data_dir, dataset_name, f'{dataset_name}_constraints.txt'),
        os.path.join(data_dir, dataset_name, 'constraints.txt'),
        os.path.join(data_dir, dataset_name, 'dc_rules.txt')
    ]
    
    rule_path = None
    for cf in constraints_files:
        if os.path.exists(cf):
            rule_path = cf
            break
    
    if not rule_path:
        raise FileNotFoundError(f"No constraints file found for {dataset_name}")
    
    # Create output directory
    output_path = os.path.join(output_dir, 'bigdansing', dataset_name, 'repaired.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # BigDansing needs to run from its own directory
    bigdansing_dir = os.path.join(os.path.dirname(__file__), 'bigdansing_holistic')
    
    # Run BigDansing
    cmd = [
        sys.executable, 'bigdansing.py',
        '--clean_path', clean_path,
        '--dirty_path', dirty_path,
        '--rule_path', rule_path,
        '--task_name', dataset_name,
        '--onlyed', '0',
        '--perfected', '0'
    ]
    
    result = subprocess.run(
        cmd,
        cwd=bigdansing_dir,
        capture_output=True,
        text=True,
        timeout=3600
    )
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Copy result from BigDansing output directory to our output directory
    bigdansing_output = os.path.join(bigdansing_dir, 'Repaired_res', 'bigdansing', 
                                     dataset_name[:-1] if len(dataset_name) > 1 else dataset_name,
                                     'repaired.csv')
    
    if os.path.exists(bigdansing_output):
        import shutil
        shutil.copy(bigdansing_output, output_path)
        print(f"Repaired data saved to: {output_path}")
    else:
        print(f"Warning: BigDansing output not found at {bigdansing_output}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    
    return {
        'runtime': runtime,
        'success': result.returncode == 0
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BigDansing baseline')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/data1/qianc/EMCL/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='/data1/qianc/EMCL/baselines/results',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    print(f"Running BigDansing on {args.dataset}...")
    
    try:
        results = repair_with_bigdansing(args.dataset, args.data_dir, args.output_dir)
        
        print(f"\nResults:")
        print(f"  Success: {results['success']}")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        
        # Save runtime info
        runtime_path = os.path.join(args.output_dir, 'bigdansing', args.dataset, 'runtime.json')
        with open(runtime_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRuntime info saved to: {runtime_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

