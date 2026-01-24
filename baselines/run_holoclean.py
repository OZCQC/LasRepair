"""
HoloClean Baseline for Data Repair
Uses probabilistic inference with denial constraints
"""
import os
import sys
import time
import argparse
import json
import pandas as pd

# Add holoclean to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'holoclean'))

import holoclean
from holoclean.detect import NullDetector, ViolationDetector
from holoclean.repair.featurize import *


def repair_with_holoclean(dataset_name, data_dir, output_dir):
    """
    Repair dirty data using HoloClean
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
    
    constraints_path = None
    for cf in constraints_files:
        if os.path.exists(cf):
            constraints_path = cf
            break
    
    # Create output directory
    output_path = os.path.join(output_dir, 'holoclean', dataset_name, 'repaired.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Setup HoloClean session
    hc = holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=False,
        timeout=3*60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=False
    ).session
    
    # Load data
    hc.load_data(dataset_name, dirty_path)
    
    # Load constraints if available
    if constraints_path:
        hc.load_dcs(constraints_path)
        hc.ds.set_constraints(hc.get_dcs())
    
    # Detect errors
    detectors = [NullDetector(), ViolationDetector()]
    hc.detect_errors(detectors)
    
    # Setup domain and repair
    hc.setup_domain()
    
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        ConstraintFeaturizer(),
    ]
    
    hc.repair_errors(featurizers)
    
    # Get repaired data
    repaired_df = hc.get_repaired_dataset()
    repaired_df.to_csv(output_path, index=False)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate metrics
    clean_df = pd.read_csv(clean_path)
    dirty_df = pd.read_csv(dirty_path)
    
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
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'runtime': runtime
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HoloClean baseline')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/data1/qianc/EMCL/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='/data1/qianc/EMCL/baselines/results',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    print(f"Running HoloClean on {args.dataset}...")
    
    try:
        results = repair_with_holoclean(args.dataset, args.data_dir, args.output_dir)
        
        print(f"\nResults:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        
        # Save runtime info
        runtime_path = os.path.join(args.output_dir, 'holoclean', args.dataset, 'runtime.json')
        with open(runtime_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRuntime info saved to: {runtime_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

