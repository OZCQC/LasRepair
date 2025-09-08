#!/usr/bin/env python3
"""
Comprehensive Error Correction Script using Baran System
Runs Baran error correction on all available datasets and evaluates performance
"""

import os
import sys
import time
import pandas as pd
import raha

def get_available_datasets(datasets_root="/root/datarepair/datasets"):
    """Find all available datasets with both dirty and clean files"""
    datasets = []
    
    for dataset_name in os.listdir(datasets_root):
        dataset_path = os.path.join(datasets_root, dataset_name)
        if os.path.isdir(dataset_path):
            # Check for different dirty file patterns
            dirty_files = []
            clean_file = None
            
            # Look for dirty files (dirty.csv, dirty_20.csv, etc.)
            for file in os.listdir(dataset_path):
                if file.startswith("dirty") and file.endswith(".csv"):
                    dirty_files.append(file)
                elif file == "clean.csv":
                    clean_file = file
            
            # Add valid dataset configurations
            if clean_file:
                for dirty_file in dirty_files:
                    dataset_config = {
                        "name": f"{dataset_name}_{dirty_file.replace('.csv', '')}",
                        "path": os.path.join(dataset_path, dirty_file),
                        "clean_path": os.path.join(dataset_path, clean_file)
                    }
                    datasets.append(dataset_config)
    
    return datasets

def run_error_correction(dataset_dict, verbose=True):
    """Run Baran error correction on a single dataset"""
    print(f"\n{'='*80}")
    print(f"RUNNING ERROR CORRECTION ON: {dataset_dict['name']}")
    print(f"{'='*80}")
    
    try:
        # Load dataset and get actual errors
        data = raha.dataset.Dataset(dataset_dict)
        data.detected_cells = dict(data.get_actual_errors_dictionary())
        
        print(f"Dataset shape: {data.dataframe.shape}")
        print(f"Total detected errors: {len(data.detected_cells)}")
        
        # Run correction with Baran
        app = raha.correction.Correction()
        app.VERBOSE = verbose
        app.LABELING_BUDGET = 20  # Adjust this for faster/slower but more/less accurate results
        
        start_time = time.time()
        correction_dictionary = app.run(data)
        end_time = time.time()
        
        # Evaluate results
        evaluation_results = data.get_data_cleaning_evaluation(correction_dictionary)
        ed_precision, ed_recall, ed_f1, ec_precision, ec_recall, ec_f1 = evaluation_results
        
        results = {
            'dataset': dataset_dict['name'],
            'dataset_shape': data.dataframe.shape,
            'total_errors': len(data.detected_cells),
            'corrected_errors': len(correction_dictionary),
            'runtime_seconds': end_time - start_time,
            'error_detection_precision': ed_precision,
            'error_detection_recall': ed_recall,
            'error_detection_f1': ed_f1,
            'error_correction_precision': ec_precision,
            'error_correction_recall': ec_recall,
            'error_correction_f1': ec_f1
        }
        
        print(f"\nRESULTS for {dataset_dict['name']}:")
        print(f"  Runtime: {results['runtime_seconds']:.2f} seconds")
        print(f"  Error Detection  - Precision: {ed_precision:.4f}, Recall: {ed_recall:.4f}, F1: {ed_f1:.4f}")
        print(f"  Error Correction - Precision: {ec_precision:.4f}, Recall: {ec_recall:.4f}, F1: {ec_f1:.4f}")
        
        return results
        
    except Exception as e:
        print(f"ERROR processing {dataset_dict['name']}: {str(e)}")
        return {
            'dataset': dataset_dict['name'],
            'error': str(e)
        }

def main():
    """Main function to run error correction on all datasets"""
    print("BARAN ERROR CORRECTION SYSTEM")
    print("Running error correction on all available datasets...")
    
    # Get all available datasets
    datasets = get_available_datasets()
    print(f"\nFound {len(datasets)} dataset configurations:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset['name']}")
    
    # Results storage
    all_results = []
    
    # Process each dataset
    for i, dataset_dict in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing {dataset_dict['name']}...")
        
        # Skip very large datasets initially for testing
        try:
            # Quick check of dataset size
            temp_df = pd.read_csv(dataset_dict['path'])
            if temp_df.shape[0] > 10000:  # Skip very large datasets initially
                print(f"  Skipping {dataset_dict['name']} (too large: {temp_df.shape[0]} rows)")
                continue
        except:
            pass
        
        result = run_error_correction(dataset_dict, verbose=False)  # Set to True for detailed output
        all_results.append(result)
    
    # Summary report
    print(f"\n{'='*80}")
    print("FINAL SUMMARY REPORT")
    print(f"{'='*80}")
    
    successful_results = [r for r in all_results if 'error' not in r]
    failed_results = [r for r in all_results if 'error' in r]
    
    if successful_results:
        print(f"\nSuccessfully processed {len(successful_results)} datasets:")
        print(f"{'Dataset':<30} {'Errors':<8} {'Runtime':<10} {'Correction F1':<15}")
        print(f"{'-'*70}")
        
        for result in successful_results:
            print(f"{result['dataset']:<30} {result['total_errors']:<8} {result['runtime_seconds']:<10.2f} {result['error_correction_f1']:<15.4f}")
        
        # Calculate averages
        avg_correction_f1 = sum(r['error_correction_f1'] for r in successful_results) / len(successful_results)
        avg_detection_f1 = sum(r['error_detection_f1'] for r in successful_results) / len(successful_results)
        total_runtime = sum(r['runtime_seconds'] for r in successful_results)
        
        print(f"{'-'*70}")
        print(f"{'AVERAGES':<30} {'':<8} {total_runtime:<10.2f} {avg_correction_f1:<15.4f}")
        print(f"\nAverage Error Detection F1: {avg_detection_f1:.4f}")
        print(f"Average Error Correction F1: {avg_correction_f1:.4f}")
        print(f"Total Runtime: {total_runtime:.2f} seconds")
    
    if failed_results:
        print(f"\n\nFailed to process {len(failed_results)} datasets:")
        for result in failed_results:
            print(f"  - {result['dataset']}: {result['error']}")
    
    # Save detailed results
    results_file = "/root/datarepair/baran_error_correction_results.csv"
    if successful_results:
        results_df = pd.DataFrame(successful_results)
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
