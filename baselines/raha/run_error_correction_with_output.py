#!/usr/bin/env python3
"""
Error Correction Script with Dataset Output
Runs Baran error correction and saves repaired datasets to a new folder
"""

import os
import sys
import time
import pandas as pd
import raha

def create_output_directory():
    """Create output directory for repaired datasets"""
    output_dir = "/root/datarepair/baselines/raha/repaired_datasets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

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
                        "original_name": dataset_name,
                        "dirty_file": dirty_file,
                        "path": os.path.join(dataset_path, dirty_file),
                        "clean_path": os.path.join(dataset_path, clean_file)
                    }
                    datasets.append(dataset_config)
    
    return datasets

def save_repaired_dataset(data, correction_dictionary, output_dir, dataset_config):
    """Save the repaired dataset to the output directory"""
    # Create repaired dataset
    data.create_repaired_dataset(correction_dictionary)
    
    # Create subdirectory for this dataset
    dataset_output_dir = os.path.join(output_dir, dataset_config['original_name'])
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)
    
    # Save repaired dataset
    repaired_filename = f"repaired_{dataset_config['dirty_file']}"
    repaired_path = os.path.join(dataset_output_dir, repaired_filename)
    
    # Use the dataset's built-in method to write CSV
    raha.dataset.Dataset.write_csv_dataset(repaired_path, data.repaired_dataframe)
    
    # Also save correction metadata
    metadata_filename = f"corrections_{dataset_config['dirty_file'].replace('.csv', '.txt')}"
    metadata_path = os.path.join(dataset_output_dir, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        f.write(f"Dataset: {dataset_config['name']}\n")
        f.write(f"Original file: {dataset_config['path']}\n")
        f.write(f"Clean file: {dataset_config['clean_path']}\n")
        f.write(f"Repaired file: {repaired_path}\n")
        f.write(f"Total errors detected: {len(data.detected_cells)}\n")
        f.write(f"Total corrections applied: {len(correction_dictionary)}\n")
        f.write(f"\nCorrections applied:\n")
        f.write("Row,Column,Original_Value,Corrected_Value\n")
        
        for (row, col), corrected_value in correction_dictionary.items():
            original_value = data.dataframe.iloc[row, col]
            f.write(f"{row},{col},{original_value},{corrected_value}\n")
    
    return repaired_path, metadata_path

def run_error_correction_with_output(dataset_dict, output_dir, verbose=True):
    """Run Baran error correction on a single dataset and save results"""
    print(f"\n{'='*80}")
    print(f"RUNNING ERROR CORRECTION ON: {dataset_dict['name']}")
    print(f"{'='*80}")
    
    try:
        # Load dataset and get actual errors
        data = raha.dataset.Dataset(dataset_dict)
        data.detected_cells = dict(data.get_actual_errors_dictionary())
        
        print(f"Dataset shape: {data.dataframe.shape}")
        print(f"Total detected errors: {len(data.detected_cells)}")
        
        if len(data.detected_cells) == 0:
            print("No errors detected in this dataset. Skipping correction.")
            return None
        
        # Run correction with Baran
        app = raha.correction.Correction()
        app.VERBOSE = verbose
        app.LABELING_BUDGET = 20  # Adjust this for faster/slower but more/less accurate results
        
        start_time = time.time()
        correction_dictionary = app.run(data)
        end_time = time.time()
        
        # Save repaired dataset
        repaired_path, metadata_path = save_repaired_dataset(data, correction_dictionary, output_dir, dataset_dict)
        
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
            'error_correction_f1': ec_f1,
            'repaired_file': repaired_path,
            'metadata_file': metadata_path
        }
        
        print(f"\nRESULTS for {dataset_dict['name']}:")
        print(f"  Runtime: {results['runtime_seconds']:.2f} seconds")
        print(f"  Error Detection  - Precision: {ed_precision:.4f}, Recall: {ed_recall:.4f}, F1: {ed_f1:.4f}")
        print(f"  Error Correction - Precision: {ec_precision:.4f}, Recall: {ec_recall:.4f}, F1: {ec_f1:.4f}")
        print(f"  Repaired dataset saved to: {repaired_path}")
        print(f"  Correction metadata saved to: {metadata_path}")
        
        return results
        
    except Exception as e:
        print(f"ERROR processing {dataset_dict['name']}: {str(e)}")
        return {
            'dataset': dataset_dict['name'],
            'error': str(e)
        }

def main():
    """Main function to run error correction on all datasets"""
    print("BARAN ERROR CORRECTION SYSTEM WITH OUTPUT")
    print("Running error correction and saving repaired datasets...")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Repaired datasets will be saved to: {output_dir}")
    
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
        
        # Quick check of dataset size and skip very large ones initially
        try:
            temp_df = pd.read_csv(dataset_dict['path'])
            if temp_df.shape[0] > 10000:
                print(f"  Skipping {dataset_dict['name']} (too large: {temp_df.shape[0]} rows)")
                continue
        except:
            pass
        
        result = run_error_correction_with_output(dataset_dict, output_dir, verbose=False)
        if result:
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
    results_file = os.path.join(output_dir, "baran_correction_summary.csv")
    if successful_results:
        results_df = pd.DataFrame(successful_results)
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
    
    print(f"\nAll repaired datasets are stored in: {output_dir}")

if __name__ == "__main__":
    main()
