#!/usr/bin/env python3
"""
Single Dataset Error Correction Script
Runs Baran error correction on a single dataset and saves the repaired version
"""

import os
import sys
import time
import raha

def correct_and_save_dataset(dataset_name, dirty_filename="dirty.csv", output_dir=None):
    """
    Run error correction on a single dataset and save the repaired version
    
    Args:
        dataset_name: Name of the dataset folder (e.g., 'beers', 'movies')
        dirty_filename: Name of the dirty file (default: 'dirty.csv')
        output_dir: Where to save repaired datasets (default: './repaired_datasets')
    """
    
    # Setup paths
    if output_dir is None:
        output_dir = "/root/datarepair/baselines/raha/repaired_datasets"
    
    datasets_root = "/root/datarepair/datasets"
    dataset_path = os.path.join(datasets_root, dataset_name)
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)
    
    # Setup dataset configuration
    dataset_dict = {
        "name": dataset_name,
        "path": os.path.join(dataset_path, dirty_filename),
        "clean_path": os.path.join(dataset_path, "clean.csv")
    }
    
    # Verify files exist
    if not os.path.exists(dataset_dict["path"]):
        print(f"Error: Dirty file not found: {dataset_dict['path']}")
        return None
    
    if not os.path.exists(dataset_dict["clean_path"]):
        print(f"Error: Clean file not found: {dataset_dict['clean_path']}")
        return None
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Dirty file: {dataset_dict['path']}")
    print(f"Clean file: {dataset_dict['clean_path']}")
    
    try:
        # Load dataset and get actual errors
        data = raha.dataset.Dataset(dataset_dict)
        data.detected_cells = dict(data.get_actual_errors_dictionary())
        
        print(f"Dataset shape: {data.dataframe.shape}")
        print(f"Total detected errors: {len(data.detected_cells)}")
        
        if len(data.detected_cells) == 0:
            print("No errors detected in this dataset. Creating copy as 'repaired'.")
            repaired_path = os.path.join(dataset_output_dir, f"repaired_{dirty_filename}")
            raha.dataset.Dataset.write_csv_dataset(repaired_path, data.dataframe)
            return {
                'dataset': dataset_name,
                'total_errors': 0,
                'corrected_errors': 0,
                'repaired_file': repaired_path
            }
        
        # Run correction with Baran
        app = raha.correction.Correction()
        app.VERBOSE = True  # Show progress
        app.LABELING_BUDGET = 20
        
        print(f"\nStarting Baran error correction...")
        start_time = time.time()
        correction_dictionary = app.run(data)
        end_time = time.time()
        
        # Create repaired dataset
        data.create_repaired_dataset(correction_dictionary)
        
        # Save repaired dataset
        repaired_filename = f"repaired_{dirty_filename}"
        repaired_path = os.path.join(dataset_output_dir, repaired_filename)
        raha.dataset.Dataset.write_csv_dataset(repaired_path, data.repaired_dataframe)
        
        # Save correction metadata
        metadata_filename = f"corrections_{dirty_filename.replace('.csv', '.txt')}"
        metadata_path = os.path.join(dataset_output_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Original file: {dataset_dict['path']}\n")
            f.write(f"Clean file: {dataset_dict['clean_path']}\n")
            f.write(f"Repaired file: {repaired_path}\n")
            f.write(f"Processing time: {end_time - start_time:.2f} seconds\n")
            f.write(f"Total errors detected: {len(data.detected_cells)}\n")
            f.write(f"Total corrections applied: {len(correction_dictionary)}\n")
            f.write(f"\nCorrections applied (Row,Column,Original_Value,Corrected_Value):\n")
            
            for (row, col), corrected_value in correction_dictionary.items():
                original_value = data.dataframe.iloc[row, col]
                f.write(f"{row},{col},\"{original_value}\",\"{corrected_value}\"\n")
        
        # Evaluate results
        evaluation_results = data.get_data_cleaning_evaluation(correction_dictionary)
        ed_precision, ed_recall, ed_f1, ec_precision, ec_recall, ec_f1 = evaluation_results
        
        # Print results
        print(f"\n{'='*60}")
        print(f"CORRECTION COMPLETED FOR: {dataset_name}")
        print(f"{'='*60}")
        print(f"Runtime: {end_time - start_time:.2f} seconds")
        print(f"Total errors detected: {len(data.detected_cells)}")
        print(f"Total corrections applied: {len(correction_dictionary)}")
        print(f"Error Detection  - Precision: {ed_precision:.4f}, Recall: {ed_recall:.4f}, F1: {ed_f1:.4f}")
        print(f"Error Correction - Precision: {ec_precision:.4f}, Recall: {ec_recall:.4f}, F1: {ec_f1:.4f}")
        print(f"Repaired dataset saved to: {repaired_path}")
        print(f"Correction metadata saved to: {metadata_path}")
        
        return {
            'dataset': dataset_name,
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
        
    except Exception as e:
        print(f"ERROR processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function with command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python correct_single_dataset.py <dataset_name> [dirty_filename]")
        print("Example: python correct_single_dataset.py beers")
        print("Example: python correct_single_dataset.py shuttle dirty_20.csv")
        
        # Show available datasets
        datasets_root = "/root/datarepair/datasets"
        if os.path.exists(datasets_root):
            print(f"\nAvailable datasets in {datasets_root}:")
            for dataset in os.listdir(datasets_root):
                dataset_path = os.path.join(datasets_root, dataset)
                if os.path.isdir(dataset_path):
                    print(f"  - {dataset}")
        return
    
    dataset_name = sys.argv[1]
    dirty_filename = sys.argv[2] if len(sys.argv) > 2 else "dirty.csv"
    
    result = correct_and_save_dataset(dataset_name, dirty_filename)
    
    if result:
        print(f"\nSuccess! Repaired dataset available at: {result['repaired_file']}")
    else:
        print(f"\nFailed to process dataset: {dataset_name}")

if __name__ == "__main__":
    main()
