#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

# Add the activedetect path
sys.path.append('baselines/activedetect')

from activedetect.learning.BoostClean import BoostClean
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule
from activedetect.reporting.CSVLogging import CSVLogging

def load_data(dirty_path, clean_path, max_rows=1000):
    """Load dirty and clean datasets"""
    print(f"Loading data from {dirty_path} and {clean_path}")
    
    # Load data
    dirty_df = pd.read_csv(dirty_path).head(max_rows)
    clean_df = pd.read_csv(clean_path).head(max_rows)
    
    print(f"Loaded {len(dirty_df)} rows with {len(dirty_df.columns)} columns")
    
    # Convert to list of lists format that BoostClean expects
    dirty_data = dirty_df.values.tolist()
    clean_data = clean_df.values.tolist()
    
    # Convert all values to strings (BoostClean expects string data)
    dirty_data = [[str(cell) for cell in row] for row in dirty_data]
    clean_data = [[str(cell) for cell in row] for row in clean_data]
    
    return dirty_data, clean_data, dirty_df.columns.tolist()

def create_synthetic_labels(data):
    """Create synthetic binary labels for BoostClean (required for the base model)"""
    # For demonstration, create random binary labels
    # In a real scenario, these would be meaningful classification labels
    np.random.seed(42)
    return [np.random.randint(0, 2) for _ in range(len(data))]

def calculate_cell_level_f1(original_data, corrected_data, ground_truth_data):
    """Calculate F1 score at cell level"""
    total_cells = 0
    tp = 0  # True positives: correctly repaired cells
    fp = 0  # False positives: incorrectly "repaired" clean cells
    fn = 0  # False negatives: errors that weren't fixed
    tn = 0  # True negatives: clean cells left unchanged
    
    for i in range(len(original_data)):
        for j in range(len(original_data[i])):
            if i < len(corrected_data) and j < len(corrected_data[i]) and \
               i < len(ground_truth_data) and j < len(ground_truth_data[i]):
                
                original_val = original_data[i][j]
                corrected_val = corrected_data[i][j]
                truth_val = ground_truth_data[i][j]
                
                total_cells += 1
                
                was_error = (original_val != truth_val)
                was_corrected = (corrected_val != original_val)
                is_now_correct = (corrected_val == truth_val)
                
                if was_error and was_corrected and is_now_correct:
                    tp += 1  # Correctly fixed an error
                elif not was_error and was_corrected:
                    fp += 1  # "Fixed" something that wasn't broken
                elif was_error and not is_now_correct:
                    fn += 1  # Failed to fix an error
                else:
                    tn += 1  # Correctly left alone
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nCell-level Error Correction Results:")
    print(f"Total cells: {total_cells}")
    print(f"True Positives (correctly fixed): {tp}")
    print(f"False Positives (incorrectly changed): {fp}")
    print(f"False Negatives (errors not fixed): {fn}")
    print(f"True Negatives (correctly unchanged): {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return f1, precision, recall

def main():
    print("Starting BoostClean Error Correction")
    
    # Load data (using first 1000 rows for faster testing)
    dirty_data, clean_data, column_names = load_data(
        'datasets/movies/dirty.csv', 
        'datasets/movies/clean.csv', 
        max_rows=1000
    )
    
    # Create synthetic labels for the base model
    labels = create_synthetic_labels(dirty_data)
    
    print(f"Created {len(labels)} synthetic labels")
    
    # Set up BoostClean
    print("Setting up BoostClean...")
    
    # Configure error detection modules
    q_detect = QuantitativeErrorModule
    punc_detect = PuncErrorModule
    config = [{'thresh': 10}, {}]  # Configuration for the modules
    
    # Create logger
    logger = CSVLogging("boostclean_experiment.log")
    
    # Initialize BoostClean
    boostclean = BoostClean(
        modules=[q_detect, punc_detect],
        config=config,
        base_model=RandomForestClassifier(n_estimators=10, random_state=42),
        features=dirty_data,
        labels=labels,
        logging=logger
    )
    
    print("Running BoostClean ensemble...")
    
    # Run BoostClean (j=3 means 3 boosting rounds)
    ensemble = boostclean.run(j=3)
    
    print(f"BoostClean completed with {len(ensemble)} ensemble members")
    
    # The BoostClean in this implementation focuses on classification, not direct data repair
    # For demonstration, we'll show how to extract the "corrected" predictions
    # In practice, you'd need to adapt this to get actual cell corrections
    
    print("Evaluating results...")
    
    # Note: BoostClean is designed for improving classification accuracy with noisy data
    # rather than direct data repair. For actual data repair, you would need to:
    # 1. Use the error detection to identify cells
    # 2. Apply correction strategies based on the detected errors
    
    # For this demo, let's at least show that we can identify the setup works
    print("BoostClean setup and execution completed successfully!")
    print("Note: BoostClean is primarily designed for classification with noisy data,")
    print("not direct data repair. For cell-level corrections, you would typically")
    print("combine it with specific repair strategies.")
    
    # Calculate how many errors exist in the original data
    error_count = 0
    total_cells = 0
    for i in range(len(dirty_data)):
        for j in range(len(dirty_data[i])):
            if i < len(clean_data) and j < len(clean_data[i]):
                total_cells += 1
                if dirty_data[i][j] != clean_data[i][j]:
                    error_count += 1
    
    error_rate = error_count / total_cells if total_cells > 0 else 0
    print(f"\nDataset Statistics:")
    print(f"Total cells: {total_cells}")
    print(f"Error cells: {error_count}")
    print(f"Error rate: {error_rate:.4f}")

if __name__ == "__main__":
    main() 