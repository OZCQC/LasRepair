#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from collections import Counter

# Add the activedetect path
sys.path.append('baselines/activedetect')

from activedetect.error_detectors.ErrorDetector import ErrorDetector
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule

def load_data(dirty_path, clean_path, max_rows=1000):
    """Load dirty and clean datasets"""
    print("Loading data from {} and {}".format(dirty_path, clean_path))
    
    # Load data
    dirty_df = pd.read_csv(dirty_path).head(max_rows)
    clean_df = pd.read_csv(clean_path).head(max_rows)
    
    print("Loaded {} rows with {} columns".format(len(dirty_df), len(dirty_df.columns)))
    
    # Convert to list of lists format for error detection
    dirty_data = dirty_df.values.tolist()
    clean_data = clean_df.values.tolist()
    
    # Convert all values to strings for consistency
    dirty_data = [[str(cell) if pd.notna(cell) else "" for cell in row] for row in dirty_data]
    clean_data = [[str(cell) if pd.notna(cell) else "" for cell in row] for row in clean_data]
    
    return dirty_data, clean_data, dirty_df.columns.tolist()

def detect_errors(data):
    """Use ActiveDetect to detect errors in the data"""
    print("Detecting errors using ActiveDetect...")
    
    try:
        # Configure error detection modules
        modules = [QuantitativeErrorModule, PuncErrorModule]
        config = [{'thresh': 2.5}, {}]  # Lower threshold for more sensitivity
        
        # Create error detector
        detector = ErrorDetector(data, modules=modules, config=config, use_word2vec=False)
        detector.fit()
        
        # Get detector function
        detect_fn = detector.getDetectorFunction()
        
        # Collect error locations
        error_cells = set()
        for i, row in enumerate(data):
            is_error, col_idx = detect_fn(row)
            if is_error and col_idx >= 0:
                error_cells.add((i, col_idx))
        
        print("Detected {} potential error cells".format(len(error_cells)))
        return error_cells, detect_fn
        
    except Exception as e:
        print("Error detection failed: {}".format(e))
        return set(), None

def simple_repair(data, error_cells):
    """Apply simple repair strategies to detected errors"""
    print("Applying simple repair strategies...")
    
    repaired_data = [row[:] for row in data]  # Deep copy
    
    # Group errors by column
    errors_by_col = {}
    for row_idx, col_idx in error_cells:
        if col_idx not in errors_by_col:
            errors_by_col[col_idx] = []
        errors_by_col[col_idx].append(row_idx)
    
    # Repair each column
    for col_idx, error_rows in errors_by_col.items():
        print("Repairing column {} with {} errors".format(col_idx, len(error_rows)))
        
        # Get clean values from the column (excluding error rows)
        clean_values = []
        for row_idx, row in enumerate(data):
            if row_idx not in error_rows and col_idx < len(row):
                val = row[col_idx]
                if val and str(val).strip() not in ['', 'N/A', 'NULL', 'null', '?', 'nan']:
                    clean_values.append(str(val).strip())
        
        if not clean_values:
            continue
        
        # Determine repair strategy based on data type
        repair_value = get_repair_value(clean_values)
        
        # Apply repair
        for row_idx in error_rows:
            if row_idx < len(repaired_data) and col_idx < len(repaired_data[row_idx]):
                repaired_data[row_idx][col_idx] = repair_value
                
    return repaired_data

def get_repair_value(clean_values):
    """Determine the best repair value based on the data characteristics"""
    # Try to determine if it's numeric
    numeric_values = []
    for val in clean_values:
        try:
            numeric_values.append(float(val))
        except (ValueError, TypeError):
            pass
    
    # If most values are numeric, use median
    if len(numeric_values) > len(clean_values) * 0.7:
        return str(np.median(numeric_values))
    else:
        # For categorical data, use most common value
        counter = Counter(clean_values)
        return counter.most_common(1)[0][0]

def calculate_f1_score(original_data, corrected_data, ground_truth_data):
    """Calculate F1 score at cell level"""
    total_cells = 0
    tp = 0  # True positives: correctly repaired cells
    fp = 0  # False positives: incorrectly "repaired" clean cells
    fn = 0  # False negatives: errors that weren't fixed
    tn = 0  # True negatives: clean cells left unchanged
    
    for i in range(len(original_data)):
        for j in range(min(len(original_data[i]), len(corrected_data[i]), len(ground_truth_data[i]))):
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
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nError Correction Results:")
    print("Total cells: {}".format(total_cells))
    print("True Positives (correctly fixed): {}".format(tp))
    print("False Positives (incorrectly changed): {}".format(fp))
    print("False Negatives (errors not fixed): {}".format(fn))
    print("True Negatives (correctly unchanged): {}".format(tn))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 Score: {:.4f}".format(f1))
    
    return f1, precision, recall

def calculate_error_statistics(dirty_data, clean_data):
    """Calculate basic error statistics"""
    error_count = 0
    total_cells = 0
    
    for i in range(len(dirty_data)):
        for j in range(min(len(dirty_data[i]), len(clean_data[i]))):
            total_cells += 1
            if dirty_data[i][j] != clean_data[i][j]:
                error_count += 1
    
    error_rate = error_count / float(total_cells) if total_cells > 0 else 0
    
    print("\nDataset Statistics:")
    print("Total cells: {}".format(total_cells))
    print("Error cells: {}".format(error_count))
    print("Error rate: {:.4f}".format(error_rate))
    
    return error_count, total_cells, error_rate

def main():
    print("Starting Simple Error Correction with ActiveDetect")
    
    # Load data (using first 1000 rows for faster testing)
    dirty_data, clean_data, column_names = load_data(
        'datasets/movies/dirty.csv', 
        'datasets/movies/clean.csv', 
        max_rows=1000
    )
    
    # Calculate basic error statistics
    error_count, total_cells, error_rate = calculate_error_statistics(dirty_data, clean_data)
    
    # Detect errors using ActiveDetect
    error_cells, detect_fn = detect_errors(dirty_data)
    
    if not error_cells:
        print("No errors detected. Trying with relaxed threshold...")
        # Try with more relaxed threshold
        try:
            modules = [QuantitativeErrorModule, PuncErrorModule]
            config = [{'thresh': 1.0}, {}]  # Much lower threshold
            detector = ErrorDetector(dirty_data, modules=modules, config=config, use_word2vec=False)
            detector.fit()
            detect_fn = detector.getDetectorFunction()
            
            for i, row in enumerate(dirty_data):
                is_error, col_idx = detect_fn(row)
                if is_error and col_idx >= 0:
                    error_cells.add((i, col_idx))
            
            print("Detected {} potential error cells with relaxed threshold".format(len(error_cells)))
        except Exception as e:
            print("Relaxed detection also failed: {}".format(e))
    
    if error_cells:
        # Apply simple repair
        corrected_data = simple_repair(dirty_data, error_cells)
        
        # Calculate F1 score
        f1, precision, recall = calculate_f1_score(dirty_data, corrected_data, clean_data)
        
        print("\nFinal Results:")
        print("F1 Score: {:.4f}".format(f1))
        
    else:
        print("No errors detected, cannot perform correction")
        print("This might indicate:")
        print("1. The error detection thresholds are too strict")
        print("2. The data doesn't contain the types of errors these detectors look for")
        print("3. The error detectors need different configuration")

if __name__ == "__main__":
    main() 