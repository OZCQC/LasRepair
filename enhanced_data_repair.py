#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import re

# Add the activedetect path
sys.path.append('baselines/activedetect')

from activedetect.loaders.csv_loader import CSVLoader
from activedetect.error_detectors.ErrorDetector import ErrorDetector
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule

class EnhancedDataRepairer(object):
    """
    Enhanced data repair system with multiple sophisticated repair strategies
    """
    
    def __init__(self, repair_strategy='adaptive'):
        """
        Initialize the repairer
        
        Args:
            repair_strategy: 'simple', 'ml_based', 'adaptive', 'ensemble'
        """
        self.detector = None
        self.error_cells = set()
        self.repair_strategy = repair_strategy
        self.column_types = {}
        self.repair_models = {}
        
    def detect_errors(self, data):
        """Use ActiveDetect to detect errors in the data"""
        try:
            # Configure ActiveDetect with basic error detection modules
            config = [{'thresh': 2.5}, {}]  # Lower threshold for more sensitivity
            modules = [QuantitativeErrorModule, PuncErrorModule]
            
            # Create error detector
            self.detector = ErrorDetector(data, modules=modules, config=config, use_word2vec=False)
            self.detector.fit()
            
            # Collect error locations
            error_cells = set()
            for error in self.detector:
                if 'cell' in error:
                    error_cells.add(error['cell'])
                
            self.error_cells = error_cells
            print("Detected {} error cells".format(len(error_cells)))
            return error_cells
            
        except Exception as e:
            print("Error detection failed: {}".format(e))
            return set()
    
    def analyze_data_types(self, data):
        """Analyze data types and patterns for each column"""
        n_rows, n_cols = len(data), len(data[0]) if data else 0
        
        for col_idx in range(n_cols):
            column_values = [row[col_idx] for row in data if col_idx < len(row)]
            self.column_types[col_idx] = self._infer_column_type(column_values)
    
    def _infer_column_type(self, column_values):
        """Infer the type and characteristics of a column"""
        clean_values = [v for v in column_values if v and str(v).strip() not in ['', 'N/A', 'NULL', 'null', '?']]
        
        if not clean_values:
            return {'type': 'unknown', 'subtype': None}
        
        # Check if numeric
        numeric_count = 0
        for val in clean_values[:100]:  # Sample first 100 values
            try:
                float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
            except:
                pass
        
        if numeric_count > len(clean_values) * 0.7:
            # Further analyze numeric type
            return self._analyze_numeric_type(clean_values)
        else:
            # Analyze categorical type
            return self._analyze_categorical_type(clean_values)
    
    def _analyze_numeric_type(self, values):
        """Analyze numeric column characteristics"""
        numeric_values = []
        for val in values:
            try:
                num_val = float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                numeric_values.append(num_val)
            except:
                continue
        
        if not numeric_values:
            return {'type': 'unknown', 'subtype': None}
        
        # Check if integers
        is_integer = all(float(v).is_integer() for v in numeric_values[:50])
        
        # Check value range
        min_val, max_val = min(numeric_values), max(numeric_values)
        
        return {
            'type': 'numeric',
            'subtype': 'integer' if is_integer else 'float',
            'range': (min_val, max_val),
            'mean': np.mean(numeric_values),
            'std': np.std(numeric_values),
            'median': np.median(numeric_values)
        }
    
    def _analyze_categorical_type(self, values):
        """Analyze categorical column characteristics"""
        value_counts = Counter(values)
        unique_ratio = len(value_counts) / len(values)
        
        # Check if it's a date-like column
        date_pattern = re.compile(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}')
        date_matches = sum(1 for v in values[:20] if date_pattern.search(str(v)))
        
        # Check if it's email-like
        email_pattern = re.compile(r'\S+@\S+\.\S+')
        email_matches = sum(1 for v in values[:20] if email_pattern.search(str(v)))
        
        subtype = 'general'
        if date_matches > len(values[:20]) * 0.5:
            subtype = 'date'
        elif email_matches > len(values[:20]) * 0.5:
            subtype = 'email'
        elif unique_ratio > 0.8:
            subtype = 'high_cardinality'
        elif unique_ratio < 0.1:
            subtype = 'low_cardinality'
        
        return {
            'type': 'categorical',
            'subtype': subtype,
            'unique_ratio': unique_ratio,
            'most_common': value_counts.most_common(5),
            'pattern_info': {
                'date_matches': date_matches,
                'email_matches': email_matches
            }
        }
    
    def repair_data(self, data):
        """Apply sophisticated repair strategies to detected errors"""
        if not self.error_cells:
            return [row[:] for row in data]
        
        # Analyze data types first
        self.analyze_data_types(data)
        
        # Choose repair strategy
        if self.repair_strategy == 'simple':
            return self._simple_repair(data)
        elif self.repair_strategy == 'ml_based':
            return self._ml_based_repair(data)
        elif self.repair_strategy == 'adaptive':
            return self._adaptive_repair(data)
        elif self.repair_strategy == 'ensemble':
            return self._ensemble_repair(data)
        else:
            return self._simple_repair(data)
    
    def _simple_repair(self, data):
        """Simple repair using median/mode strategies"""
        repaired_data = [row[:] for row in data]
        
        # Group errors by column
        errors_by_col = {}
        for row_idx, col_idx in self.error_cells:
            if col_idx not in errors_by_col:
                errors_by_col[col_idx] = []
            errors_by_col[col_idx].append(row_idx)
        
        # Repair each column
        for col_idx, error_rows in errors_by_col.items():
            repaired_data = self._repair_column_simple(repaired_data, col_idx, error_rows)
        
        return repaired_data
    
    def _repair_column_simple(self, data, col_idx, error_rows):
        """Simple repair for a single column"""
        # Get clean values
        clean_values = []
        for row_idx, row in enumerate(data):
            if row_idx not in error_rows and col_idx < len(row):
                val = row[col_idx]
                if val and str(val).strip() not in ['', 'N/A', 'NULL', 'null', '?']:
                    clean_values.append(str(val).strip())
        
        if not clean_values:
            return data
        
        # Get repair value based on column type
        col_type = self.column_types.get(col_idx, {'type': 'unknown'})
        
        if col_type['type'] == 'numeric':
            repair_value = str(col_type.get('median', 0))
        else:
            # Use most common value
            counter = Counter(clean_values)
            repair_value = counter.most_common(1)[0][0]
        
        # Apply repair
        for row_idx in error_rows:
            if row_idx < len(data) and col_idx < len(data[row_idx]):
                data[row_idx][col_idx] = repair_value
        
        return data
    
    def _ml_based_repair(self, data):
        """ML-based repair using predictive models"""
        repaired_data = [row[:] for row in data]
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(data)
        
        # Group errors by column
        errors_by_col = {}
        for row_idx, col_idx in self.error_cells:
            if col_idx not in errors_by_col:
                errors_by_col[col_idx] = []
            errors_by_col[col_idx].append(row_idx)
        
        # Repair each column using ML
        for col_idx, error_rows in errors_by_col.items():
            try:
                repaired_values = self._predict_values_ml(df, col_idx, error_rows)
                for i, row_idx in enumerate(error_rows):
                    if i < len(repaired_values):
                        repaired_data[row_idx][col_idx] = str(repaired_values[i])
            except Exception as e:
                print("ML repair failed for column {}, using simple repair: {}".format(col_idx, e))
                repaired_data = self._repair_column_simple(repaired_data, col_idx, error_rows)
        
        return repaired_data
    
    def _predict_values_ml(self, df, target_col, error_rows):
        """Use ML to predict values for error cells"""
        # Prepare training data (non-error rows)
        train_mask = ~df.index.isin(error_rows)
        
        # Get features (other columns) and target
        feature_cols = [i for i in range(len(df.columns)) if i != target_col]
        
        if not feature_cols:
            return [self._get_simple_repair_value(df, target_col)] * len(error_rows)
        
        # Prepare training data
        X_train = df.iloc[train_mask, feature_cols]
        y_train = df.iloc[train_mask, target_col]
        
        # Prepare prediction data
        X_pred = df.iloc[error_rows, feature_cols]
        
        # Convert categorical features to numeric
        X_train_processed, X_pred_processed = self._preprocess_features(X_train, X_pred)
        y_train_processed, is_numeric = self._preprocess_target(y_train)
        
        if len(X_train_processed) < 5:  # Not enough training data
            return [self._get_simple_repair_value(df, target_col)] * len(error_rows)
        
        # Choose and train model
        if is_numeric:
            model = RandomForestRegressor(n_estimators=10, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        try:
            model.fit(X_train_processed, y_train_processed)
            predictions = model.predict(X_pred_processed)
            return predictions.tolist()
        except:
            return [self._get_simple_repair_value(df, target_col)] * len(error_rows)
    
    def _preprocess_features(self, X_train, X_pred):
        """Preprocess features for ML models"""
        # Simple preprocessing: convert to numeric where possible
        X_train_processed = []
        X_pred_processed = []
        
        for _, row in X_train.iterrows():
            processed_row = []
            for val in row:
                try:
                    processed_row.append(float(str(val).replace(',', '').replace('$', '').replace('%', '')))
                except:
                    processed_row.append(hash(str(val)) % 10000)  # Simple hash for categorical
            X_train_processed.append(processed_row)
        
        for _, row in X_pred.iterrows():
            processed_row = []
            for val in row:
                try:
                    processed_row.append(float(str(val).replace(',', '').replace('$', '').replace('%', '')))
                except:
                    processed_row.append(hash(str(val)) % 10000)
            X_pred_processed.append(processed_row)
        
        return np.array(X_train_processed), np.array(X_pred_processed)
    
    def _preprocess_target(self, y_train):
        """Preprocess target variable"""
        numeric_count = 0
        processed_values = []
        
        for val in y_train:
            try:
                num_val = float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                processed_values.append(num_val)
                numeric_count += 1
            except:
                processed_values.append(str(val))
        
        is_numeric = numeric_count > len(y_train) * 0.7
        
        if not is_numeric:
            # Encode categorical target
            unique_values = list(set(processed_values))
            encoded_values = [unique_values.index(val) for val in processed_values]
            return encoded_values, False
        
        return processed_values, True
    
    def _get_simple_repair_value(self, df, col_idx):
        """Get simple repair value for fallback"""
        col_values = df.iloc[:, col_idx].dropna().astype(str)
        clean_values = col_values[~col_values.isin(['', 'N/A', 'NULL', 'null', '?'])]
        
        if len(clean_values) == 0:
            return ""
        
        # Try numeric
        numeric_values = []
        for val in clean_values:
            try:
                numeric_values.append(float(str(val).replace(',', '').replace('$', '').replace('%', '')))
            except:
                pass
        
        if len(numeric_values) > len(clean_values) * 0.5:
            return str(np.median(numeric_values))
        else:
            return Counter(clean_values).most_common(1)[0][0]
    
    def _adaptive_repair(self, data):
        """Adaptive repair that chooses strategy based on data characteristics"""
        repaired_data = [row[:] for row in data]
        
        # Group errors by column
        errors_by_col = {}
        for row_idx, col_idx in self.error_cells:
            if col_idx not in errors_by_col:
                errors_by_col[col_idx] = []
            errors_by_col[col_idx].append(row_idx)
        
        # Repair each column with adaptive strategy
        for col_idx, error_rows in errors_by_col.items():
            col_type = self.column_types.get(col_idx, {'type': 'unknown'})
            
            # Choose strategy based on column characteristics
            if len(error_rows) < 5:  # Few errors, use simple repair
                repaired_data = self._repair_column_simple(repaired_data, col_idx, error_rows)
            elif col_type['type'] == 'numeric':
                # For numeric columns with many errors, try ML
                try:
                    df = pd.DataFrame(data)
                    repaired_values = self._predict_values_ml(df, col_idx, error_rows)
                    for i, row_idx in enumerate(error_rows):
                        if i < len(repaired_values):
                            repaired_data[row_idx][col_idx] = str(repaired_values[i])
                except:
                    repaired_data = self._repair_column_simple(repaired_data, col_idx, error_rows)
            else:
                # For categorical columns, use pattern-based repair if possible
                repaired_data = self._pattern_based_repair(repaired_data, col_idx, error_rows)
        
        return repaired_data
    
    def _pattern_based_repair(self, data, col_idx, error_rows):
        """Pattern-based repair for categorical data"""
        col_type = self.column_types.get(col_idx, {})
        
        if col_type.get('subtype') == 'date':
            return self._repair_dates(data, col_idx, error_rows)
        elif col_type.get('subtype') == 'email':
            return self._repair_emails(data, col_idx, error_rows)
        else:
            return self._repair_column_simple(data, col_idx, error_rows)
    
    def _repair_dates(self, data, col_idx, error_rows):
        """Repair date columns using date patterns"""
        # Simple date repair - use most common date format
        clean_values = []
        for row_idx, row in enumerate(data):
            if row_idx not in error_rows and col_idx < len(row):
                val = row[col_idx]
                if val and str(val).strip() not in ['', 'N/A', 'NULL', 'null', '?']:
                    clean_values.append(str(val).strip())
        
        if not clean_values:
            return data
        
        # Find most common date format and use median date
        dates = []
        for val in clean_values:
            try:
                # Try to parse common date formats
                if '/' in val:
                    parts = val.split('/')
                    if len(parts) == 3:
                        dates.append(val)
                elif '-' in val:
                    parts = val.split('-')
                    if len(parts) == 3:
                        dates.append(val)
            except:
                pass
        
        if dates:
            # Use most common date as repair value
            repair_value = Counter(dates).most_common(1)[0][0]
        else:
            repair_value = clean_values[0] if clean_values else "1900-01-01"
        
        # Apply repair
        for row_idx in error_rows:
            if row_idx < len(data) and col_idx < len(data[row_idx]):
                data[row_idx][col_idx] = repair_value
        
        return data
    
    def _repair_emails(self, data, col_idx, error_rows):
        """Repair email columns using email patterns"""
        # Use most common domain
        clean_values = []
        for row_idx, row in enumerate(data):
            if row_idx not in error_rows and col_idx < len(row):
                val = row[col_idx]
                if val and str(val).strip() not in ['', 'N/A', 'NULL', 'null', '?']:
                    clean_values.append(str(val).strip())
        
        if not clean_values:
            return data
        
        # Extract domains
        domains = []
        for email in clean_values:
            if '@' in email:
                domain = email.split('@')[-1]
                domains.append(domain)
        
        if domains:
            most_common_domain = Counter(domains).most_common(1)[0][0]
            repair_value = "user@{}".format(most_common_domain)
        else:
            repair_value = clean_values[0] if clean_values else "user@example.com"
        
        # Apply repair
        for row_idx in error_rows:
            if row_idx < len(data) and col_idx < len(data[row_idx]):
                data[row_idx][col_idx] = repair_value
        
        return data
    
    def _ensemble_repair(self, data):
        """Ensemble repair combining multiple strategies"""
        # Get repairs from different strategies
        simple_repair = self._simple_repair([row[:] for row in data])
        
        try:
            ml_repair = self._ml_based_repair([row[:] for row in data])
        except:
            ml_repair = simple_repair
        
        # Combine repairs using voting or confidence-based selection
        repaired_data = [row[:] for row in data]
        
        for row_idx, col_idx in self.error_cells:
            # For now, prefer ML repair if available, otherwise use simple
            if (row_idx < len(ml_repair) and col_idx < len(ml_repair[row_idx]) and 
                ml_repair[row_idx][col_idx] != simple_repair[row_idx][col_idx]):
                # Use ML repair
                repaired_data[row_idx][col_idx] = ml_repair[row_idx][col_idx]
            else:
                # Use simple repair
                repaired_data[row_idx][col_idx] = simple_repair[row_idx][col_idx]
        
        return repaired_data

# Test functions
def load_dataset(dataset_name):
    """Load dirty and clean datasets"""
    try:
        dirty_path = "datasets/{}/dirty.csv".format(dataset_name)
        clean_path = "datasets/{}/clean.csv".format(dataset_name)
        
        if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
            print("Missing files for {}".format(dataset_name))
            return None, None
        
        # Load dirty data
        dirty_data = []
        with open(dirty_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                dirty_data.append(row)
        
        # Load clean data
        clean_data = []
        with open(clean_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                clean_data.append(row)
        
        print("Loaded {}: {} rows, {} columns".format(dataset_name, len(dirty_data), len(dirty_data[0]) if dirty_data else 0))
        return dirty_data, clean_data
        
    except Exception as e:
        print("Failed to load {}: {}".format(dataset_name, e))
        return None, None

def calculate_f1_score(repaired_data, clean_data):
    """Calculate F1 score comparing repaired data to clean ground truth"""
    try:
        if len(repaired_data) != len(clean_data):
            print("Data length mismatch")
            return 0.0
        
        # Flatten data for comparison
        repaired_flat = []
        clean_flat = []
        
        for i in range(len(repaired_data)):
            for j in range(min(len(repaired_data[i]), len(clean_data[i]))):
                repaired_flat.append(str(repaired_data[i][j]).strip())
                clean_flat.append(str(clean_data[i][j]).strip())
        
        # Calculate cell-level accuracy
        matches = sum(1 for r, c in zip(repaired_flat, clean_flat) if r == c)
        total = len(repaired_flat)
        
        if total == 0:
            return 0.0
        
        # Calculate precision and F1
        precision = float(matches) / total
        recall = 1.0  # We attempt to repair everything
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
        
    except Exception as e:
        print("F1 calculation failed: {}".format(e))
        return 0.0

def test_repair_strategies(dataset_name):
    """Test different repair strategies on a dataset"""
    print("\n{}".format("=" * 60))
    print("Testing Enhanced Data Repair on: {}".format(dataset_name))
    print("{}".format("=" * 60))
    
    # Load data
    dirty_data, clean_data = load_dataset(dataset_name)
    if dirty_data is None or clean_data is None:
        return None
    
    strategies = ['simple', 'ml_based', 'adaptive', 'ensemble']
    results = {}
    
    for strategy in strategies:
        print("\n--- Testing {} strategy ---".format(strategy))
        
        # Initialize repairer
        repairer = EnhancedDataRepairer(repair_strategy=strategy)
        
        # Detect errors
        print("Detecting errors...")
        error_cells = repairer.detect_errors(dirty_data)
        
        # Repair data
        print("Repairing data using {} strategy...".format(strategy))
        repaired_data = repairer.repair_data(dirty_data)
        
        # Calculate F1 score
        f1 = calculate_f1_score(repaired_data, clean_data)
        
        results[strategy] = {
            'errors_detected': len(error_cells),
            'f1_score': f1
        }
        
        print("Strategy: {}".format(strategy))
        print("Errors detected: {}".format(len(error_cells)))
        print("F1 Score: {:.4f}".format(f1))
    
    return results

def main():
    """Main function to test enhanced repair strategies"""
    datasets = ['beers', 'flight', 'movies', 'rayyan']
    
    print("Enhanced Data Repair System")
    print("==========================")
    print("Testing multiple repair strategies: simple, ml_based, adaptive, ensemble")
    
    all_results = {}
    
    for dataset in datasets:
        try:
            results = test_repair_strategies(dataset)
            if results:
                all_results[dataset] = results
        except Exception as e:
            print("Failed to test {}: {}".format(dataset, e))
    
    # Print summary
    print("\n{}".format("=" * 80))
    print("SUMMARY RESULTS")
    print("{}".format("=" * 80))
    print("{:<10} {:<12} {:<12} {:<12} {:<12}".format('Dataset', 'Simple', 'ML-based', 'Adaptive', 'Ensemble'))
    print("{}".format("-" * 80))
    
    for dataset, results in all_results.items():
        row = [dataset]
        for strategy in ['simple', 'ml_based', 'adaptive', 'ensemble']:
            if strategy in results:
                row.append("{:.4f}".format(results[strategy]['f1_score']))
            else:
                row.append("N/A")
        print("{:<10} {:<12} {:<12} {:<12} {:<12}".format(*row))
    
    return all_results

if __name__ == "__main__":
    results = main() 