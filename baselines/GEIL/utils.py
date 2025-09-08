import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import time


class ErrorGenerator:
    """Generate synthetic errors for testing and training."""
    
    def __init__(self, error_rate: float = 0.1):
        self.error_rate = error_rate
        self.error_types = ['typo', 'case', 'format', 'missing', 'outlier', 'swap']
    
    def introduce_errors(self, clean_table: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Introduce synthetic errors into a clean table."""
        dirty_table = clean_table.copy()
        error_log = []
        
        total_cells = len(clean_table) * len(clean_table.columns)
        num_errors = int(total_cells * self.error_rate)
        
        for _ in range(num_errors):
            row_idx = np.random.randint(0, len(clean_table))
            col_idx = np.random.randint(0, len(clean_table.columns))
            
            original_value = clean_table.iloc[row_idx, col_idx]
            error_type = np.random.choice(self.error_types)
            
            corrupted_value = self._apply_error(original_value, error_type)
            
            dirty_table.iloc[row_idx, col_idx] = corrupted_value
            
            error_log.append({
                'row_idx': row_idx,
                'col_idx': col_idx,
                'original_value': original_value,
                'corrupted_value': corrupted_value,
                'error_type': error_type
            })
        
        return dirty_table, error_log
    
    def _apply_error(self, value: Any, error_type: str) -> Any:
        """Apply a specific type of error to a value."""
        if pd.isna(value):
            return value
        
        if error_type == 'typo' and isinstance(value, str):
            return self._introduce_typo(value)
        elif error_type == 'case' and isinstance(value, str):
            return self._change_case(value)
        elif error_type == 'format' and isinstance(value, str):
            return self._format_error(value)
        elif error_type == 'missing':
            return np.nan
        elif error_type == 'outlier' and isinstance(value, (int, float)):
            return self._introduce_outlier(value)
        elif error_type == 'swap' and isinstance(value, str):
            return self._character_swap(value)
        
        return value
    
    def _introduce_typo(self, text: str) -> str:
        if len(text) <= 1:
            return text
        
        idx = np.random.randint(0, len(text))
        chars = list(text)
        
        # Replace with random character
        chars[idx] = chr(np.random.randint(97, 123))
        
        return ''.join(chars)
    
    def _change_case(self, text: str) -> str:
        return text.swapcase()
    
    def _format_error(self, text: str) -> str:
        # Add extra spaces, underscores, or remove spaces
        operations = [
            lambda x: x.replace(' ', '_'),
            lambda x: x.replace(' ', ''),
            lambda x: x + ' ',
            lambda x: ' ' + x
        ]
        
        operation = np.random.choice(operations)
        return operation(text)
    
    def _introduce_outlier(self, value: float) -> float:
        # Make value 5-10 times larger or smaller
        multiplier = np.random.choice([0.1, 0.2, 5, 10])
        return value * multiplier
    
    def _character_swap(self, text: str) -> str:
        if len(text) <= 1:
            return text
        
        chars = list(text)
        idx1 = np.random.randint(0, len(text))
        idx2 = np.random.randint(0, len(text))
        
        chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
        
        return ''.join(chars)


class EvaluationMetrics:
    """Evaluate data repair performance."""
    
    @staticmethod
    def calculate_repair_accuracy(original_table: pd.DataFrame,
                                dirty_table: pd.DataFrame,
                                repaired_table: pd.DataFrame,
                                error_positions: List[Tuple[int, int]]) -> Dict[str, float]:
        """Calculate various repair accuracy metrics."""
        
        total_errors = len(error_positions)
        if total_errors == 0:
            return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        correct_repairs = 0
        attempted_repairs = 0
        
        for row_idx, col_idx in error_positions:
            original_val = original_table.iloc[row_idx, col_idx]
            dirty_val = dirty_table.iloc[row_idx, col_idx]
            repaired_val = repaired_table.iloc[row_idx, col_idx]
            
            if repaired_val != dirty_val:  # Repair was attempted
                attempted_repairs += 1
                
                if EvaluationMetrics._values_equal(original_val, repaired_val):
                    correct_repairs += 1
        
        accuracy = correct_repairs / total_errors if total_errors > 0 else 0
        precision = correct_repairs / attempted_repairs if attempted_repairs > 0 else 0
        recall = correct_repairs / total_errors if total_errors > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_errors": total_errors,
            "attempted_repairs": attempted_repairs,
            "correct_repairs": correct_repairs
        }
    
    @staticmethod
    def _values_equal(val1: Any, val2: Any) -> bool:
        """Check if two values are equal, handling NaN cases."""
        if pd.isna(val1) and pd.isna(val2):
            return True
        if pd.isna(val1) or pd.isna(val2):
            return False
        return val1 == val2
    
    @staticmethod
    def calculate_column_wise_metrics(original_table: pd.DataFrame,
                                    dirty_table: pd.DataFrame,
                                    repaired_table: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each column separately."""
        
        column_metrics = {}
        
        for col in original_table.columns:
            original_col = original_table[col]
            dirty_col = dirty_table[col]
            repaired_col = repaired_table[col]
            
            # Find positions where values differ between original and dirty
            error_mask = ~(original_col.equals(dirty_col))
            error_positions = [(i, original_table.columns.get_loc(col)) 
                             for i in error_mask[error_mask].index]
            
            metrics = EvaluationMetrics.calculate_repair_accuracy(
                original_table, dirty_table, repaired_table, error_positions
            )
            
            column_metrics[col] = metrics
        
        return column_metrics


class DataLoader:
    """Utilities for loading and preprocessing data."""
    
    @staticmethod
    def load_csv_with_preprocessing(file_path: str, 
                                  encoding: str = 'utf-8',
                                  delimiter: str = ',') -> pd.DataFrame:
        """Load CSV file with robust preprocessing."""
        try:
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
        except UnicodeDecodeError:
            # Try different encodings
            for enc in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    df = pd.read_csv(file_path, encoding=enc, delimiter=delimiter)
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not decode file {file_path}")
        
        # Basic preprocessing
        df = df.dropna(how='all')  # Remove empty rows
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
        
        return df
    
    @staticmethod
    def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """Infer appropriate data types for columns."""
        column_types = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    column_types[col] = 'numeric'
                except:
                    # Check if it's a date
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        column_types[col] = 'datetime'
                    except:
                        column_types[col] = 'categorical'
            else:
                column_types[col] = 'numeric'
        
        return column_types


class ConfigManager:
    """Manage configuration settings."""
    
    DEFAULT_CONFIG = {
        "error_detection": {
            "outlier_threshold": 1.5,
            "pattern_rules": [],
            "custom_validators": []
        },
        "correction": {
            "implicit": {
                "model_name": "microsoft/DialoGPT-medium",
                "training_epochs": 3,
                "batch_size": 8,
                "learning_rate": 5e-5
            },
            "explicit": {
                "max_iterations": 3,
                "confidence_threshold": 0.7
            }
        },
        "graph_refinement": {
            "fd_threshold": 0.95,
            "id_threshold": 0.8,
            "gnn_hidden_dim": 64,
            "gnn_epochs": 100
        },
        "selection": {
            "method": "heuristic",  # or "learned"
            "adaptation_threshold": 0.1
        }
    }
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(config_path):
            return ConfigManager.DEFAULT_CONFIG.copy()
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Merge with default config
        merged_config = ConfigManager.DEFAULT_CONFIG.copy()
        ConfigManager._deep_update(merged_config, config)
        
        return merged_config
    
    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                ConfigManager._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to file."""
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


class Logger:
    """Simple logging utility."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.start_time = time.time()
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time
        
        log_entry = f"[{timestamp}] [{level}] [{elapsed:.2f}s] {message}"
        
        print(log_entry)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def warning(self, message: str):
        self.log(message, "WARNING")
    
    def error(self, message: str):
        self.log(message, "ERROR")


class Profiler:
    """Simple profiling utility."""
    
    def __init__(self):
        self.timings = {}
        self.completed_timings = {}
    
    def start(self, name: str):
        """Start timing an operation."""
        self.timings[name] = time.time()
    
    def end(self, name: str) -> float:
        """End timing an operation and return duration."""
        if name not in self.timings:
            return 0.0
        
        duration = time.time() - self.timings[name]
        del self.timings[name]
        
        # Store completed timing
        self.completed_timings[name] = duration
        
        return duration
    
    def get_operation_time(self, name: str) -> float:
        """Get the duration of a completed operation."""
        return self.completed_timings.get(name, 0.0)
    
    def get_all_times(self) -> dict:
        """Get all completed operation timings."""
        return self.completed_timings.copy()
    
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        return TimingContext(self, name)


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, profiler: Profiler, name: str):
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        self.profiler.start(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.profiler.end(self.name)
        print(f"Operation '{self.name}' took {duration:.2f} seconds")


# Import os at the top level for ConfigManager
import os