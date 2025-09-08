#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import json
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gidcl import GIDCL, ErrorCell, ErrorType
from EMCL.utils import EvaluationMetrics, Logger, Profiler


class DatasetTester:
    """Test GEIL on multiple datasets using Nebius API."""
    
    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # DeepSeek API configuration
        self.config = {
            "llm_provider": {
                "type": "deepseek",
                "api_keys": {
                    "deepseek": "sk-1015b59d6db94c42b1c5d34276b109d1"
                },
                "models": {
                    "deepseek": "deepseek-chat"
                },
                "rate_limit": 60,  # DeepSeek rate limit
                "use_rate_limiting": True
            },
            "correction": {
                "implicit": {
                    "use_api": True,
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "few_shot_examples": 3
                },
                "explicit": {
                    "use_api": True,
                    "temperature": 0.3,
                    "max_tokens": 800,
                    "max_iterations": 2,
                    "confidence_threshold": 0.7
                }
            }
        }
        
        self.logger = Logger()
        self.profiler = Profiler()
        
        # Dataset paths
        self.datasets_dir = Path("../../datasets")
        self.dataset_names = ["beers", "movies", "flight", "shuttle", "rayyan", "walmart"]
        
        # Results storage
        self.all_results = {}
        
    def detect_errors(self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame) -> List[ErrorCell]:
        """Detect errors by comparing clean and dirty datasets."""
        error_cells = []
        
        if clean_df.shape != dirty_df.shape:
            self.logger.warning(f"Shape mismatch: clean {clean_df.shape} vs dirty {dirty_df.shape}")
            return error_cells
        
        for row_idx in range(len(clean_df)):
            for col_idx in range(len(clean_df.columns)):
                clean_val = clean_df.iloc[row_idx, col_idx]
                dirty_val = dirty_df.iloc[row_idx, col_idx]
                
                # Compare values (handle NaN specially)
                if pd.isna(clean_val) and pd.isna(dirty_val):
                    continue
                elif str(clean_val).strip() != str(dirty_val).strip():
                    # Determine error type based on the nature of the error
                    error_type = self._classify_error_type(clean_val, dirty_val, clean_df.columns[col_idx])
                    
                    error_cell = ErrorCell(
                        row_idx=row_idx,
                        col_idx=col_idx,
                        value=dirty_val,
                        error_type=error_type,
                        confidence=0.9,
                        context={"clean_value": clean_val}
                    )
                    error_cells.append(error_cell)
        
        return error_cells
    
    def _classify_error_type(self, clean_val: Any, dirty_val: Any, column_name: str) -> ErrorType:
        """Classify the type of error based on clean and dirty values."""
        clean_str = str(clean_val).strip() if not pd.isna(clean_val) else ""
        dirty_str = str(dirty_val).strip() if not pd.isna(dirty_val) else ""
        
        # Pattern errors (format changes)
        if any(char in dirty_str for char in ['%', 'oz', 'N/A', '.']):
            return ErrorType.PATTERN
        
        # Case/formatting errors
        if clean_str.lower() == dirty_str.lower():
            return ErrorType.FORMATTING
        
        # Missing data
        if dirty_str in ['', 'null', 'NULL', 'None']:
            return ErrorType.CONTEXT_DEPENDENT
        
        # Default to semantic error
        return ErrorType.SEMANTIC
    
    def test_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Test GEIL on a specific dataset."""
        self.logger.info(f"Testing dataset: {dataset_name}")
        
        dataset_path = self.datasets_dir / dataset_name
        clean_path = dataset_path / "clean.csv"
        dirty_path = dataset_path / "dirty.csv"
        
        if not clean_path.exists() or not dirty_path.exists():
            self.logger.error(f"Dataset files not found for {dataset_name}")
            return {"error": "Dataset files not found"}
        
        try:
            # Load datasets
            self.logger.info(f"Loading {dataset_name} dataset...")
            with self.profiler.time_operation(f"{dataset_name}_load"):
                clean_df = pd.read_csv(clean_path)
                dirty_df = pd.read_csv(dirty_path)
            
            self.logger.info(f"Dataset shape: {dirty_df.shape}")
            
            # Detect errors
            self.logger.info("Detecting errors...")
            with self.profiler.time_operation(f"{dataset_name}_error_detection"):
                error_cells = self.detect_errors(clean_df, dirty_df)
            
            self.logger.info(f"Found {len(error_cells)} error cells")
            
            if len(error_cells) == 0:
                self.logger.warning("No errors detected - datasets might be identical")
                return {"error": "No errors detected"}
            
            # Limit errors for large datasets to avoid excessive API calls
            max_errors = 100  # Reasonable limit for testing
            if len(error_cells) > max_errors:
                self.logger.info(f"Limiting to {max_errors} errors for testing")
                error_cells = error_cells[:max_errors]
            
            # Initialize GIDCL with DeepSeek API
            self.logger.info("Initializing GIDCL with DeepSeek API...")
            with self.profiler.time_operation(f"{dataset_name}_init"):
                gidcl = GIDCL(device="cpu", config=self.config)
            
            # Create schema info
            schema_info = {
                "table_name": dataset_name,
                "columns": {col: {"type": str(dirty_df[col].dtype)} for col in dirty_df.columns},
                "num_rows": len(dirty_df),
                "num_columns": len(dirty_df.columns)
            }
            
            # Prepare auxiliary context with examples
            auxiliary_context = {
                "examples": self._get_dataset_specific_examples(dataset_name),
                "external_data": []
            }
            
            # Repair table
            self.logger.info("Repairing table using DeepSeek API...")
            with self.profiler.time_operation(f"{dataset_name}_repair"):
                repaired_df, correction_results = gidcl.repair_table(
                    dirty_df.copy(),
                    error_cells,
                    schema_info,
                    auxiliary_context
                )
            
            # Save repaired dataset
            output_path = self.results_dir / f"{dataset_name}_repaired.csv"
            repaired_df.to_csv(output_path, index=False)
            self.logger.info(f"Repaired dataset saved to: {output_path}")
            
            # Calculate metrics
            self.logger.info("Calculating evaluation metrics...")
            error_positions = [(cell.row_idx, cell.col_idx) for cell in error_cells]
            
            metrics = EvaluationMetrics.calculate_repair_accuracy(
                clean_df, dirty_df, repaired_df, error_positions
            )
            
            # Calculate F1 score specifically
            f1_score = metrics.get('f1', 0.0)
            
            # Gather detailed results
            results = {
                "dataset": dataset_name,
                "shape": dirty_df.shape,
                "total_errors_found": len(self.detect_errors(clean_df, dirty_df)),
                "errors_tested": len(error_cells),
                "corrections_made": len(correction_results),
                "f1_score": f1_score,
                "accuracy": metrics.get('accuracy', 0.0),
                "precision": metrics.get('precision', 0.0),
                "recall": metrics.get('recall', 0.0),
                "timing": {
                    "load_time": self.profiler.get_operation_time(f"{dataset_name}_load"),
                    "error_detection_time": self.profiler.get_operation_time(f"{dataset_name}_error_detection"),
                    "init_time": self.profiler.get_operation_time(f"{dataset_name}_init"),
                    "repair_time": self.profiler.get_operation_time(f"{dataset_name}_repair"),
                },
                "method_distribution": self._analyze_correction_methods(correction_results),
                "average_confidence": np.mean([r.confidence for r in correction_results]) if correction_results else 0.0,
                "output_path": str(output_path)
            }
            
            # Print immediate results
            print(f"\n{'='*60}")
            print(f"RESULTS for {dataset_name.upper()}")
            print(f"{'='*60}")
            print(f"Dataset shape: {results['shape']}")
            print(f"Errors found: {results['total_errors_found']}")
            print(f"Errors tested: {results['errors_tested']}")
            print(f"Corrections made: {results['corrections_made']}")
            print(f"F1 Score: {results['f1_score']:.4f}")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"Repair time: {results['timing']['repair_time']:.2f}s")
            print(f"Average confidence: {results['average_confidence']:.3f}")
            print(f"Method distribution: {results['method_distribution']}")
            print(f"Repaired file: {results['output_path']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing {dataset_name}: {str(e)}")
            return {"error": str(e), "dataset": dataset_name}
    
    def _get_dataset_specific_examples(self, dataset_name: str) -> List[Tuple[str, str]]:
        """Get dataset-specific examples for few-shot learning."""
        examples = {
            "beers": [
                ("12.0 oz", "12"),
                ("0.05%", "0.05"),
                ("N/A", "")
            ],
            "movies": [
                ("1990s", "1990"),
                ("USA", "United States"),
                ("sci-fi", "Science Fiction")
            ],
            "flight": [
                ("1200pm", "12:00"),
                ("JFK", "John F. Kennedy International Airport"),
                ("delayed", "Delayed")
            ],
            "shuttle": [
                ("OK", "Nominal"),
                ("FAIL", "Failure"),
                ("AUTO", "Automatic")
            ],
            "rayyan": [
                ("YES", "Yes"),
                ("NO", "No"),
                ("UNKNOWN", "Unknown")
            ],
            "walmart": [
                ("$19.99", "19.99"),
                ("IN STOCK", "In Stock"),
                ("N/A", "")
            ]
        }
        
        return examples.get(dataset_name, [])
    
    def _analyze_correction_methods(self, correction_results: List) -> Dict[str, int]:
        """Analyze the distribution of correction methods used."""
        method_counts = {}
        for result in correction_results:
            method = result.method.value
            method_counts[method] = method_counts.get(method, 0) + 1
        return method_counts
    
    def test_all_datasets(self) -> Dict[str, Any]:
        """Test GEIL on all datasets."""
        print("="*80)
        print("GEIL DATASET TESTING WITH NEBIUS API")
        print("="*80)
        print(f"Testing datasets: {', '.join(self.dataset_names)}")
        print(f"Results will be saved in: {self.results_dir}")
        print("="*80)
        
        all_results = {}
        summary_results = []
        
        for dataset_name in self.dataset_names:
            print(f"\nStarting test for {dataset_name}...")
            
            try:
                result = self.test_dataset(dataset_name)
                all_results[dataset_name] = result
                
                if "error" not in result:
                    summary_results.append({
                        "dataset": dataset_name,
                        "f1_score": result["f1_score"],
                        "accuracy": result["accuracy"],
                        "repair_time": result["timing"]["repair_time"]
                    })
                
                # Small delay between datasets to respect rate limits
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Failed to test {dataset_name}: {str(e)}")
                all_results[dataset_name] = {"error": str(e)}
        
        # Save detailed results
        results_file = self.results_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Print summary
        self._print_summary(summary_results)
        
        # Save summary
        summary_file = self.results_dir / "summary_results.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        return all_results
    
    def _print_summary(self, summary_results: List[Dict]):
        """Print a summary of all test results."""
        print("\n" + "="*80)
        print("SUMMARY RESULTS")
        print("="*80)
        
        if not summary_results:
            print("No successful tests completed.")
            return
        
        print(f"{'Dataset':<15} {'F1 Score':<10} {'Accuracy':<10} {'Time (s)':<10}")
        print("-" * 50)
        
        for result in summary_results:
            print(f"{result['dataset']:<15} {result['f1_score']:<10.4f} "
                  f"{result['accuracy']:<10.4f} {result['repair_time']:<10.2f}")
        
        # Overall statistics
        f1_scores = [r['f1_score'] for r in summary_results]
        accuracies = [r['accuracy'] for r in summary_results]
        
        print("-" * 50)
        print(f"{'AVERAGE':<15} {np.mean(f1_scores):<10.4f} "
              f"{np.mean(accuracies):<10.4f} {np.mean([r['repair_time'] for r in summary_results]):<10.2f}")
        
        print(f"\nBest F1 Score: {max(f1_scores):.4f} ({summary_results[np.argmax(f1_scores)]['dataset']})")
        print(f"Best Accuracy: {max(accuracies):.4f} ({summary_results[np.argmax(accuracies)]['dataset']})")


def main():
    parser = argparse.ArgumentParser(description='Test GEIL on datasets using Nebius API')
    parser.add_argument('--dataset', type=str, help='Test specific dataset')
    parser.add_argument('--all', action='store_true', help='Test all datasets')
    parser.add_argument('--results-dir', type=str, default='test_results', help='Results directory')
    
    args = parser.parse_args()
    
    tester = DatasetTester(args.results_dir)
    
    if args.dataset:
        if args.dataset in tester.dataset_names:
            tester.test_dataset(args.dataset)
        else:
            print(f"Dataset '{args.dataset}' not found. Available datasets: {tester.dataset_names}")
    elif args.all:
        tester.test_all_datasets()
    else:
        # Default: start with beers dataset as requested
        print("Testing beers dataset (use --all to test all datasets)")
        tester.test_dataset("beers")


if __name__ == "__main__":
    main() 