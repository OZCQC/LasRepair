#!/usr/bin/env python3

import pandas as pd
import numpy as np
from gidcl import GIDCL, ErrorCell, ErrorType
from utils import ErrorGenerator, EvaluationMetrics, Logger, Profiler
import argparse
import json


def create_sample_data():
    """Create sample data for demonstration."""
    data = {
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
        'Age': [25, 30, 35, 28, 42],
        'Email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
        'Phone': ['123-456-7890', '234-567-8901', '345-678-9012', '456-789-0123', '567-890-1234'],
        'Salary': [50000, 60000, 70000, 55000, 80000],
        'Department': ['Engineering', 'Marketing', 'Sales', 'HR', 'Engineering']
    }
    
    return pd.DataFrame(data)


def run_demo():
    """Run a complete demonstration of the GIDCL framework."""
    
    logger = Logger()
    profiler = Profiler()
    
    logger.info("=== GIDCL Framework Demo ===")
    
    # Step 1: Create sample data
    logger.info("Step 1: Creating sample data")
    clean_table = create_sample_data()
    logger.info(f"Created table with shape: {clean_table.shape}")
    print("\nClean table:")
    print(clean_table.to_string())
    
    # Step 2: Introduce synthetic errors
    logger.info("Step 2: Introducing synthetic errors")
    error_generator = ErrorGenerator(error_rate=0.2)
    
    with profiler.time_operation("error_generation"):
        dirty_table, error_log = error_generator.introduce_errors(clean_table)
    
    logger.info(f"Introduced {len(error_log)} errors")
    print(f"\nDirty table:")
    print(dirty_table.to_string())
    
    print("\nError log:")
    for i, error in enumerate(error_log):
        print(f"  {i+1}. Row {error['row_idx']}, Col {error['col_idx']}: "
              f"'{error['original_value']}' -> '{error['corrupted_value']}' "
              f"(Type: {error['error_type']})")
    
    # Step 3: Initialize GIDCL
    logger.info("Step 3: Initializing GIDCL framework")
    
    with profiler.time_operation("gidcl_init"):
        gidcl = GIDCL(llm_model_name="microsoft/DialoGPT-medium", device="cpu")
    
    # Step 4: Create error cells from error log
    logger.info("Step 4: Creating error cells for repair")
    error_cells = []
    
    for error in error_log:
        error_type_mapping = {
            'typo': ErrorType.FORMATTING,
            'case': ErrorType.FORMATTING,
            'format': ErrorType.PATTERN,
            'missing': ErrorType.CONTEXT_DEPENDENT,
            'outlier': ErrorType.SEMANTIC,
            'swap': ErrorType.FORMATTING
        }
        
        error_cell = ErrorCell(
            row_idx=error['row_idx'],
            col_idx=error['col_idx'],
            value=error['corrupted_value'],
            error_type=error_type_mapping.get(error['error_type'], ErrorType.FORMATTING),
            confidence=0.8,
            context={}
        )
        
        error_cells.append(error_cell)
    
    # Step 5: Create schema information
    logger.info("Step 5: Creating schema information")
    schema_info = {
        "columns": {},
        "table_name": "demo_table"
    }
    
    for col in clean_table.columns:
        schema_info["columns"][col] = {
            "type": str(clean_table[col].dtype),
            "sample_values": clean_table[col].head(3).tolist()
        }
    
    # Step 6: Repair the table
    logger.info("Step 6: Repairing the table")
    
    with profiler.time_operation("table_repair"):
        repaired_table, correction_results = gidcl.repair_table(
            dirty_table,
            error_cells,
            schema_info,
            auxiliary_context={}
        )
    
    print(f"\nRepaired table:")
    print(repaired_table.to_string())
    
    # Step 7: Evaluate results
    logger.info("Step 7: Evaluating repair results")
    
    error_positions = [(error['row_idx'], error['col_idx']) for error in error_log]
    
    with profiler.time_operation("evaluation"):
        metrics = EvaluationMetrics.calculate_repair_accuracy(
            clean_table, dirty_table, repaired_table, error_positions
        )
    
    logger.info("=== REPAIR RESULTS ===")
    print(f"\nRepair Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1']:.3f}")
    print(f"  Total Errors: {metrics['total_errors']}")
    print(f"  Attempted Repairs: {metrics['attempted_repairs']}")
    print(f"  Correct Repairs: {metrics['correct_repairs']}")
    
    # Step 8: Show correction details
    print(f"\nCorrection Details:")
    for i, result in enumerate(correction_results):
        print(f"  {i+1}. '{result.original_value}' -> '{result.corrected_value}' "
              f"(Method: {result.method.value}, Confidence: {result.confidence:.3f})")
        if result.rule:
            print(f"      Rule: {result.rule}")
    
    # Step 9: Method usage summary
    method_counts = {}
    for result in correction_results:
        method = result.method.value
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"\nMethod Usage:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} corrections")
    
    logger.info("Demo completed successfully!")


def run_benchmarks():
    """Run benchmarks on different table sizes and error rates."""
    
    logger = Logger()
    profiler = Profiler()
    
    logger.info("=== GIDCL Benchmarks ===")
    
    # Test different configurations
    test_configs = [
        {"rows": 50, "cols": 5, "error_rate": 0.1},
        {"rows": 100, "cols": 8, "error_rate": 0.15},
        {"rows": 200, "cols": 10, "error_rate": 0.2},
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"Testing configuration: {config}")
        
        # Generate synthetic table
        data = {}
        for i in range(config["cols"]):
            if i % 3 == 0:
                data[f"text_col_{i}"] = [f"value_{j}_{i}" for j in range(config["rows"])]
            elif i % 3 == 1:
                data[f"num_col_{i}"] = np.random.randint(1, 100, config["rows"])
            else:
                data[f"cat_col_{i}"] = np.random.choice(['A', 'B', 'C'], config["rows"])
        
        clean_table = pd.DataFrame(data)
        
        # Introduce errors
        error_generator = ErrorGenerator(error_rate=config["error_rate"])
        
        with profiler.time_operation("error_generation"):
            dirty_table, error_log = error_generator.introduce_errors(clean_table)
        
        # Initialize GIDCL
        gidcl = GIDCL(llm_model_name="microsoft/DialoGPT-medium", device="cpu")
        
        # Create error cells
        error_cells = []
        for error in error_log:
            error_cell = ErrorCell(
                row_idx=error['row_idx'],
                col_idx=error['col_idx'],
                value=error['corrupted_value'],
                error_type=ErrorType.FORMATTING,
                confidence=0.8,
                context={}
            )
            error_cells.append(error_cell)
        
        # Repair table
        with profiler.time_operation("repair"):
            repaired_table, correction_results = gidcl.repair_table(
                dirty_table, error_cells, {}, {}
            )
        
        # Evaluate
        error_positions = [(error['row_idx'], error['col_idx']) for error in error_log]
        metrics = EvaluationMetrics.calculate_repair_accuracy(
            clean_table, dirty_table, repaired_table, error_positions
        )
        
        result = {
            "config": config,
            "metrics": metrics,
            "num_corrections": len(correction_results)
        }
        
        results.append(result)
        
        logger.info(f"Results: Accuracy={metrics['accuracy']:.3f}, "
                   f"F1={metrics['f1']:.3f}, Corrections={len(correction_results)}")
    
    # Summary
    logger.info("=== BENCHMARK SUMMARY ===")
    print("\nBenchmark Results:")
    print("Config (Rows x Cols, Error Rate) | Accuracy | F1 Score | Corrections")
    print("-" * 70)
    
    for result in results:
        config = result["config"]
        metrics = result["metrics"]
        print(f"{config['rows']}x{config['cols']}, {config['error_rate']:.1f}% "
              f"| {metrics['accuracy']:.3f} | {metrics['f1']:.3f} | {result['num_corrections']}")


def main():
    parser = argparse.ArgumentParser(description='GIDCL Demo and Benchmarks')
    parser.add_argument('--demo', action='store_true', help='Run the demo')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--all', action='store_true', help='Run both demo and benchmarks')
    
    args = parser.parse_args()
    
    if args.demo or args.all:
        run_demo()
    
    if args.benchmark or args.all:
        run_benchmarks()
    
    if not any([args.demo, args.benchmark, args.all]):
        print("Please specify --demo, --benchmark, or --all")
        run_demo()  # Default to demo


if __name__ == "__main__":
    main()