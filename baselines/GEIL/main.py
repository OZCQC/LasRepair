#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from gidcl import GIDCL, ErrorCell, ErrorType, CorrectionMethod


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def detect_errors(table: pd.DataFrame, 
                 error_detection_config: Dict[str, Any]) -> List[ErrorCell]:
    """
    Detect errors in the table using various heuristics.
    This is a simplified error detection - in practice, you'd use more sophisticated methods.
    """
    errors = []
    
    for col_idx, column in enumerate(table.columns):
        col_data = table[column]
        
        # Detect null values
        null_indices = col_data.isnull()
        for row_idx in null_indices[null_indices].index:
            errors.append(ErrorCell(
                row_idx=row_idx,
                col_idx=col_idx,
                value=col_data.iloc[row_idx],
                error_type=ErrorType.CONTEXT_DEPENDENT,
                confidence=0.9,
                context={}
            ))
        
        # Detect outliers in numeric columns
        if col_data.dtype in ['int64', 'float64']:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            for row_idx in outlier_mask[outlier_mask].index:
                errors.append(ErrorCell(
                    row_idx=row_idx,
                    col_idx=col_idx,
                    value=col_data.iloc[row_idx],
                    error_type=ErrorType.SEMANTIC,
                    confidence=0.7,
                    context={"outlier_bounds": [lower_bound, upper_bound]}
                ))
        
        # Detect formatting issues in string columns
        if col_data.dtype == 'object':
            for row_idx, value in enumerate(col_data):
                if pd.notna(value) and isinstance(value, str):
                    # Check for common formatting issues
                    if value != value.strip():  # Leading/trailing whitespace
                        errors.append(ErrorCell(
                            row_idx=row_idx,
                            col_idx=col_idx,
                            value=value,
                            error_type=ErrorType.FORMATTING,
                            confidence=0.95,
                            context={"issue": "whitespace"}
                        ))
                    elif any(char in value for char in ['_', '-']) and ' ' in value:
                        errors.append(ErrorCell(
                            row_idx=row_idx,
                            col_idx=col_idx,
                            value=value,
                            error_type=ErrorType.PATTERN,
                            confidence=0.8,
                            context={"issue": "mixed_separators"}
                        ))
    
    return errors


def create_schema_info(table: pd.DataFrame) -> Dict[str, Any]:
    """Create schema information for the table."""
    schema = {
        "columns": {},
        "table_name": "input_table",
        "num_rows": len(table),
        "num_columns": len(table.columns)
    }
    
    for col in table.columns:
        col_data = table[col]
        schema["columns"][col] = {
            "type": str(col_data.dtype),
            "nullable": col_data.isnull().any(),
            "unique_values": col_data.nunique(),
            "sample_values": col_data.dropna().head(5).tolist()
        }
    
    return schema


def main():
    parser = argparse.ArgumentParser(description='GIDCL Data Repair Framework')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--model', default='microsoft/DialoGPT-medium', help='LLM model name')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--train', action='store_true', help='Train the implicit correction model')
    parser.add_argument('--training-data', help='Training data CSV file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
    
    # Load input data
    print(f"Loading data from {args.input}...")
    input_table = pd.read_csv(args.input)
    
    if args.verbose:
        print(f"Table shape: {input_table.shape}")
        print(f"Columns: {list(input_table.columns)}")
    
    # Initialize GIDCL
    print("Initializing GIDCL framework...")
    gidcl = GIDCL(llm_model_name=args.model, device=args.device)
    
    # Train implicit model if requested
    if args.train and args.training_data:
        print("Training implicit correction model...")
        training_table = pd.read_csv(args.training_data)
        training_data = gidcl.generate_training_data([training_table])
        
        training_result = gidcl.train_implicit_model(training_data, epochs=3)
        print(f"Training completed. Loss: {training_result['training_loss']:.4f}")
    
    # Detect errors
    print("Detecting errors...")
    error_detection_config = config.get('error_detection', {})
    error_cells = detect_errors(input_table, error_detection_config)
    
    if args.verbose:
        print(f"Found {len(error_cells)} error cells")
        for i, error in enumerate(error_cells[:5]):  # Show first 5 errors
            print(f"  Error {i+1}: Row {error.row_idx}, Col {error.col_idx}, "
                  f"Value: '{error.value}', Type: {error.error_type.value}")
    
    # Create schema information
    schema_info = create_schema_info(input_table)
    
    # Prepare auxiliary context
    auxiliary_context = config.get('auxiliary_context', {})
    
    # Repair table
    print("Repairing table...")
    repaired_table, correction_results = gidcl.repair_table(
        input_table,
        error_cells,
        schema_info,
        auxiliary_context
    )
    
    # Save results
    print(f"Saving repaired table to {args.output}...")
    repaired_table.to_csv(args.output, index=False)
    
    # Save correction report
    report_path = args.output.replace('.csv', '_report.json')
    report = {
        "original_shape": input_table.shape,
        "repaired_shape": repaired_table.shape,
        "num_errors_detected": len(error_cells),
        "num_corrections_made": len(correction_results),
        "corrections": []
    }
    
    for result in correction_results:
        report["corrections"].append({
            "original_value": str(result.original_value),
            "corrected_value": str(result.corrected_value),
            "method": result.method.value,
            "confidence": result.confidence,
            "rule": result.rule
        })
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Correction report saved to {report_path}")
    
    # Print summary
    print("\n=== REPAIR SUMMARY ===")
    print(f"Original table shape: {input_table.shape}")
    print(f"Repaired table shape: {repaired_table.shape}")
    print(f"Errors detected: {len(error_cells)}")
    print(f"Corrections made: {len(correction_results)}")
    
    method_counts = {}
    for result in correction_results:
        method = result.method.value
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print("Correction methods used:")
    for method, count in method_counts.items():
        print(f"  {method}: {count}")
    
    avg_confidence = np.mean([r.confidence for r in correction_results])
    print(f"Average correction confidence: {avg_confidence:.3f}")


if __name__ == "__main__":
    main()