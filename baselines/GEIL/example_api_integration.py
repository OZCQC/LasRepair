#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import argparse
from gidcl import GIDCL, ErrorCell, ErrorType
from utils import ErrorGenerator, EvaluationMetrics, Logger, Profiler


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


def demo_api_provider(provider_name: str):
    """Demonstrate GEIL with a specific API provider."""
    
    logger = Logger()
    profiler = Profiler()
    
    logger.info(f"=== GEIL API Integration Demo - {provider_name.upper()} ===")
    
    # Create configuration for the provider
    config = {
        "llm_provider": {
            "type": provider_name,
            "api_keys": {
                "openai": "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",
                "google": "AIzaSyDK4XLQ-q0A8ocPyuUA1izkUWqOSoZ3YTk",
                "anthropic": "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",
                "nebius": "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExMzAwNjU2MDY3MzI0ODc4MjUxOSIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwNDM1MzM2MCwidXVpZCI6IjNmMDJlNWJiLWFmYmEtNDY3NS05MDkzLThhNzc5ZGYzMDM5NCIsIm5hbWUiOiJIU0VfUHJvamVjdCIsImV4cGlyZXNfYXQiOiIyMDMwLTA1LTA3VDAzOjAyOjQwKzAwMDAifQ.EIpDHQXxtful-8joG8MFXAsORSxlsp7p3NFU0kz9jJ0",
                "deepseek": "sk-1015b59d6db94c42b1c5d34276b109d1"
            },
            "models": {
                "openai": "gpt-4o-mini",
                "google": "gemini-1.5-flash",
                "anthropic": "claude-3-haiku-20240307",
                "nebius": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "deepseek": "deepseek-chat"
            },
            "rate_limit": 60,
            "use_rate_limiting": True
        },
        "correction": {
            "implicit": {
                "use_api": True,
                "temperature": 0.7,
                "max_tokens": 200,
                "few_shot_examples": 5
            },
            "explicit": {
                "use_api": True,
                "temperature": 0.3,
                "max_tokens": 300,
                "max_iterations": 3,
                "confidence_threshold": 0.7
            }
        }
    }
    
    # Step 1: Create sample data
    logger.info("Step 1: Creating sample data")
    clean_table = create_sample_data()
    logger.info(f"Created table with shape: {clean_table.shape}")
    print("\nClean table:")
    print(clean_table.to_string())
    
    # Step 2: Introduce synthetic errors
    logger.info("Step 2: Introducing synthetic errors")
    error_generator = ErrorGenerator(error_rate=0.3)  # Higher error rate for demo
    
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
    
    # Step 3: Initialize GIDCL with API provider
    logger.info(f"Step 3: Initializing GIDCL with {provider_name}")
    
    with profiler.time_operation("gidcl_init"):
        gidcl = GIDCL(device="cpu", config=config)  # Use CPU for demo
    
    logger.info(f"GIDCL initialized with {provider_name} provider")
    
    # Step 4: Create error cells from error log
    logger.info("Step 4: Creating error cells")
    error_cells = []
    
    for error in error_log:
        # Map error types to GIDCL error types
        error_type_map = {
            'typo': ErrorType.SEMANTIC,
            'case': ErrorType.FORMATTING,
            'format': ErrorType.FORMATTING,
            'missing': ErrorType.CONTEXT_DEPENDENT,
            'outlier': ErrorType.SEMANTIC,
            'swap': ErrorType.PATTERN
        }
        
        error_cell = ErrorCell(
            row_idx=error['row_idx'],
            col_idx=error['col_idx'],
            value=error['corrupted_value'],
            error_type=error_type_map.get(error['error_type'], ErrorType.SEMANTIC),
            confidence=0.8,
            context={}
        )
        error_cells.append(error_cell)
    
    # Step 5: Repair table using API
    logger.info(f"Step 5: Repairing table using {provider_name}")
    
    schema_info = {
        "table_name": "employee_data",
        "columns": {col: {"type": str(dirty_table[col].dtype)} for col in dirty_table.columns}
    }
    
    auxiliary_context = {
        "examples": [
            ("  John Doe  ", "John Doe"),
            ("jane SMITH", "Jane Smith"),
            ("123.456.7890", "123-456-7890")
        ]
    }
    
    try:
        with profiler.time_operation("repair"):
            repaired_table, correction_results = gidcl.repair_table(
                dirty_table,
                error_cells,
                schema_info,
                auxiliary_context
            )
        
        logger.info("Table repair completed successfully")
        
        # Step 6: Display results
        print(f"\nRepaired table:")
        print(repaired_table.to_string())
        
        print(f"\nCorrection results:")
        for i, result in enumerate(correction_results):
            print(f"  {i+1}. '{result.original_value}' -> '{result.corrected_value}' "
                  f"(Method: {result.method.value}, Confidence: {result.confidence:.3f})")
            if result.rule:
                print(f"      Rule: {result.rule.replace(chr(10), ' ')[:100]}...")
        
        # Step 7: Evaluate accuracy
        logger.info("Step 7: Evaluating accuracy")
        
        error_positions = [(e['row_idx'], e['col_idx']) for e in error_log]
        metrics = EvaluationMetrics.calculate_repair_accuracy(
            clean_table, dirty_table, repaired_table, error_positions
        )
        
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Provider: {provider_name}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")
        
        # Step 8: Performance metrics
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Error generation time: {profiler.get_operation_time('error_generation'):.3f}s")
        print(f"GIDCL initialization time: {profiler.get_operation_time('gidcl_init'):.3f}s")
        print(f"Repair time: {profiler.get_operation_time('repair'):.3f}s")
        
        method_counts = {}
        for result in correction_results:
            method = result.method.value
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"Methods used: {method_counts}")
        avg_confidence = np.mean([r.confidence for r in correction_results])
        print(f"Average confidence: {avg_confidence:.3f}")
        
        return metrics, profiler.get_all_times()
        
    except Exception as e:
        logger.error(f"Error during repair with {provider_name}: {str(e)}")
        print(f"Failed to repair table with {provider_name}: {str(e)}")
        return None, None


def compare_providers():
    """Compare different API providers."""
    
    providers = ["openai", "anthropic", "google", "deepseek"]  # Skip nebius for now
    results = {}
    
    print("=" * 60)
    print("COMPARING API PROVIDERS")
    print("=" * 60)
    
    for provider in providers:
        print(f"\nTesting {provider}...")
        try:
            metrics, times = demo_api_provider(provider)
            if metrics:
                results[provider] = {
                    "metrics": metrics,
                    "times": times
                }
                print(f"✅ {provider}: Success (Accuracy: {metrics['accuracy']:.3f})")
            else:
                print(f"❌ {provider}: Failed")
        except Exception as e:
            print(f"❌ {provider}: Error - {str(e)}")
    
    # Summary comparison
    if results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"{'Provider':<12} {'Accuracy':<10} {'F1 Score':<10} {'Repair Time':<12}")
        print("-" * 50)
        
        for provider, data in results.items():
            metrics = data["metrics"]
            times = data["times"]
            repair_time = times.get("repair", 0)
            
            print(f"{provider:<12} {metrics['accuracy']:<10.3f} {metrics['f1']:<10.3f} {repair_time:<12.3f}s")
        
        # Best provider
        best_accuracy = max(results.items(), key=lambda x: x[1]["metrics"]["accuracy"])
        best_speed = min(results.items(), key=lambda x: x[1]["times"].get("repair", float('inf')))
        
        print(f"\n🏆 Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['metrics']['accuracy']:.3f})")
        print(f"⚡ Fastest: {best_speed[0]} ({best_speed[1]['times'].get('repair', 0):.3f}s)")


def main():
    parser = argparse.ArgumentParser(description='GEIL API Integration Demo')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'google', 'nebius', 'deepseek', 'all'], 
                       default='openai', help='API provider to test')
    parser.add_argument('--compare', action='store_true', help='Compare all providers')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_providers()
    elif args.provider == 'all':
        for provider in ['openai', 'anthropic', 'google', 'deepseek']:
            demo_api_provider(provider)
            print("\n" + "="*80 + "\n")
    else:
        demo_api_provider(args.provider)


if __name__ == "__main__":
    main() 