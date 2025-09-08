#!/usr/bin/env python3
"""
Script to run BigDansing on all available datasets
"""

import os
import subprocess
import time
from pathlib import Path

# Base paths
BASELINES_DIR = "/root/datarepair/baselines/bigdansing_holistic"
DATASETS_DIR = "/root/datarepair/datasets"

# Available datasets (excluding tax as requested)
DATASETS = ["beers", "flight", "movies", "rayyan", "shuttle", "walmart"]

def setup_output_directories(dataset_name):
    """Create necessary output directories for a dataset"""
    task_dir = dataset_name[:-1] if len(dataset_name) > 1 else dataset_name  # Remove last character as per BigDansing logic
    
    exp_dir = Path(BASELINES_DIR) / "Exp_result" / "bigdansing" / task_dir
    rep_dir = Path(BASELINES_DIR) / "Repaired_res" / "bigdansing" / task_dir
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directories for {dataset_name}: {task_dir}")

def run_bigdansing(dataset_name):
    """Run BigDansing on a specific dataset"""
    print(f"\n{'='*60}")
    print(f"Running BigDansing on {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Setup directories
    setup_output_directories(dataset_name)
    
    # Construct file paths
    clean_path = f"{DATASETS_DIR}/{dataset_name}/clean.csv"
    
    # Find dirty file (different naming conventions)
    dirty_files = [
        f"{DATASETS_DIR}/{dataset_name}/dirty.csv",
        f"{DATASETS_DIR}/{dataset_name}/dirty_20.csv"
    ]
    
    dirty_path = None
    for df in dirty_files:
        if os.path.exists(df):
            dirty_path = df
            break
    
    # Find constraints file (different naming conventions)
    constraints_files = [
        f"{DATASETS_DIR}/{dataset_name}/{dataset_name}_constraints.txt",
        f"{DATASETS_DIR}/{dataset_name}/constraints.txt",
        f"{DATASETS_DIR}/{dataset_name}/dc_rules.txt"
    ]
    
    rule_path = None
    for cf in constraints_files:
        if os.path.exists(cf):
            rule_path = cf
            break
    
    if not rule_path:
        print(f"❌ No constraints file found for {dataset_name}")
        return False
    
    if not dirty_path:
        print(f"❌ No dirty file found for {dataset_name}")
        return False
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [clean_path, dirty_path, rule_path]):
        print(f"❌ Missing files for {dataset_name}")
        print(f"  Clean: {os.path.exists(clean_path)}")
        print(f"  Dirty: {os.path.exists(dirty_path)}")
        print(f"  Rules: {os.path.exists(rule_path)}")
        return False
    
    # Run BigDansing
    cmd = [
        "python", "bigdansing.py",
        "--clean_path", clean_path,
        "--dirty_path", dirty_path,
        "--rule_path", rule_path,
        "--task_name", dataset_name,
        "--onlyed", "0",
        "--perfected", "0"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=BASELINES_DIR,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {dataset_name} completed successfully in {execution_time:.2f} seconds")
            return True
        else:
            print(f"❌ {dataset_name} failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset_name} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"💥 {dataset_name} failed with exception: {e}")
        return False

def main():
    """Main function to run BigDansing on all datasets"""
    print("🚀 Starting BigDansing evaluation on all datasets")
    print(f"Datasets to process: {', '.join(DATASETS)}")
    
    results = {}
    overall_start = time.time()
    
    for dataset in DATASETS:
        success = run_bigdansing(dataset)
        results[dataset] = success
    
    overall_end = time.time()
    total_time = overall_end - overall_start
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    successful = [d for d, success in results.items() if success]
    failed = [d for d, success in results.items() if not success]
    
    print(f"✅ Successful ({len(successful)}/{len(DATASETS)}): {', '.join(successful)}")
    if failed:
        print(f"❌ Failed ({len(failed)}/{len(DATASETS)}): {', '.join(failed)}")
    
    print(f"\n📁 Results are stored in: {BASELINES_DIR}/Exp_result/bigdansing/")
    print(f"📁 Repaired data stored in: {BASELINES_DIR}/Repaired_res/bigdansing/")

if __name__ == "__main__":
    main()
