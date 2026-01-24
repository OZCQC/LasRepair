"""
Master script to run all baselines on specified datasets
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

BASELINES = [
    'jellyfish',
    'raha',
    'activedetect',
    'bigdansing',
    'holoclean',
    'gidcl',
    'mlnclean'
]

DATASETS = [
    'beers',
    'flight',
    'hospital',
    'movies',
    'rayyan',
    'shuttle',
    'tax_200k',
    'tax_20k',
    'walmart'
]


def run_baseline(baseline, dataset, data_dir, output_dir):
    """Run a specific baseline on a specific dataset"""
    script_name = f'run_{baseline}.py'
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running {baseline.upper()} on {dataset}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, script_path,
        '--dataset', dataset,
        '--data_dir', data_dir,
        '--output_dir', output_dir
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {baseline} on {dataset} completed successfully")
            return True
        else:
            print(f"❌ {baseline} on {dataset} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {baseline} on {dataset} timed out after 2 hours")
        return False
    except Exception as e:
        print(f"💥 {baseline} on {dataset} failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all baselines on datasets')
    parser.add_argument('--baselines', nargs='+', default=BASELINES,
                        choices=BASELINES, help='Baselines to run')
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help='Datasets to test')
    parser.add_argument('--data_dir', type=str, default='/data1/qianc/EMCL/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='/data1/qianc/EMCL/baselines/results',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    print("🚀 Starting baseline evaluation")
    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    
    results = {}
    
    for baseline in args.baselines:
        results[baseline] = {}
        for dataset in args.datasets:
            # Check if dataset exists
            dataset_path = os.path.join(args.data_dir, dataset)
            if not os.path.exists(dataset_path):
                print(f"⚠️  Dataset {dataset} not found at {dataset_path}, skipping")
                results[baseline][dataset] = 'skipped'
                continue
            
            success = run_baseline(baseline, dataset, args.data_dir, args.output_dir)
            results[baseline][dataset] = 'success' if success else 'failed'
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    for baseline in args.baselines:
        successful = [d for d, r in results[baseline].items() if r == 'success']
        failed = [d for d, r in results[baseline].items() if r == 'failed']
        skipped = [d for d, r in results[baseline].items() if r == 'skipped']
        
        print(f"\n{baseline.upper()}:")
        print(f"  ✅ Success: {len(successful)}/{len(args.datasets)}")
        if failed:
            print(f"  ❌ Failed: {', '.join(failed)}")
        if skipped:
            print(f"  ⚠️  Skipped: {', '.join(skipped)}")
    
    print(f"\n📁 Results are stored in: {args.output_dir}")


if __name__ == "__main__":
    main()

