import sys
sys.path.append('../')
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *
import pandas as pd


def create_evaluation_format(clean_csv_path, output_path):
    """Convert clean.csv to the format expected by HoloClean evaluation"""
    df = pd.read_csv(clean_csv_path)
    
    eval_data = []
    for tid, row in df.iterrows():
        for attr in df.columns:
            eval_data.append({
                'tid': tid,
                'attribute': attr, 
                'correct_val': row[attr]
            })
    
    eval_df = pd.DataFrame(eval_data)
    eval_df.to_csv(output_path, index=False)
    print(f"Created evaluation file: {output_path}")


# 1. Create evaluation format from clean data
create_evaluation_format('../../../datasets/beers/clean.csv', 
                        '../../../datasets/beers/beers_clean_eval.csv')

# 2. Setup a HoloClean session.
hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    epochs=10,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=False,  # Reduce verbosity for cleaner output
    timeout=3*60000,
    feature_norm=False,
    weight_norm=False,
    print_fw=False
).session

# 3. Load training data and denial constraints.
hc.load_data('beers', '../../../datasets/beers/dirty.csv')
hc.load_dcs('../../../datasets/beers/beers_constraints.txt')
hc.ds.set_constraints(hc.get_dcs())

# 4. Detect erroneous cells using these two detectors.
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)

# 5. Repair errors utilizing the defined features.
hc.setup_domain()
featurizers = [
    InitAttrFeaturizer(),
    OccurAttrFeaturizer(),
    FreqFeaturizer(),
    ConstraintFeaturizer(),
]

hc.repair_errors(featurizers)

# 6. Evaluate and get F1 score
print("=== BEERS DATASET EVALUATION RESULTS ===")
report = hc.evaluate(fpath='../../../datasets/beers/beers_clean_eval.csv',
                    tid_col='tid',
                    attr_col='attribute', 
                    val_col='correct_val')

print(f"F1 Score: {report.f1:.4f}")
print(f"Precision: {report.precision:.4f}")
print(f"Recall: {report.recall:.4f}")
print(f"Repairing F1: {report.repair_f1:.4f}")
print(f"Correct Repairs: {report.correct_repairs}")
print(f"Total Repairs: {report.total_repairs}")
print(f"Total Errors: {report.total_errors}") 