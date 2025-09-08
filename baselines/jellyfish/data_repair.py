import pandas as pd
import jellyfish

# Load dirty and clean datasets
dirty_df = pd.read_csv("../../datasets/beers/dirty.csv")
clean_df = pd.read_csv("../../datasets/beers/clean.csv")

# Create a boolean mask of errors (True where dirty≠clean)
error_mask = dirty_df != clean_df

# Create a copy of dirty data for repairs
repaired_df = dirty_df.copy()

# Repair each error by finding the best match using Jaro-Winkler similarity
for col in dirty_df.columns:
    # Get all unique values in this column as candidates
    candidates = dirty_df[col].astype(str).unique()
    
    # Find rows that have errors in this column
    error_rows = error_mask[col]
    
    # Repair each error in this column
    for idx in error_rows[error_rows].index:
        dirty_value = str(dirty_df.at[idx, col])
        
        # Find the best candidate using Jaro-Winkler similarity
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            score = jellyfish.jaro_winkler_similarity(dirty_value, str(candidate))
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        # Apply the repair
        repaired_df.at[idx, col] = best_candidate

# Save repaired data
repaired_df.to_csv("repaired.csv", index=False)

# Calculate evaluation metrics
tp = ((dirty_df != clean_df) & (repaired_df == clean_df)).sum().sum()
fp = ((dirty_df == clean_df) & (repaired_df != clean_df)).sum().sum()
fn = ((dirty_df != clean_df) & (repaired_df != clean_df)).sum().sum()

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"Errors repaired: {tp}/{(dirty_df != clean_df).sum().sum()}")