import numpy as np
import pandas as pd
from .max_modularity import spectral_modularity_maximization, embedding2graph

"""
df1: clean, df2: repaired, df3: original
"""
def error_drop_rate(df_clean, df_dirty, df_repaired):
    """
    Calculate the error drop rate using edit distance.
    
    This metric measures how much the repair process reduced the distance
    to the clean data compared to the original dirty data.
    
    Args:
        df_clean (pd.DataFrame): Clean/ground truth dataset
        df_dirty (pd.DataFrame): Original dirty dataset
        df_repaired (pd.DataFrame): Repaired dataset
        
    Returns:
        float: Error drop rate as (d2-d1)/d1 where:
               d1 = edit_distance(clean, dirty)
               d2 = edit_distance(clean, repaired)
               
               Positive values indicate improvement (error reduction)
               Negative values indicate degradation (error increase)
    """
    # Calculate edit distance between clean and dirty (baseline error)
    d1 = edit_distance(df_clean, df_dirty)
    
    # Calculate edit distance between clean and repaired (remaining error)
    d2 = edit_distance(df_clean, df_repaired)
    
    # Calculate error drop rate: (d2-d1)/d1
    # Note: This is the negative of improvement rate
    # Positive values mean errors increased (bad)
    # Negative values mean errors decreased (good)
    if d1 == 0:
        # If clean and dirty are identical, return 0 or handle special case
        return 0.0 if d2 == 0 else float('inf')
    
    error_drop_rate = (d2 - d1) / d1
    return error_drop_rate


def F1_score(df1, df2, df3):
    new_errors = (df1 != df2)
    old_errors = (df1 != df3)
    new_errors_count = int(new_errors.values.sum())
    old_errors_count = int(old_errors.values.sum())
    recall = (old_errors_count-new_errors_count) / old_errors_count
    f1 = 2 * recall / (1 + recall)
    return f1


def edit_distance(df1, df2):
    """
    Calculate the average normalized Levenshtein distance between two dataframes.
    
    Compares corresponding cells in two dataframes and calculates the normalized
    Levenshtein distance for each pair, then returns the average.
    
    Args:
        df1 (pd.DataFrame): First dataframe
        df2 (pd.DataFrame): Second dataframe
        
    Returns:
        float: Average normalized Levenshtein distance (0-1, where 0 means identical)
    """
    # Ensure dataframes have the same shape
    if df1.shape != df2.shape:
        raise ValueError(f"DataFrames must have the same shape. Got {df1.shape} and {df2.shape}")
    
    total_distance = 0
    total_comparisons = 0
    
    # Iterate through all cells in the dataframes
    for col in df1.columns:
        for idx in df1.index:
            s1 = str(df1.loc[idx, col])
            s2 = str(df2.loc[idx, col])
            
            # Calculate Levenshtein distance
            lev_dist = levenshtein_distance(s1, s2)
            
            # Normalize by the maximum length of the two strings
            max_len = max(len(s1), len(s2))
            if max_len > 0:
                normalized_dist = lev_dist / max_len
            else:
                # Both strings are empty, distance is 0
                normalized_dist = 0
            
            total_distance += normalized_dist
            total_comparisons += 1
    
    # Return average normalized distance
    if total_comparisons > 0:
        return total_distance / total_comparisons
    else:
        return 0


def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    
    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string
    into another.
    
    Args:
        s1 (str): First string
        s2 (str): Second string
        
    Returns:
        int: The Levenshtein distance between s1 and s2
    """
    # Convert to strings in case non-string inputs are passed
    s1, s2 = str(s1), str(s2)
    
    # If one string is empty, return the length of the other
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    # Create a matrix to store distances
    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Initialize first row and column
    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j
    
    # Fill the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1
            
            dist[i][j] = min(
                dist[i-1][j] + 1,      # deletion
                dist[i][j-1] + 1,      # insertion
                dist[i-1][j-1] + cost  # substitution
            )
    
    return dist[rows-1][cols-1]


def all_error_correcter(df1, df2, correct_rate=0.2, threshold=0.95):
    """
    df1: dirty, df2: clean
    retrun a new df with some errors corrected
    here the corrected errors are randomly selected
    """
    new_df = df1.copy()
    n_rows = len(new_df)
    for col in new_df.columns:
        error_mask = (new_df[col] != df2[col])
        error_rate = error_mask.sum() / n_rows
        if error_rate > correct_rate:
            new_df[col] = df2[col]
            to_correct = error_mask[error_mask].sample(frac=correct_rate+error_rate-1, random_state=42).index
            new_df.loc[to_correct, col] = df2.loc[to_correct, col]
    return new_df



def dataset_corrupter(dataset_path, error_rate, output_path):
    """
    Read CSV, introduce errors, and save corrupted version.
    
    Args:
        dataset_path: Path to input CSV file
        error_rate: Rate of corruption (0-1)
        output_path: Path to save corrupted CSV
    """
    # Step 1: Read the CSV file
    print(f"Reading CSV from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    
    return df

def test(a, b):
    return a + b

def group_by_modularity(df, sample_rows=30, dimensions=200, resolution=1):
    similarity_matrix = embedding2graph(df, num_rows=sample_rows, dimensions=dimensions)
    labels, Q = spectral_modularity_maximization(similarity_matrix, resolution=resolution)
    return labels

def all_wrong_corrector(clean_df, dirty_df, error_df, prop=0.2):
    """
    used to correct columns that are all wrong
    """
    col_sum = error_df.sum(axis=0)
    for col in col_sum.index:
        # too many error, do correction
        if col_sum[col] > (1-prop) * error_df.shape[0]:
            correct_sample = np.random.choice(error_df.shape[0], int(prop * error_df.shape[0]), replace=False)
            dirty_df.loc[correct_sample, col] = clean_df.loc[correct_sample, col]
        else:
            continue

    error_df = (clean_df != dirty_df)
    return clean_df, dirty_df, error_df


if __name__ == "__main__":
    dirty_path = './dataset/flight/dirty.csv'
    clean_path = './dataset/flight/clean.csv'

    df1 = pd.read_csv(clean_path)
    df2 = pd.read_csv(dirty_path)
    print(F1_score(df1, df2, df2))
