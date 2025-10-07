import numpy as np
import pandas as pd
from utils import F1_score, all_wrong_corrector


def show_diff(repaired_path, clean_path, dirty_path):
    repaired_df = pd.read_csv(repaired_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '').astype(str)
    clean_df = pd.read_csv(clean_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '').astype(str)
    dirty_df = pd.read_csv(dirty_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '').astype(str)
    repaired_df.columns = clean_df.columns
    dirty_df.columns = clean_df.columns
    count = 0
    print(f"F1 score: {F1_score(clean_df, repaired_df, dirty_df)}")
    for row in range(repaired_df.shape[0]):
        for col in range(repaired_df.shape[1]):
            if repaired_df.iloc[row, col] != clean_df.iloc[row, col]:
                print(f"clean: {clean_df.iloc[row, col]}, repaired: {repaired_df.iloc[row, col]}, dirty: {dirty_df.iloc[row, col]}")
                count += 1
                if count > 20:
                    return 0

    return 0


if __name__ == "__main__":
    experiment = "movies"
    # repaired_path = f"/data1/qianc/result/{experiment}_repaired_original.csv"
    repaired_path = f"/data1/qianc/result/{experiment}_repaired_original.csv"
    clean_path = f"/data1/qianc/EMCL/datasets/{experiment}/clean.csv"
    dirty_path = f"/data1/qianc/EMCL/datasets/{experiment}/dirty.csv"
    show_diff(repaired_path, clean_path, dirty_path)