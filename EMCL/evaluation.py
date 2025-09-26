import numpy as np
import pandas as pd


def show_diff(repaired_path, clean_path, dirty_path):
    repaired_df = pd.read_csv(repaired_path)
    clean_df = pd.read_csv(clean_path)
    dirty_df = pd.read_csv(dirty_path)
    count = 0
    for row in range(repaired_df.shape[0]):
        for col in range(repaired_df.shape[1]):
            if repaired_df.iloc[row, col] != clean_df.iloc[row, col]:
                print(f"clean: {clean_df.iloc[row, col]}, repaired: {repaired_df.iloc[row, col]}, dirty: {dirty_df.iloc[row, col]}")
                count += 1
                if count > 20:
                    return 0

    return 0


if __name__ == "__main__":
    repaired_path = "/data1/qianc/result/flight_repaired_multi.csv"
    clean_path = "/data1/qianc/EMCL/datasets/flight/clean.csv"
    dirty_path = "/data1/qianc/EMCL/datasets/flight/dirty.csv"
    show_diff(repaired_path, clean_path, dirty_path)
    # print(error_df)