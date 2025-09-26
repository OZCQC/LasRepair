# from embedding2graph import embedding2graph
# from max_modularity import spectral_modularity_maximization
# import pandas as pd

# test_csv = pd.read_csv("/data1/qianc/EMCL/datasets/flight/clean.csv")
# similarity_matrix = embedding2graph(test_csv, num_rows=30, dimensions=200)
# labels, Q = spectral_modularity_maximization(similarity_matrix, resolution=1)
# a = zip(test_csv.columns.tolist(), labels)
# a = sorted(a, key=lambda x: x[1])
# print(a)
# print(Q)
import numpy as np
import pandas as pd


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


# clean = pd.read_csv("/data1/qianc/EMCL/datasets/beers/clean.csv")
# dirty = pd.read_csv("/data1/qianc/EMCL/datasets/beers/dirty.csv")
# error_df = (clean != dirty)
# print(error_df.sum(axis=0))

# clean_df, dirty_df, error_df = all_wrong_corrector(clean, dirty, error_df)
# print(error_df.sum(axis=0))

a = np.random.choice(2034, 5, replace=False)
print(a)
print(type(a))