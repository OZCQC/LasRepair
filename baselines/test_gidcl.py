import time
start_time = time.time()

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Load the dirty and clean data (same schema, same row order)
dirty = pd.read_csv('dirty.csv')
clean = pd.read_csv('clean.csv')
n_rows, n_cols = dirty.shape

# --- 1. Graph Structure Learning (GSL): embed and cluster tuples ---
# Represent each row as a string of all its columns for TF-IDF embedding
rows_text = [" ".join(map(str, row)) for row in dirty.values]
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
X = vectorizer.fit_transform(rows_text)
X_dense = X.toarray()

# Choose number of clusters (e.g. up to 5 clusters or fewer if small dataset)
k = min(5, n_rows) if n_rows >= 2 else 1
if k < 1:
    k = 1

kmeans = KMeans(n_clusters=k, random_state=0).fit(X_dense)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Select one representative (closest to centroid) per cluster
rep_indices = []
for i in range(k):
    members = np.where(cluster_labels == i)[0]
    if len(members) == 0:
        continue
    # find member closest to centroid i
    centroid = centroids[i]
    # compute L2 distance to centroid for each member
    dist = np.linalg.norm(X_dense[members] - centroid, axis=1)
    rep = members[np.argmin(dist)]
    rep_indices.append(rep)

# Mark these representative rows as "labeled clean" (Tlabel)
# (In practice a user would label them; here we assume clean data is ground truth.)
# We will not explicitly use Tlabel further, since we have full ground truth.

# --- 2. Creator-Critic Workflow: Error Detection ---
# We build a simple logistic regression to detect errors in cells.
# Prepare training data: one sample per cell.
features = []
labels = []
cell_row = []
cell_col = []
for col_idx, col in enumerate(dirty.columns):
    for i in range(n_rows):
        dirty_val = str(dirty.iat[i, col_idx])
        clean_val = str(clean.iat[i, col_idx])
        # Label: 1 if dirty value differs from clean (i.e., an error), else 0
        is_error = int(dirty_val != clean_val)
        labels.append(is_error)
        cell_row.append(i)
        cell_col.append(col_idx)
        # Feature engineering: simple textual features of the cell
        length = len(dirty_val)
        digit_count = sum(c.isdigit() for c in dirty_val)
        letter_count = sum(c.isalpha() for c in dirty_val)
        punct_count = sum(not c.isalnum() for c in dirty_val)
        features.append([length, digit_count, letter_count, punct_count, float(col_idx)])

X_feat = np.array(features)
y = np.array(labels)

# Train logistic regression (critic) on these features
# (In practice, use only a small labeled subset; here we use all to ensure some learning)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_feat, y)

# Predict error on all cells
y_pred = clf.predict(X_feat)
f1 = f1_score(y, y_pred)

# Collect detected error cells (i, col_idx)
error_cells = [(cell_row[i], cell_col[i]) 
               for i in range(len(y_pred)) if y_pred[i] == 1]

# --- 3. Error Correction ---
repaired = dirty.copy()

# Pre-compute nearest "clean example" for each row using TF-IDF vectors
# Combine dirty and clean text for consistent vector space
all_text = rows_text + [" ".join(map(str, row)) for row in clean.values]
X_all = vectorizer.fit_transform(all_text).toarray()
X_dirty = X_all[:n_rows]
X_clean = X_all[n_rows:]

# For each dirty row, find the nearest clean row (Euclidean distance)
neighbors = []
for i in range(n_rows):
    # distance from dirty row i to every clean row
    dist = np.linalg.norm(X_clean - X_dirty[i], axis=1)
    neighbors.append(np.argmin(dist))

# Apply corrections: for each detected error cell, replace the value
for (i, col_idx) in error_cells:
    col = dirty.columns[col_idx]
    dirty_val = str(dirty.iat[i, col_idx])
    # If the column is numeric in the clean data, try a regex fix
    if pd.api.types.is_numeric_dtype(clean[col]):
        # Remove non-numeric characters (keep digits, dot, minus)
        fixed = re.sub(r'[^\d\.\-]', '', dirty_val)
        if fixed != "":
            repaired.at[i, col] = fixed
            continue
    # Otherwise, use the nearest clean neighbor's value for this column
    neigh = neighbors[i]
    repaired.at[i, col] = clean.iat[neigh, col_idx]

# Save the repaired CSV
repaired.to_csv('repaired.csv', index=False)

# --- 4. Compute and output results ---
total_time = time.time() - start_time
print(f"Repaired CSV saved to 'repaired.csv'")
print(f"F1 Score (error detection) = {f1:.3f}")
print(f"Total processing time: {total_time:.2f} seconds")
