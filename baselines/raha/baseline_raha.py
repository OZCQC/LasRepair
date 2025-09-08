import os
import raha

# Update this to point to your dataset, only error correction
# dataset_name = "movies"
# dataset_dictionary = {
#     "name": dataset_name,
#     "path": "/root/datarepair/datasets/movies/dirty.csv",
#     "clean_path": "/root/datarepair/datasets/movies/clean.csv"
# }
dataset_name = "beers"
dataset_dictionary = {
    "name": dataset_name,
    "path": "/root/datarepair/datasets/beers/dirty.csv",
    "clean_path": "/root/datarepair/datasets/beers/clean.csv"
}

# Load dataset and get actual errors
data = raha.dataset.Dataset(dataset_dictionary)
data.detected_cells = dict(data.get_actual_errors_dictionary())

# Run correction
app = raha.correction.Correction()
app.VERBOSE = True  # Set to see progress
correction_dictionary = app.run(data)

# Evaluate results
p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
print("Baran's correction performance on {}:\nPrecision = {:.4f}\nRecall = {:.4f}\nF1 = {:.4f}".format(data.name, p, r, f))