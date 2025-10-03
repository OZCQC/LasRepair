import os
import raha
import argparse
import time

# Update this to point to your dataset, only error correction
# dataset_name = "movies"
# dataset_dictionary = {
#     "name": dataset_name,
#     "path": "/root/datarepair/datasets/movies/dirty.csv",
#     "clean_path": "/root/datarepair/datasets/movies/clean.csv"
# }
if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="walmart")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_dictionary = {
        "name": "not " + dataset_name,
        "path": f"/data1/qianc/EMCL/datasets/{dataset_name}/dirty.csv",
        "clean_path": f"/data1/qianc/EMCL/datasets/{dataset_name}/clean.csv"
    }

    # Load dataset and get actual errors
    data = raha.dataset.Dataset(dataset_dictionary)
    data.detected_cells = dict(data.get_actual_errors_dictionary())

    # Run correction
    app = raha.correction.Correction()
    app.VERBOSE = False  # Set to see progress
    correction_dictionary = app.run(data)

    # Create repaired dataset and save as CSV
    data.create_repaired_dataset(correction_dictionary)

    # Save the corrected dataset as CSV
    output_dir = "/data1/qianc/results/raha"
    os.makedirs(output_dir, exist_ok=True)
    corrected_csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    data.repaired_dataframe.to_csv(corrected_csv_path, index=False)

    print(f"Corrected dataset saved to: {corrected_csv_path}")

    # Evaluate results
    p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
    print("Baran's correction performance on {}:\nPrecision = {:.4f}\nRecall = {:.4f}\nF1 = {:.4f}".format(data.name, p, r, f))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")