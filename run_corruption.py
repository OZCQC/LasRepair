from dataset_corrupter import dataset_corrupter

# Run corruption on Walmart dataset
dataset_path = "/root/datarepair/datasets/shuttle/clean.csv"
output_path = "/root/datarepair/datasets/shuttle/dirty_20.csv"
error_rate = 0.2
available_columns_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # All 5 columns

print("=== WALMART DATASET CORRUPTION ===")
print(f"Input: {dataset_path}")
print(f"Output: {output_path}")
print(f"Error rate: {error_rate}")
print(f"Available columns for corruption: {available_columns_index}")
print("="*50)

# Run the corruption
df_corrupted, corruption_log = dataset_corrupter(
    dataset_path=dataset_path,
    error_rate=error_rate, 
    output_path=output_path,
    available_columns_index=available_columns_index
)

print("\n" + "="*50)
print("CORRUPTION COMPLETED!")
print("="*50) 