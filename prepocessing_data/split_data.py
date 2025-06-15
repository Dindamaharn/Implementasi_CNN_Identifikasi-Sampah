# preprocessing_data/split_data.py
from splitfolders import ratio

# Split data: 80% train, 20% val
input_folder = "dataset/original"
output_folder = "dataset/datasplit"

ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.2), group_prefix=None)
print("✅ Data split selesai!")
