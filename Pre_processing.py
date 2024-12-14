import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the EMNIST dataset
# Replace 'path_to_emnist.csv' with the actual path to the EMNIST dataset
file_path = 'emnist-bymerge-train.csv'  # Update this path
emnist_data = pd.read_csv(file_path)

# Assume the EMNIST dataset has columns 'label' and features as 'pixel1', 'pixel2', ..., 'pixel784'
# Adjust if the column names differ
labels = emnist_data.iloc[:, 0]
print(labels)
# Group by label and find the minimum number of samples across all labels

# Specify the desired minimum samples per label
desired_min_samples = 2500  # Adjust as needed

# Balance the dataset by ensuring each label has the desired minimum samples
balanced_data = (
    emnist_data.groupby(emnist_data.columns[0])
    .apply(lambda x: x.sample(n=min(len(x), desired_min_samples), random_state=42, replace=True))
    .reset_index(drop=True)
)

# Save the balanced dataset to a CSV file
output_path = 'emnist_balanced_train.csv'  # Desired output file path
balanced_data.to_csv(output_path, index=False)

print(f"Balanced dataset saved to {output_path}")

# Plot the frequency of each label
label_counts = balanced_data[balanced_data.columns[0]].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(label_counts.index, label_counts.values, color='skyblue')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.title('Frequency of Each Label in the Balanced Dataset')
plt.xticks(label_counts.index)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
