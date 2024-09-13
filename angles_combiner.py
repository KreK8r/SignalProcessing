import os
import pandas as pd

# Define the path to your folder
name = 'fft_angles'
folder = 'YbTi3Bi4-1'
folder_path = '/root/research/publications/' + folder + '/fft_angles'

# List of file numbers and corresponding ordered filenames
#numbers = [0.5, 0.7, 1.0, 1.6, 5, 7, 10, 14, 20, 35]
numbers = [-28.0, -21.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0, 24.5, 28.0, 31.5, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0, 84.0, 91.0, 98.0, 105.0, 112.0, 119.0, 126.0, 133.0, 140.0, 147.0]
ordered_files = [str(number) + '.txt' for number in numbers]

# Define columns to keep (0-based index)
columns_to_keep = [0, 1]  # Adjust as needed

# Initialize an empty list to hold DataFrames
dataframes = []

# Read each file in the specified order, select specific columns, and append its DataFrame to the list
for file in ordered_files:
    if file != '007.txt':
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, sep='\s+', header=None)  # Use sep='\s+' to handle whitespace delimiter
        selected_columns = df.iloc[:, columns_to_keep]
        dataframes.append(selected_columns)

# Concatenate all DataFrames along columns
combined_df = pd.concat(dataframes, axis=1)

# Save the combined DataFrame to a CSV file
output_file_path = f'/root/research/publications/YbTi3Bi4-1/{name}.csv'
combined_df.to_csv(output_file_path, index=False, header=False)

print(f"Combined file saved to: {output_file_path}")