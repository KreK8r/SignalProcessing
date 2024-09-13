import os
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def extract_columns(data):
    x_data = data.iloc[:, ::2].values
    y_data = data.iloc[:, 1::2].values
    return x_data, y_data

import numpy as np

def load_data(filepath):
    """Load data from a file."""
    with open(filepath, 'r') as file:
        data = file.readlines()
    return data

def extr_c(data):
    """Extract columns from the data."""
    x = []
    y = []
    for line in data:
        if '\t' in line:
            columns = line.strip().split('\t')
            try:
                x.append(float(columns[0]))
                y.append(float(columns[1]))
            except ValueError as e:
                print(f"Skipping line due to error: {e}")
                continue
    return np.array(x), np.array(y)

def main():
    filepath = 'data/PracticeData.txt'  # Adjust the path as needed

    # Load data from file
    data = load_data(filepath)
    
    # Extract x and y columns
    x, y = extract_columns(data)
    
    # Print the extracted data for verification
    print("x:", x)
    print("y:", y)

if __name__ == "__main__":
    main()


def save_to_csv(data, output_dir, filename, columns=None):
    """
    Save data to a CSV file.

    Parameters:
    - data: Data to save, as a list of lists or a 2D array.
    - output_dir: Directory where the CSV will be saved.
    - filename: Name of the output CSV file.
    - columns: Optional list of column names for the CSV file.
    
    Returns:
    - None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define full file path
    full_path = os.path.join(output_dir, filename)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(full_path, index=False)

    print(f"Data saved to {full_path}")

def process_files(name, folder, path, numbers, columns_to_keep=[0, 1]):
    """
    Process files from a folder, keep specified columns, and combine them into a single CSV.

    Args:
        name (str): The name of the output file.
        folder (str): The folder where the files are located.
        path (str): The path to the folder.
        numbers (list): A list of numbers used to generate the filenames.
        columns_to_keep (list): The indices of columns to keep from each file (default: [0, 1]).

    Returns:
        str: The path to the saved combined CSV file.
    """
    folder_path = os.path.join(path, folder)

    # Generate the list of ordered filenames
    ordered_files = [str(number) + '.txt' for number in numbers]

    # Initialize an empty list to hold DataFrames
    dataframes = []

    # Read each file, select specific columns, and append to the list
    for file in ordered_files:
        file_path = os.path.join(folder_path, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep='\s+', header=None)
            selected_columns = df.iloc[:, columns_to_keep]
            dataframes.append(selected_columns)
        else:
            print(f"Warning: File {file_path} does not exist.")

    # Concatenate all DataFrames along columns
    combined_df = pd.concat(dataframes, axis=1)

    # Save the combined DataFrame to a CSV file
    output_file_path = os.path.join(path, folder, f'{name}.csv')
    combined_df.to_csv(output_file_path, index=False, header=False)

    print(f"Combined file saved to: {output_file_path}")
    return output_file_path