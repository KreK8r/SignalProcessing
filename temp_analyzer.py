import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.signal import find_peaks

temperatures = [0.5, 0.7, 1.0, 1.6, 5, 7, 10, 14, 20, 35]

def load_data(filepath):
    return pd.read_csv(filepath)

def extract_columns(data):
    x_data = data.iloc[:, ::2].values
    y_data = data.iloc[:, 1::2].values
    return x_data, y_data

def find_largest_peaks(x_fft, y_fft):
    largest_peaks_y = []
    largest_peaks_x = []

    for i in range(y_fft.shape[1]):  # Loop through each column in y_fft
        peaks, _ = find_peaks(y_fft[:, i])  # Find peaks in y_fft column
        
        if len(peaks) > 0:
            # Get the largest peak
            largest_peak_index = np.argmax(y_fft[peaks, i])  # Find index of the largest peak
            largest_peak_y = y_fft[peaks[largest_peak_index], i]  # Corresponding largest y value
            largest_peak_x = x_fft[peaks[largest_peak_index], i]  # Corresponding x value
        else:
            largest_peak_y = np.nan
            largest_peak_x = np.nan

        largest_peaks_y.append(largest_peak_y)
        largest_peaks_x.append(largest_peak_x)

    return largest_peaks_y, largest_peaks_x

# Load and extract the data
td_fft = load_data('fft.csv')
x_fft, y_fft = extract_columns(td_fft)

# Find the largest peaks and corresponding x values
largest_peaks_y, largest_peaks_x = find_largest_peaks(x_fft, y_fft)

# Plot the data and the largest peaks
colors = cm.nipy_spectral(np.linspace(0, 1, len(temperatures)))

for i in range(len(temperatures)):
    plt.plot(x_fft[:, i], y_fft[:, i], label=f"{temperatures[i]}K", color=colors[i])
    plt.plot(largest_peaks_x[i], largest_peaks_y[i], 'x', color='black', markersize=10)  # Mark the largest peak

plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True)
plt.show()

# Print the largest peak values for each temperature
print(temperatures)
print(f"frequency: {largest_peaks_x}")
print(f"amplitudes: {largest_peaks_y}")
