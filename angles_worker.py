import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import statsmodels.api as sm
from scipy.interpolate import interp1d
from numpy.fft import fft, fftfreq
from scipy.signal.windows import hamming

def scale_dimensions(width, height, width_scale=0.99, height_scale=0.95):
    """Scale the dimensions based on given scaling factors."""
    scaled_width = width * width_scale
    scaled_height = height * height_scale
    return scaled_width, scaled_height

def load_data(filepath):
    """Load the data from a CSV file."""
    return pd.read_csv(filepath)

def extract_columns(data):
    """Extract x and y data from the loaded DataFrame."""
    x_data = data.iloc[:, ::2].values
    y_data = data.iloc[:, 1::2].values
    return x_data, y_data

def plot_data(x_data, y_data, labels='', colors='black', x_label='', y_label='', title='', xlim=None, ylim=None, legend_title=''):
    """Plot the data with the specified parameters."""
    plt.figure(figsize=(scaled_width / 100, scaled_height / 100))
    for i, label in enumerate(labels):
        plt.plot(x_data[:, i], y_data[:, i], label=f'{label}', color=colors[i])

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title=legend_title)
    plt.grid(True)
    plt.show()

def trim_data(field, y):
    minimal = min(field) + 0.01
    maximum = max(field) - 0.01
    filtered = [(x, y_val) for x, y_val in zip(field, y) if x >= minimal]
    filtered2 = [(x, y_val) for x, y_val in filtered if x <= maximum]
    x_filtered, y_filtered = zip(*filtered2)
    return list(x_filtered), list(y_filtered)

def remove_duplicates(x, y):
    """Removes duplicate x-values by averaging the corresponding y-values."""
    unique_x, indices = np.unique(x, return_index=True)
    unique_y = [np.mean([y[i] for i in range(len(x)) if x[i] == ux]) for ux in unique_x]
    return unique_x, unique_y

def lowess_smoothing(x, y, frac=0.3):
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    trend_x = lowess[:, 0]
    trend_y = lowess[:, 1]
    return trend_x, trend_y

def polynomial_fit(x, y, degree = 3):
    """
    Fits a polynomial of a given degree to the data.

    Parameters:
    - x: A list or array of x-values.
    - y: A list or array of y-values.
    - degree: The degree of the polynomial to fit.

    Returns:
    - polynomial: A numpy poly1d object representing the fitted polynomial.
    - coefficients: The coefficients of the fitted polynomial.
    """
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    return polynomial, coefficients

def filter_bot(field, y, cutoff):
    filtered = [(x, y_val) for x, y_val in zip(field, y) if x >= cutoff]
    x_filtered, y_filtered = zip(*filtered)
    return list(x_filtered), list(y_filtered)

def filter_top(field, y, cutoff):
    filtered = [(x, y_val) for x, y_val in zip(field, y) if x <= cutoff]
    x_filtered, y_filtered = zip(*filtered)
    return list(x_filtered), list(y_filtered)

def compute_difference(x_filtered, y_filtered, x_lowess, y_lowess):
    """
    Compute the difference between y_filtered and y_lowess after interpolation.
    
    Parameters:
        x_filtered (list): x-values for filtered data.
        y_filtered (list): y-values for filtered data.
        x_lowess (list): x-values for LOWESS trend data.
        y_lowess (list): y-values for LOWESS trend data.
        
    Returns:
        x_common (list): Common x-values for the difference plot.
        difference (list): Difference between y_filtered and y_lowess at common x-values.
    """

    # Interpolate both sets of y-values
    interp_filtered = interp1d(x_filtered, y_filtered, kind='cubic', fill_value='extrapolate')
    interp_lowess = interp1d(x_lowess, y_lowess, kind='cubic', fill_value='extrapolate')

    # Create a common set of x-values within overlapping range
    min_common = max(min(x_filtered), min(x_lowess))
    max_common = min(max(x_filtered), max(x_lowess))
    x_common = np.linspace(min_common, max_common, num=10000)

    # Interpolate y-values for the common x-values
    y_filtered_interp = interp_filtered(x_common)
    y_lowess_interp = interp_lowess(x_common)

    # Compute the difference
    difference = y_filtered_interp - y_lowess_interp
    #plot_data(np.array(x_common), np.array(difference), title='Difference')
    #print(x_common)
    #print(difference)
    
    return x_common.tolist(), difference.tolist()

def perform_fft(x, y):
    """
    Performs Fast Fourier Transform (FFT) on interpolated data using a rectangular window (no window).

    Parameters:
    - x: A list or array of x-values (assumed to be uniformly spaced).
    - y: A list or array of y-values corresponding to the x-values.

    Returns:
    - xf: The frequency components.
    - yf: The amplitude spectrum of the FFT.
    """
    N = len(x)
    T = x[1] - x[0]  # Sample spacing

    # Remove the DC component (mean value)
    y_detrended = y - np.mean(y)
    
    # Perform FFT
    yf = fft(y_detrended)
    
    # Zero out the 0 frequency component
    yf[0] = 0
    
    xf = fftfreq(N, T)[:N//2]  # Only positive frequencies

    # Return frequency and amplitude
    return xf, 2.0/N * np.abs(yf[:N//2])

def full_analysis_lowess(x, y, frac_values=[0.3, 0.35, 0.4]):
    x_trim, y_trim = trim_data(x, y)
    x_trim, y_trim = remove_duplicates(x_trim, y_trim)

    # Colors for different frac values
    colors = cm.nipy_spectral(np.linspace(0, 1, len(frac_values)))

    # Prepare data for plot_data function
    lowess_x_data = []
    lowess_y_data = []
    labels = ['Original Data'] + [f'LOWESS frac={frac}' for frac in frac_values]
    plot_colors = ['red'] + list(colors)

    # Original data (first in the list)
    lowess_x_data.append(np.array(x_trim))
    lowess_y_data.append(np.array(y_trim))

    # LOWESS trends for different frac values
    for frac in frac_values:
        lowess_x, lowess_y = lowess_smoothing(x_trim, y_trim, frac=frac)
        lowess_x_data.append(lowess_x)
        lowess_y_data.append(lowess_y)

    # Plot using plot_data
    '''plot_data(np.array(lowess_x_data).T, np.array(lowess_y_data).T, 
              labels=labels, colors=plot_colors, 
              x_label='Magnetic Field (Tesla)', 
              y_label='Frequency (Hz)', 
              title='Trimmed Data with LOWESS Trend - Multiple Fracs',
              legend_title='Trend Type')
    '''
    #select = int(input("Which one you prefer: "))
    lowess_x = lowess_x_data[-1].tolist()    #lowess_x_data[select-2].tolist()
    lowess_y = lowess_y_data[-1].tolist()    #lowess_y_data[select-2].tolist()

    x_data = np.column_stack((x_trim, lowess_x))
    y_data = np.column_stack((y_trim, lowess_y))

    # Labels and colors for the lines
    labels = ['Dataset 1', 'Dataset 2']
    colors = ['blue', 'red']

# Plotting the data
    '''plot_data(x_data, y_data, labels=labels, colors=colors, 
          x_label='X-axis', y_label='Y-axis', 
          title='Plot of Dataset 1 and Dataset 2', 
          legend_title='Datasets')
'''
    #cutoff = float(input("Enter the cut-off value from left side for x: "))
    cutoff = 10
    x_filtered, y_filtered = filter_bot(x_trim, y_trim, cutoff)
    x_lowess, y_lowess = filter_bot(lowess_x, lowess_y, cutoff)

    #cutoff = float(input("Enter the cut-off value from right side for x: "))
    #x_filtered, y_filtered = filter_top(x_filtered, y_filtered, cutoff)
    #x_lowess, y_lowess = filter_top(x_lowess, y_lowess, cutoff)

    x_data = np.column_stack((x_filtered, x_lowess))
    y_data = np.column_stack((y_filtered, y_lowess))

    # Labels and colors for the lines
    labels = ['Dataset 1', 'Dataset 2']
    colors = ['blue', 'red']

# Plotting the data
    '''plot_data(x_data, y_data, labels=labels, colors=colors, 
          x_label='X-axis', y_label='Y-axis', 
          title='Plot of Dataset 1 and Dataset 2', 
          legend_title='Datasets')
    '''
    x_common, difference = compute_difference(x_filtered, y_filtered, x_lowess, y_lowess)
    #plot_data(np.array(x_common), np.array(difference))
    x_reverted = [1 / i for i in x_common]
    y_reverted = difference 

    #plot_data(np.array(x_reverted), np.array(y_reverted))

    interp = interp1d(x_reverted, y_reverted, kind='cubic', fill_value='extrapolate')
    x_int = np.linspace(min(x_reverted), max(x_reverted), num=1000)
    y_int = interp(x_int)

    xf, yf = perform_fft(x_int, y_int)
    '''plt.plot(xf, yf, marker='o', linestyle='-')

# Add labels and title
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Simple Plot of xf vs yf')

# Display the plot
    plt.grid(True)
    plt.show()'''
    print('Done!')
    return xf, yf


def full_analysis_polynomial(x, y, degree=3):
    df = pd.DataFrame(x, columns=['Column1'])
    df.to_csv('lol1.csv', index=False)
    df = pd.DataFrame(y, columns=['Column1'])
    df.to_csv('lol2.csv', index=False)
    x_trim, y_trim = trim_data(x, y)
    x_trim, y_trim = remove_duplicates(x_trim, y_trim)
    df = pd.DataFrame(x_trim, columns=['Column1'])
    df.to_csv('extra1.csv', index=False)
    df = pd.DataFrame(y_trim, columns=['Column1'])
    df.to_csv('extra2.csv', index=False)

    plt.plot(x_trim, y_trim, label='data')
    plt.xlabel('Field')
    plt.ylabel('Frequency')
    plt.title('Initial data')
    plt.show()

    #cutoff = float(input("Enter cutoff Field: "))
    cutoff = 5
    x_filtered, y_filtered = filter_bot(x_trim, y_trim, cutoff)
    x_poly, y_poly = polynomial_fit(x_filtered, y_filtered, degree)
    df = pd.DataFrame(x_filtered, columns=['Column1'])
    df.to_csv('output1.csv', index=False)
    df = pd.DataFrame(y_filtered, columns=['Column1'])
    df.to_csv('output2.csv', index=False)
    

    interp = interp1d(x_filtered, y_filtered, kind='cubic', fill_value='extrapolate')
    x_filtered = np.linspace(min(x_filtered), max(x_filtered), num=5000)
    y_filtered = interp(x_filtered)

    polynomial, coefficients = polynomial_fit(x_filtered, y_filtered, degree)
    x_poly = x_filtered
    y_poly = polynomial(x_poly)
    df = pd.DataFrame(x_poly, columns=['Column1'])
    df.to_csv('output3.csv', index=False)
    df = pd.DataFrame(y_poly, columns=['Column1'])
    df.to_csv('output4.csv', index=False)

    '''plt.plot(x_filtered, y_filtered, label='Filtered Data')
    plt.plot(x_poly, y_poly, label='Polynomial Fit')
    plt.xlabel('Field')
    plt.ylabel('Frequency')
    plt.title('Filtered Data and Polynomial Fit')
    plt.show()
'''
    x_common, difference = compute_difference(x_filtered, y_filtered, x_poly, y_poly)
    x_reverted = [1 / i for i in x_common]
    y_reverted = difference 

    interp = interp1d(x_reverted, y_reverted, kind='cubic', fill_value='extrapolate')
    x_int = np.linspace(min(x_reverted), max(x_reverted), num=5000)
    y_int = interp(x_int)
    df = pd.DataFrame(x_int, columns=['Column1'])
    df.to_csv('output5.csv', index=False)
    df = pd.DataFrame(y_int, columns=['Column1'])
    df.to_csv('output6.csv', index=False)

    xf, yf = perform_fft(x_int, y_int)

    print('Done!')
    return xf, yf, x_poly, y_filtered, x_poly, y_poly, x_int, y_int

def pad_list(lst, length):
    return lst + [np.nan] * (length - len(lst))

def main():
    # Assume a typical screen size as fallback
    screen_width, screen_height = 1920, 1080
    global scaled_width, scaled_height
    scaled_width, scaled_height = scale_dimensions(screen_width, screen_height)

    # Angles and temperatures data
    angles = np.array([-28, -21, -14, -7, 0, 7, 14, 21, 24.5, 28, 31.5, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147])
    temperatures = [0.5, 0.7, 1.0, 1.6, 5, 7, 10, 14, 20, 35]

    # Filepaths and labels
    name = 'YbTi3Sb4-1'
    ad_filepath = 'ad.csv'
    td_filepath = 'td.csv'
    output_dir = '/root/research/publications/YbTi3Bi4-1/fft_angles/'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data_ad = load_data(ad_filepath)

    # Extract x and y data
    x_ad, y_ad = extract_columns(data_ad)
    
    
     # Run full analysis on temperature data
    x, y, x_filtered, y_filtered, x_poly, y_poly, x_int, y_int = full_analysis_polynomial(x_ad[:, 9].tolist(), y_ad[:, 9].tolist())
    '''for num, item in enumerate(angles):
        x, y = full_analysis_polynomial(x_ad[:, num].tolist(), y_ad[:, num].tolist())
        with open(f'{output_dir}{item}.txt', 'w') as f:
            for c1, c2 in zip(x, y):
                f.write(f'{c1}\t{c2}\n')
    '''

if __name__ == "__main__":
    main()
