from storage import load_data, extract_columns, save_to_csv, extr_c
from calculator import full_analysis_polynomial
import numpy as np
import matplotlib.pyplot as plt

def scale_dimensions(width, height, width_scale=0.99, height_scale=0.95):
    """Scale the dimensions based on given scaling factors."""
    scaled_width = width * width_scale
    scaled_height = height * height_scale
    return scaled_width, scaled_height


def main():
    # Assume a typical screen size as fallback
    screen_width, screen_height = 1920, 1080
    global scaled_width, scaled_height
    scaled_width, scaled_height = scale_dimensions(screen_width, screen_height)

    # Angles and temperatures data
    angles = np.array([-28, -21, -14, -7, 0, 7, 14, 21, 24.5, 28, 31.5, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147])
    temperatures = [0.5, 0.7, 1.0, 1.6, 5, 7, 10, 14, 20, 35]

    # Filepaths and labels
    name = 'results'
    filepath = 'PracticeData.txt'
    output_dir = f'./data/'+name+'/fft/'

    data = load_data(f'./data/{filepath}')

    x, y = extr_c(data)
    
    x, y = full_analysis_polynomial(x, y)
    '''for num, item in enumerate(angles):
        x, y = full_analysis_polynomial(x_ad[:, num].tolist(), y_ad[:, num].tolist())
        with open(f'{output_dir}{item}.txt', 'w') as f:
            for c1, c2 in zip(x, y):
                f.write(f'{c1}\t{c2}\n')
    '''
    plt.plot(x, y)
    plt.show()
    save_to_csv([x, y], output_dir,f'{name}.csv' )

if __name__ == "__main__":
    main()
