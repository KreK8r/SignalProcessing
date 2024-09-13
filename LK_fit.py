import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model_function(x, A, mc):
    x = np.array(x)  # Ensure x is a NumPy array
    H_min = 5
    H_max = 41.5
    Havg = 2 * H_min * H_max / (H_max + H_min)
    const = 1.62e31
    return A * const * mc * (x / Havg) / np.sinh(const * mc * (x / Havg))

T = [0.5, 0.7, 1.0, 1.6, 5, 7, 10, 14, 20, 35]
Amp = [0.0000121001669110151, 0.0000120642921034255, 0.0000120302019023034, 0.0000118413471223889, 9.32247154968247E-06, 7.72965292227617E-06, 5.62097079527173E-06, 3.63702802046915E-06, 1.82465580407839E-06, 3.05067756018059E-07]


x_data = T
y_data = [a / max(Amp) for a in Amp]

# Initial guess for A and mc
initial_guess = [1, 1e-31]

# Fit the model to the data
params, covariance = curve_fit(model_function, x_data, y_data, p0=initial_guess)

# Use the fitted parameters to generate the fitted curve
fitted_x = np.linspace(min(x_data), max(x_data), 1000)
fitted_y = model_function(fitted_x, *params)

errors = np.sqrt(np.diag(covariance))

# Plotting

plt.scatter(x_data, y_data, label='Data', color='blue')
plt.plot(fitted_x, fitted_y, label='Fitted curve', color='red')
plt.legend()
plt.xlabel('Temperature (T)', fontsize=22)
plt.ylabel('Amplitude (Normalized)', fontsize=22)
plt.title('Fitting frequency picks to Lifshitz Kosevich formula', fontsize=25)
plt.show()


# Display fitted parameters
print("Fitted parameters (A, mc):", params)
print(f"Parameter errors (standard deviations): {errors}")
print(f"Effective mass: {params[1]/(9.11*1e-31)} +- {errors[1]/(9.11*1e-31)}")
print(f"Error is {errors[1]/params[1]*100} %")
