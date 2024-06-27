import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

# Parameters of the parabola
a = 0.25
b = -1

# Define the differential arclength function
def ds_dx(x):
    return np.sqrt(1 + (2 * a * x)**2)

# Compute the total arclength from x = -x_max to x = x_max
x_max = 10  # You can adjust this as needed
S_max, _ = scipy.integrate.quad(ds_dx, -x_max, x_max)

# Generate N uniformly spaced arclengths
N = 100
S_values = np.linspace(0, S_max, N)

# Inverse function to find x for a given arclength S
def arclength_to_x(S):
    def f(x):
        result, _ = scipy.integrate.quad(ds_dx, -x_max, x)
        return result - S

    return scipy.optimize.root_scalar(f, bracket=[-x_max, x_max]).root

# Compute x values corresponding to the arclengths
x_values = np.array([arclength_to_x(S) for S in S_values])
y_values = a * x_values**2 + b

# Rotate the parabola so it looks like the TDE
x_values,y_values = -y_values, x_values

# Plot and make aspect ratio 1, so that the uniform spacing is visually evident
plt.plot(x_values,y_values,'.')
plt.xlim(-19,1)
plt.ylim(-10,10)
plt.show()
