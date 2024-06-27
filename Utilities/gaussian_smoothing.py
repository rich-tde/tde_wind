import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from Utilities.operators import make_tree

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
check = 'HiRes'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
snap = '164'
path = f'TDE/{folder}{check}/{snap}'
data = make_tree(path, snap, is_tde = True, energy = False)
points = np.array([data.X, data.Y, data.Z])
points = np.transpose(points)
density = data.Den
dim_cell = data.Vol**(1/3)
tree = data.sim_tree

# Function to apply Gaussian smoothing with local bandwidths
def gaussian_smoothing(points, density, tree, dim_cell):
    smoothed_density = np.zeros_like(density)
    for i, point in enumerate(points):
        if i % 10_000 == 0:
            print(i)
        # Estimate local cell size (bandwidth)
        local_bandwidth = dim_cell[i]
        # Query points within 3*local_bandwidth (arbitrary choice, you can adjust)
        indices = tree.query_ball_point(point, 3 * local_bandwidth)
        distances = np.linalg.norm(points[indices] - point, axis=1)
        weights = np.exp(-(distances**2) / (2 * local_bandwidth**2))
        smoothed_density[i] = np.sum(weights * density[indices]) / np.sum(weights)
    return smoothed_density

# Apply Gaussian smoothing
smoothed_density = gaussian_smoothing(points, density, tree, dim_cell)
np.save(f'TDE/{folder}{check}/{snap}/smoothed_Den_{snap}.npy', smoothed_density)

