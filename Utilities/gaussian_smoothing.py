import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from Utilities.operators import make_tree

# Function to apply Gaussian smoothing with local bandwidths
def gaussian_smoothing(points, density, tree, dim_cell):
    smoothed_density = np.zeros_like(density)
    for i, point in enumerate(points):
        if i % 50_000 == 0:
            print(i)
        # Estimate local cell size (bandwidth)
        local_bandwidth = dim_cell[i]
        # Query points within 3*local_bandwidth (arbitrary choice, you can adjust)
        indices = tree.query_ball_point(point, 3 * local_bandwidth)
        distances = np.linalg.norm(points[indices] - point, axis=1)
        weights = np.exp(-(distances**2) / (2 * local_bandwidth**2))
        smoothed_density[i] = np.sum(weights * density[indices]) / np.sum(weights)
    return smoothed_density

# Try to do the same thing but something on time
def time_smoothing(times, energy):
    smoothed_energy = np.zeros_like(energy)
    for i, t in enumerate(times):
        if i == 0:
            indices = np.arange(i,i+5)
        elif i == 1:
            indices = np.arange(i-1,i+4)
        elif i == len(times) - 1:
            indices = np.arange(i-5,i)
        elif i == len(times) - 2:
            indices = np.arange(i-4,i+1)
        else:
            indices = np.arange(i-2, i+3)
        distances = np.abs(times[indices] - t)
        local_bandwidth = np.mean(distances)
        weights = np.exp(-(distances**2) / (2 * local_bandwidth**2))
        smoothed_energy[i] = np.sum(weights * energy[indices]) / np.sum(weights)
    return smoothed_energy

def time_average(times, energy):
    smoothed_energy = np.zeros_like(energy)
    for i, t in enumerate(times):
        if i == 0:
            indices = np.arange(i,i+5)
        elif i == 1:
            indices = np.arange(i-1,i+4)
        elif i == len(times) - 1:
            indices = np.arange(i-5,i)
        elif i == len(times) - 2:
            indices = np.arange(i-4,i+1)
        else:
            indices = np.arange(i-2, i+3)
        weights = times[indices]
        smoothed_energy[i] = np.sum(weights * energy[indices]) / np.sum(weights)
    return smoothed_energy

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Utilities.selectors_for_snap import select_snap
    
    # Apply Gaussian smoothing
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = 'HiRes'
    compton = 'Compton'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) #[100,115,164,199,216]
    snaps = [216]
    
    for snap in snaps:
        print(snap)
        path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
        data = make_tree(path, snap, energy = False)
        points = np.array([data.X, data.Y, data.Z])
        points = np.transpose(points)
        density = data.Den
        dim_cell = data.Vol**(1/3)
        tree = data.sim_tree

        smoothed_density = gaussian_smoothing(points, density, tree, dim_cell)
        np.save(f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}/smoothed_Den_{snap}.npy', smoothed_density)

    # path_time = f'/Users/paolamartire/shocks/data/{folder}'
    # tfb = np.loadtxt(f'{path_time}/coloredE_Low_days.txt')[1]
    # energy_ie = np.load(f'{path_time}/coloredE_Low.npy')[1]
    # energy_ie = np.abs(energy_ie.T)
    # smoothed_energy20 = time_smoothing(tfb, energy_ie[20])
    # smoothed_energy80 = time_smoothing(tfb, energy_ie[80])
    # time_average20 = time_average(tfb, energy_ie[20])
    # time_average80 = time_average(tfb, energy_ie[80])

    # # plt.plot(energy_ie[20], c = 'green', label = f'Radius 20')
    # plt.plot(energy_ie[80], c = 'r', label = f'Radius 80')
    # # plt.plot(smoothed_energy20, '--', c = 'deepskyblue', label = 'Radius 20 Smoothed')
    # plt.plot(smoothed_energy80, '--', c = 'orange', label = 'Radius 80 Smooth')
    # # plt.plot(time_average20, '-.', c = 'purple', label = 'Radius 20 Time Average')
    # plt.plot(time_average80, '-.', c = 'brown', label = 'Radius 80 Time Average')
    # plt.yscale('log')
    # plt.ylabel(r'$|$Specific energy$|$')
    # plt.xlabel('Time')
    # plt.legend()   
    # plt.show()