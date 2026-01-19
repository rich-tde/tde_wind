"""
Recurrent operators.
1) Obtain polar coordinates for the orbital plane that go clockwise:
from -pi in -x to second, first, fourth and third (pi) quadrant.
2) Sort one list according to the order of another one.
3) Compute the median of an array.
4) Make a tree from the simulation data.
5) Find the nearest cells.
6) Compute the div/grad for old stuff.
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

from Utilities.isalice import isalice
alice, plot = isalice()

import numpy as np
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
import k3match
import math
import numba
from scipy.interpolate import griddata
import Utilities.prelude

def to_cylindric(x,y):
    radius = np.sqrt(x**2+y**2)
    if np.abs(x.any()) > 1e-5: # numerical version of x.any()!= 0:
        theta_coord = np.arctan2(y,x)
    else:
        if np.abs(y.any()) < 1e-5:
            theta_coord = 0
        elif y.any()>0:
            theta_coord = np.pi/2
        else:
            theta_coord = -np.pi/2
    # theta_coord go from -pi to pi with negative values in the 3rd and 4th quadrant. You want to mirror 
    theta_ourConv = -theta_coord
    return theta_ourConv, radius

def from_cylindric(theta, r):
    # we expect theta as from the function to_cylindric, i.e. clockwise. 
    # You have to mirror it to get the angle for the python polar coordinates.
    theta_fornumpy = -theta
    x = r * np.cos(theta_fornumpy)
    y = r * np.sin(theta_fornumpy)
    return x, y

def format_pi_frac(x, pos): # write colorbar ticks in terms of pi fractions
    frac = x / np.pi
    common = {
        -0.5: r'$-\frac{\pi}{2}$',
        -1/3: r'$-\frac{\pi}{3}$',
        -0.25: r'$-\frac{\pi}{4}$',
        -1/6: r'$-\frac{\pi}{6}$',
        -1/8: r'$-\frac{\pi}{8}$',
        0: r'$0$',
        1/8: r'$\frac{\pi}{8}$',
        1/6: r'$\frac{\pi}{6}$',
        0.25: r'$\frac{\pi}{4}$',
        1/3: r'$\frac{\pi}{3}$',
        0.5: r'$\frac{\pi}{2}$',
    }
    for val, label in common.items():
        if np.isclose(frac, val, atol=1e-3):
            return label
    return r'${0:.2g}\pi$'.format(frac)

def draw_line(x_arr, alpha):
    """ Draw a line in the x-y plane with slope tg(alpha).
    Parameters
    ----------
    x_arr: array.
        x coordinates of the points where you want to draw the line.
    alpha: float.
        Angle in radians of the line you want to draw. 
        You use your own convention for the angle, so it has to be flipped (use -alpha) for numpy
    Returns
    -------
    y_arr: array.
        y coordinates of the points where you want to draw the line.
    """
    y_arr = np.tan(-alpha) * x_arr
    return y_arr

def to_spherical_coordinate(x, y, z):
    """ Transform the components of a vector from cartesian to spherical coordinates with long in [0, 2pi] and lat in [0, pi]."""
    # Accept both scalars and arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arccos(z/r) # in [0, pi]
    long = np.arctan2(y, x) # in [-pi, pi]. 
    long = np.where(long < 0, long + 2*np.pi, long)
    return r, lat, long

def to_spherical_components(vec_x, vec_y, vec_z, x, y, z):
    """ Transform the components of a vector from cartesian to spherical coordinates."""
    _, lat, long = to_spherical_coordinate(x, y, z)
    # Accept both scalars and arrays
    lat_arr = np.asarray(lat)
    long_arr = np.asarray(long)
 
    vec_r = np.sin(lat_arr) * (vec_x * np.cos(long_arr) + vec_y * np.sin(long_arr)) + vec_z * np.cos(lat_arr)
    vec_theta = np.cos(lat_arr) * (vec_x * np.cos(long_arr) + vec_y * np.sin(long_arr)) - vec_z * np.sin(lat_arr)
    vec_phi = - vec_x * np.sin(long_arr) + vec_y * np.cos(long_arr)
    return vec_r, vec_theta, vec_phi

def J_cart_in_sphere(lat, long):
    matrix = np.array([[np.sin(lat)*np.cos(long), np.cos(lat)*np.cos(long), -np.sin(long)],
                        [np.sin(lat)*np.sin(long), np.cos(lat)*np.sin(long), np.cos(long)],
                        [np.cos(lat), -np.sin(lat), 0]])
    return matrix 

def Ryan_sampler(theta_arr):
    """ Function to sample the angle in the orbital plane so that you have more points also at apocenter."""
    # theta_shift = np.pi * np.sin(theta_arr/2)
    theta_shift =  np.pi * np.tanh(0.5*theta_arr/np.pi) / np.tanh(0.5)
    return theta_shift

def find_step(theta_arr, i):
    """ Find the step of the angle array for a given element i."""
    if i == 0:
        step = theta_arr[1] - theta_arr[0]
    elif i == len(theta_arr)-1:
        step = theta_arr[-1] - theta_arr[-2]
    else:
        step = theta_arr[1] - theta_arr[0]
    return step

def choose_sections(X, Y, Z, choice = 'dark_bright_z'):
    R_cyl = np.sqrt(X**2 + Y**2)
    alpha_pole = np.pi/4 #np.pi/2-np.arccos(23/24)
    slope = np.tan(alpha_pole)  
    cond_Npole = np.logical_and(np.abs(Z) >= slope *  R_cyl, Z > 0)
    cond_Spole = np.logical_and(np.abs(Z) >= slope *  R_cyl, Z < 0)
    north = {'cond': cond_Npole, 'label': r'north pole', 'color': 'orange', 'line': 'dotted'}
    south = {'cond': cond_Spole, 'label': r'south pole', 'color': 'sienna', 'line': 'dotted'}

    if choice == 'dark_bright_z_in_out':
        cond_bright_in = np.logical_and(X >= 0, np.logical_and(Y >= 0, np.abs(Z) < slope * R_cyl))
        cond_bright_out = np.logical_and(X >= 0, np.logical_and(Y < 0, np.abs(Z) < slope * R_cyl))
        cond_dark_in = np.logical_and(X < 0, np.logical_and(Y >= 0, np.abs(Z) < slope *  R_cyl))
        cond_dark_out = np.logical_and(X < 0, np.logical_and(Y < 0, np.abs(Z) < slope *  R_cyl))
        bright_in = {'cond': cond_bright_in, 'label': r'right in', 'color': 'deepskyblue', 'line': 'dashed'}
        bright_out = {'cond': cond_bright_out, 'label': r'right out', 'color': 'b', 'line': 'dashed'}
        dark_in = {'cond': cond_dark_in, 'label': r'left in', 'color': 'forestgreen', 'line': 'solid'}
        dark_out = {'cond': cond_dark_out, 'label': r'left out', 'color': 'yellowgreen', 'line': 'solid'}
        sec = {'bright_in': bright_in, 'bright_out': bright_out, 'dark_in': dark_in, 'dark_out': dark_out, 'north': north, 'south': south}
     
    if choice == 'dark_bright_z':
        cond_bright = np.logical_and(X >= 0, np.abs(Z) < slope * R_cyl)
        cond_dark = np.logical_and(X < 0, np.abs(Z) < slope *  R_cyl)
        bright = {'cond': cond_bright, 'label': r'right', 'color': 'deepskyblue', 'line': 'dashed'}
        dark = {'cond': cond_dark, 'label': r'left', 'color': 'forestgreen', 'line': 'solid'}
        sec = {'bright': bright, 'dark': dark, 'north': north, 'south': south}
    
    if choice == 'arch':
        cond_bright_low = np.logical_and(X >= 0, np.abs(Z) < R_cyl)
        cond_bright_high = np.logical_and(X >= 0, np.logical_and(np.abs(Z) >= R_cyl, np.abs(Z) < slope * R_cyl))
        cond_dark_low = np.logical_and(X < 0, np.abs(Z) < R_cyl)
        cond_dark_high = np.logical_and(X < 0, np.logical_and(np.abs(Z) >= R_cyl, np.abs(Z) < slope * R_cyl))
        bright_low = {'cond': cond_bright_low, 'label': r'$X>0, |Z|<R$', 'color': 'deepskyblue', 'line': 'solid'}
        bright_high = {'cond': cond_bright_high, 'label': r'$X>0, R<|Z|<mR$', 'color': 'r', 'line': 'dashed'}
        dark_low = {'cond': cond_dark_low, 'label': r'$X<0, |Z|<R$', 'color': 'forestgreen', 'line': 'solid'}
        dark_high = {'cond': cond_dark_high, 'label': r'$X<0, R<|Z|<mR$', 'color': 'k', 'line': 'dashed'}
        sec = {'bright_low': bright_low, 'bright_high': bright_high, 'dark_low': dark_low, 'dark_high': dark_high, 'north': north, 'south': south}

    return sec
    
def choose_observers(observers_xyz, choice):
    """ Choose observers based on the choice string. 
    Parameters
    ----------
    observers_xyz : np.ndarray
            Array of shape 3xN with the coordinates of the observers.
    choice : str
        String that specifies the choice of observers.
    Returns
    -------
    indices_sorted : list
        List of indices of the chosen observers.
    label_obs : list
        List of labels for the chosen observers.
    colors_obs : list
        List of colors for the chosen observers.
    """
    if len(observers_xyz) != 3:
        raise ValueError("observers_xyz must be a 3xN array.")
    
    x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
    if choice == 'dark_bright_z' or choice == 'arch' or choice == 'dark_bright_z_in_out':
        indices_sorted = []
        all_idx_obs = np.arange(len(observers_xyz.T))
        # wanted_z_obs = [(0,0,1), (0,0,-1)]
        # tree_obs = KDTree(observers_xyz.T) 
        # _, indices_z = tree_obs.query(np.array(wanted_z_obs), k=4) 
        # remaining_idx_obs = all_idx_obs[~np.isin(all_idx_obs, np.concatenate(indices_z))]
        # indices_bright = remaining_idx_obs[x_obs[remaining_idx_obs] > 0]
        # indices_dark = remaining_idx_obs[x_obs[remaining_idx_obs] < 0]
        # indices_sorted.append(indices_bright)
        # indices_sorted.append(indices_dark)
        # indices_sorted.append(indices_z[0])
        # indices_sorted.append(indices_z[1])
        # label_obs = [r'$+\hat{\textbf{x}}$', r'$-\hat{\textbf{x}}$', r'$+\hat{\textbf{z}}$', r'$-\hat{\textbf{z}}$']
        # colors_obs = ['deepskyblue', 'forestgreen', 'orange', 'sienna']
        # lines_obs = ['dashed', 'solid', 'dotted', 'dashed']
        sections_ph = choose_sections(x_obs, y_obs, z_obs, choice = choice)
        label_obs = []
        colors_obs = []
        lines_obs = []
        for key in sections_ph.keys(): 
            cond_single = sections_ph[key]['cond']
            indices_sorted.append(all_idx_obs[cond_single])
            label_obs.append(sections_ph[key]['label'])
            colors_obs.append(sections_ph[key]['color'])
            lines_obs.append(sections_ph[key]['line'])

    # if choice == 'arch': # upper hemisphere
    #     wanted_obs = [(1,0,0), 
    #                 (1/np.sqrt(2), 0, 1/np.sqrt(2)),  
    #                 (0,0,1),
    #                 (-1/np.sqrt(2), 0 , 1/np.sqrt(2)),
    #                 (-1,0,0)]
        # dot_prod = np.dot(wanted_obs, observers_xyz)
        # indices_sorted = np.argmax(dot_prod, axis=1)
        # tree_obs = KDTree(observers_xyz.T) # shape is N,3
        # _, indices_sorted = tree_obs.query(np.array(wanted_obs), k=4) 
        # label_obs = ['x+', '45', 'z+', '135', 'x-']
        # colors_obs = ['k', 'green', 'orange', 'b', 'r']
    
    if choice == 'hemispheres': 
        wanted_obs = [(1,0,0), 
                    (-1,0,0),
                    (0,1,0),  
                    (0,-1,0),
                    (0,0,1),
                    (0,0,-1),
                    (1/np.sqrt(2), 0 , 1/np.sqrt(2)),
                    (1/np.sqrt(2), 0 , -1/np.sqrt(2)),
                    (-1/np.sqrt(2), 0 , 1/np.sqrt(2)),
                    (-1/np.sqrt(2), 0 , -1/np.sqrt(2))] 

        tree_obs = KDTree(observers_xyz.T) # shape is N,3
        _, indices_sorted = tree_obs.query(np.array(wanted_obs), k=4) # shape: (len(wanted_obs),k)
        label_obs = ['x', '-x', 'y', '-y', 'z', '-z', 'xz', 'x-z', '-xz', '-x-z']
        # colors_obs = ['r', 'firebrick', 'plum', 'darkviolet', 'orange', 'sienna', 'palegreen', 'forestgreen', 'dodgerblue', 'b'] # colors for \pm z
        colors_obs = ['r', 'firebrick', 'plum', 'darkviolet', 'orange', 'sienna', 'yellowgreen', 'deepskyblue', 'forestgreen', 'b']

        lines_obs = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
        # label_obs = ['x', 'y', 'z', 'xz', '-xz']
        # colors_obs = ['k', 'plum', 'orange', 'forestgreen', 'b']

    if choice == 'focus_axis': # 3D cartesian axis
        wanted_obs = [(1,0,0), 
                    (0,1,0),  
                    (-1,0,0),
                    (0,-1,0),
                    (0,0,1),
                    (0,0,-1)] 

        tree_obs = KDTree(observers_xyz.T) # shape is N,3
        _, indices_sorted = tree_obs.query(np.array(wanted_obs), k = 4) # shape: (len(wanted_obs),k)
        label_obs = ['x+', 'y+', 'x-', 'y-', 'z+', 'z-']
        colors_obs = ['dodgerblue', 'plum', 'forestgreen', 'magenta', 'orange', 'sienna']
        lines_obs = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']
    
    # if choice == 'dark_bright_z':
    #     indices_sorted = []
    #     all_idx_obs = np.arange(len(observers_xyz.T))
        # wanted_z_obs = [(0,0,1), (0,0,-1)]
        # tree_obs = KDTree(observers_xyz.T) 
        # _, indices_z = tree_obs.query(np.array(wanted_z_obs), k=4) 
        # remaining_idx_obs = all_idx_obs[~np.isin(all_idx_obs, np.concatenate(indices_z))]
        # indices_bright = remaining_idx_obs[x_obs[remaining_idx_obs] > 0]
        # indices_dark = remaining_idx_obs[x_obs[remaining_idx_obs] < 0]
        # indices_sorted.append(indices_bright)
        # indices_sorted.append(indices_dark)
        # indices_sorted.append(indices_z[0])
        # indices_sorted.append(indices_z[1])
        # label_obs = [r'$+\hat{\textbf{x}}$', r'$-\hat{\textbf{x}}$', r'$+\hat{\textbf{z}}$', r'$-\hat{\textbf{z}}$']
        # colors_obs = ['deepskyblue', 'forestgreen', 'orange', 'sienna']
        # lines_obs = ['dashed', 'solid', 'dotted', 'dashed']

    if choice == 'quadrants ': # 8 3d-quadrants 
        # Cartesian view    
        indices1 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(x_obs >= 0, y_obs >= 0))]
        indices2 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(x_obs < 0, y_obs >= 0))]
        indices3 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(x_obs < 0, y_obs < 0))]
        indices4 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(x_obs >= 0, y_obs < 0))]
        indices5 = obs_indices[np.logical_and(z_obs<0, np.logical_and(x_obs >= 0, y_obs >= 0))]
        indices6 = obs_indices[np.logical_and(z_obs<0, np.logical_and(x_obs < 0, y_obs >= 0))]
        indices7 = obs_indices[np.logical_and(z_obs<0, np.logical_and(x_obs < 0, y_obs < 0))]
        indices8 = obs_indices[np.logical_and(z_obs<0, np.logical_and(x_obs >= 0, y_obs < 0))]
        indices_sorted = [indices1, indices2, indices3, indices4, indices5, indices6, indices7, indices8]
        label_obs = ['+x+y+z', '-x+y+z', '-x-y+z', '+x-y+z',
                    '+x+y-z', '-x+y-z', '-x-y-z', '+x-y-z',]
        colors_obs = plt.cm.rainbow(np.linspace(0, 1, len(indices_sorted)))

    if choice == 'chunky_axis': # centered on the cartesian axes
        indices1 = obs_indices[np.logical_and(np.abs(z_obs) < np.abs(x_obs), np.logical_and(x_obs < 0, np.abs(y_obs) < np.abs(x_obs)))]
        indices2 = obs_indices[np.logical_and(np.abs(z_obs) < np.abs(x_obs), np.logical_and(x_obs >= 0, np.abs(y_obs) < x_obs))]
        indices3 = obs_indices[np.logical_and(np.abs(z_obs) < np.abs(y_obs), np.logical_and(y_obs < 0, np.abs(y_obs) > np.abs(x_obs)))]
        indices4 = obs_indices[np.logical_and(np.abs(z_obs) < np.abs(y_obs), np.logical_and(y_obs >= 0, y_obs > np.abs(x_obs)))]
        
        indices5 = obs_indices[np.logical_and(z_obs<0, np.logical_and(np.abs(z_obs) > np.abs(y_obs), np.abs(z_obs) > np.abs(x_obs)))]
        indices6 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(z_obs > np.abs(y_obs), z_obs > np.abs(x_obs)))]

        indices_sorted = [indices1, indices2, indices3, indices4, indices5, indices6]#, indices7, indices8]
        colors_obs = plt.cm.rainbow(np.linspace(0, 1, len(indices_sorted)))
        label_obs = ['-x','+x', '-y', '+y', 'z-', 'z+']
        lines_obs = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']
    
    if choice == '':
        indices_sorted = [np.arange(len(x_obs))]
        label_obs = ['']
        colors_obs = ['darkviolet']
        lines_obs = ['solid']

    if plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        for j, idx_list in enumerate(indices_sorted):
            ax1.scatter(x_obs[idx_list], y_obs[idx_list], s = 50, c = colors_obs[j], label = label_obs[j])
            ax2.scatter(x_obs[idx_list], z_obs[idx_list], s = 50, c = colors_obs[j], label = label_obs[j])
        for ax in [ax1, ax2]:
            ax.set_xlabel(r'$X$')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
        ax1.set_ylabel(r'$Y$')
        ax2.set_ylabel(r'$Z$')
        plt.suptitle(f'Selected observers', fontsize=15)
        # ax1.legend(fontsize = 12)
        # put the legend outside
        plt.legend(fontsize = 12, loc='upper right', bbox_to_anchor=(1.5, 1), ncol=1)
        plt.tight_layout()
        plt.show()

    return indices_sorted, label_obs, colors_obs, lines_obs

def sort_list(list_passive, leading_list, unique = False):
    """Sort list_passive based on the order of leading_list. 
       list_passive is a list of numpy arrays.
       NB: If you want to sort also leading_list, you have to add it, as final element, to list_passive.
    """
    if unique == True:
        _, unique_indices = np.unique(leading_list, return_index=True)
        leading_list = leading_list[unique_indices]  # Keep only unique values
        list_passive = [arr[unique_indices] for arr in list_passive]  # Apply unique indices to each sub-array
    sort_indices = np.argsort(leading_list)  # Get indices that would sort leading_list
    return [arr[sort_indices] for arr in list_passive]  # Apply those indices to each sub-array

def find_ratio(L1, L2):
    """ Find the ratio between the two lists."""
    if type(L1) == list or type(L1) == np.ndarray: 
        L1 = np.array(L1)
        L2 = np.array(L2)
        n = min(len(L1), len(L2))
        ratio = np.zeros(n)
        for i in range(n):
            ratio[i] = max(np.abs(L1[i]), np.abs(L2[i]))/min(np.abs(L1[i]), np.abs(L2[i]))
    else:
        ratio = max(np.abs(L1), np.abs(L2))/min(np.abs(L1), np.abs(L2))
    return ratio

class data_snap:
    # create a class to be used in make_tree so that it gives just one output.
    def __init__(self, sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, time, IE = None, Rad =None, Diss = None, Entropy = None):
        self.sim_tree = sim_tree
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Vol = Vol
        self.VX = VX
        self.VY = VY
        self.VZ = VZ
        self.Mass = Mass
        self.Den = Den
        self.Press = P
        self.Temp = T
        self.IE = IE
        self.Rad = Rad
        self.Diss = Diss
        self.Entropy = Entropy
        self.time = time

def make_tree(filename, snap, energy = False):
    """ Load data from simulation and build the tree. """
    X = np.load(f'{filename}/CMx_{snap}.npy')
    Y = np.load(f'{filename}/CMy_{snap}.npy')
    Z = np.load(f'{filename}/CMz_{snap}.npy')
    Vol = np.load(f'{filename}/Vol_{snap}.npy')
    VX = np.load(f'{filename}/Vx_{snap}.npy')
    VY = np.load(f'{filename}/Vy_{snap}.npy')
    VZ = np.load(f'{filename}/Vz_{snap}.npy')
    Den = np.load(f'{filename}/Den_{snap}.npy')
    Mass = np.load(f'{filename}/Mass_{snap}.npy')
    time = np.loadtxt(f'{filename}/tfb_{snap}.txt') 
    if energy:
        IE = np.load(f'{filename}/IE_{snap}.npy')
        Rad = np.load(f'{filename}/Rad_{snap}.npy')
        # convert from energy/mass to energy density
        IE *= Den  
        Rad *= Den
        Diss = np.load(f'{filename}/Diss_{snap}.npy') # Dissipation rate density [energy/time/volume]
        # Entropy = np.load(f'{filename}/Entropy_{snap}.npy')
             
    P = np.load(f'{filename}/P_{snap}.npy')
    T = np.load(f'{filename}/T_{snap}.npy')
    if all(T) == 0:
        print('all T=0, bro. Compute by myself!')
        T = P/Den
    Star = np.load(f'{filename}/Star_{snap}.npy')
    for i,rho in enumerate(Den):
        cell_star = Star[i]
        if ((1-cell_star) > 1e-3):
            rho = 0 

    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of shape (number_points, 3)
    sim_tree = KDTree(sim_value)#, leaf_size=50) #avoid leaf_size

    if energy:
        data = data_snap(sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, time, IE, Rad, Diss)
    else: 
        data = data_snap(sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, time)
    return data

def compute_curl(X, Y, Z, Vol, VX, VY, VZ):
    R_vec = np.transpose(np.array([X, Y, Z]))
    tree = KDTree(R_vec, leaf_size=50) 
    _, idx = tree.query(R_vec, k=20)  # idx shape: (N,k)
    idx = np.unique(idx)  
    f_inter_input = np.array([X[idx], Y[idx], Z[idx]]).T
    vx_i = VX[idx]
    vy_i = VY[idx]
    vz_i = VZ[idx]
    dx = 0.5 * (Vol[idx])**(1/3)
    Xp_dx = np.transpose(np.array([X+dx, Y, Z]))
    Xm_dx = np.transpose(np.array([X-dx, Y, Z]))
    Yp_dx = np.transpose(np.array([X, Y+dx, Z]))
    Ym_dx = np.transpose(np.array([X, Y-dx, Z]))
    Zp_dx = np.transpose(np.array([X, Y, Z+dx]))
    Zm_dx = np.transpose(np.array([X, Y, Z-dx]))
    vx_p = griddata(f_inter_input, vx_i, Xp_dx, method='linear')
    vx_m = griddata(f_inter_input, vx_i, Xm_dx, method='linear')
    vy_p = griddata(f_inter_input, vy_i, Xp_dx, method='linear')
    vy_m = griddata(f_inter_input, vy_i, Xm_dx, method='linear')
    vz_p = griddata(f_inter_input, vz_i, Xp_dx, method='linear')
    vz_m = griddata(f_inter_input, vz_i, Xm_dx, method='linear')
    # dvx_dx = np.nan_to_num((vx_p - vx_m)/(2*dx))
    dvy_dx = np.nan_to_num((vy_p - vy_m)/(2*dx))
    dvz_dx = np.nan_to_num((vz_p - vz_m)/(2*dx))
    print('Computed dv/dx', flush=True)
    vx_p = griddata(f_inter_input, vx_i, Yp_dx, method='linear')
    vx_m = griddata(f_inter_input, vx_i, Ym_dx, method='linear')
    vy_p = griddata(f_inter_input, vy_i, Yp_dx, method='linear')
    vy_m = griddata(f_inter_input, vy_i, Ym_dx, method='linear')
    vz_p = griddata(f_inter_input, vz_i, Yp_dx, method='linear')
    vz_m = griddata(f_inter_input, vz_i, Ym_dx, method='linear')
    dvx_dy = np.nan_to_num((vx_p - vx_m)/(2*dx))
    # dvy_dy = np.nan_to_num((vy_p - vy_m)/(2*dx))
    dvz_dy = np.nan_to_num((vz_p - vz_m)/(2*dx))
    print('Computed dv/dy', flush=True)
    vx_p = griddata(f_inter_input, vx_i, Zp_dx, method='linear')
    vx_m = griddata(f_inter_input, vx_i, Zm_dx, method='linear')
    vy_p = griddata(f_inter_input, vy_i, Zp_dx, method='linear')
    vy_m = griddata(f_inter_input, vy_i, Zm_dx, method='linear')
    vz_p = griddata(f_inter_input, vz_i, Zp_dx, method='linear')
    vz_m = griddata(f_inter_input, vz_i, Zm_dx, method='linear')
    dvx_dz = np.nan_to_num((vx_p - vx_m)/(2*dx))
    dvy_dz = np.nan_to_num((vy_p - vy_m)/(2*dx))
    # dvz_dz = np.nan_to_num((vz_p - vz_m)/(2*dx))
    print('Computed dv/dz', flush=True)
    # Compute curl for all particles
    curl_vec = np.zeros((len(X),3))
    curl_vec[:,0] = dvz_dy - dvy_dz  # curl_x
    curl_vec[:,1] = dvx_dz - dvz_dx  # curl_y
    curl_vec[:,2] = dvy_dx - dvx_dy  # curl_z

    return curl_vec


def single_branch(radii, R, tocast, weights, keep_track = False):
    """ Casts a quantity down to a smaller size vector
    Parameters
    ----------
    radii : arr,
        Array of radii/angles we want to cast to.
    R : arr,
        Coordinates' data from simulation to be casted.
    tocast: arr,
        Simulation data to cast corresponing to R.
    weights: arr,
        Weights to use in the casting. If it's an integer: no weights are used.
    keep_track: bool,
        If True, returns the indices of the points used in the casting.
    Returns
    -------
    final_casted: arr
        Casted down version of tocast
    all_indices: arr (optional)
        Indices used in casting if keep_track=True
    """
    gridded_tocast = np.zeros((len(radii)))
    all_indices = []  # For keep_track functionality

    use_weights = not isinstance(weights, str)
    if use_weights:
        gridded_weights = np.zeros((len(radii)))

    R = R.reshape(-1, 1) # Reshaping to 2D array with one column
    tree = KDTree(R) 

    for i in range(len(radii)):
        radius = np.array([[radii[i]]]) # reshape to match the tree
        if i == 0:
            width = radii[1] - radii[i]
        elif i == len(radii)-1:
            width = radii[i] - radii[i-1]
        else:
            width = (radii[i+1] - radii[i-1]) / 2
        # width *= 1.5 # make it slightly bigger to smooth things
        # indices = tree.query_ball_point(radius, width) #if KDTree from scipy
        indices = tree.query_radius(radius, width) #if KDTree from sklearn
        indices = np.concatenate(indices)
        if keep_track:
            all_indices.append(indices.astype(int))
        # Handle case where no points are found
        if len(indices) == 0:
            print(f'No points found for radius {radii[i]}', flush=True)
            gridded_tocast[i] = 0
            if use_weights:
                gridded_weights[i] = 0
            continue
        indices = indices.astype(int)

        if use_weights:
            gridded_tocast[i] = np.sum(tocast[indices] * weights[indices])
            gridded_weights[i] = np.sum(weights[indices])
        else:
            if weights == 'mean':
                gridded_tocast[i] = np.mean(tocast[indices])
            elif weights == 'sum':
                gridded_tocast[i] = np.sum(tocast[indices])
    if use_weights:
        gridded_weights += 1e-20 # avoid division by zero
        final_casted = np.divide(gridded_tocast, gridded_weights)
    else:
        final_casted = gridded_tocast

    if keep_track:
        return final_casted, all_indices
    else:
        return final_casted

def multiple_branch(radii, R, dim_leaf, tocast_matrix, weights_matrix, sumORmean_matrix = [], keep_track = False):
    """ Casts quantities down to a smaller size vector.
    Parameters
    ----------
    radii : arr,
        Array of radii we want to cast to.
    R : arr,
        Coordinates' data from simulation to be casted according to radii.
    dim_leaf arr,
        max distance to search in for query_radius.
    tocast_matrix: Narr,
        Simulation data (more than one) corresponing to R.
    weights: Narr,
        Weights (more than one) to use in the casting. If it's an integer: no weights are used.
    Returns
    -------
    final_casted: Narr
        Casted down version of tocast
    """
    casted_array = []
    indices_foradii = []
    R = R.reshape(-1, 1) # Reshaping to 2D array with one column
    tree = KDTree(R) 
    for i in range(len(radii)):
        # R_len_1 = np.ones(len(R))
        radius = np.array([radii[i]]).reshape(1, -1) # reshape to match the tree
        if i == 0:
            width = radii[1] - radii[0]
        elif i == len(radii)-1:
            width = radii[-1] - radii[-2]
        else:
            width = (radii[i+1] - radii[i-1])/2
        width *= 2 # make it slightly bigger to smooth things
        # indices = tree.query_ball_point(radius, width) #if KDTree from scipy
        indices = tree.query_radius(radius, dim_leaf[i]) #if KDTree from sklearn
        # _, indices, dist = k3match.cartesian(radii[i], 1, 1, R, R_len_1, R_len_1, 1e7)
        # indices = indices[dist < dim_leaf]
        # indices_foradii.append(indices)
        indices_foradii.append(np.concatenate(indices))

    for i, tocast in enumerate(tocast_matrix):
        gridded_tocast = np.zeros((len(radii)))
        weights = weights_matrix[i]
        # check if weights is an integer
        if type(weights) != int:
            print('Weighting', flush=True) 
            sys.stdout.flush()
            gridded_weights = np.zeros((len(radii)))
        else:
            sumORmean = sumORmean_matrix[i]
            # print(sumORmean, flush=True) 
            # sys.stdout.flush()
        for j in range(len(radii)):
            indices = indices_foradii[j]
            if indices.size == 0:   
                gridded_tocast[j] = 0
                if keep_track:
                    indices_foradii[j] = []
                continue
            # if len(indices) < 2 :
            #     print('small sample of indices in multiple_branch', flush=True)
            #     sys.stdout.flush()
            #     gridded_tocast[i] = 0
            #     if keep_track:
            #         cells_used.append([])
            # else:    
            indices = [int(idx) for idx in indices]
            if type(weights) != int:
                gridded_tocast[j] = np.sum(tocast[indices] * weights[indices])
                gridded_weights[j] = np.sum(weights[indices])
            else:
                if sumORmean == 'mean':
                    gridded_tocast[j] = np.mean(tocast[indices])
                if sumORmean == 'sum':
                    gridded_tocast[j] = np.sum(tocast[indices])
        if type(weights) != int:
            gridded_weights += 1e-20 # avoid division by zero
            final_casted = np.divide(gridded_tocast, gridded_weights)
        else:
            final_casted = gridded_tocast
        casted_array.append(final_casted)

    if keep_track:
        return casted_array, indices_foradii
    
    return casted_array

def calc_deriv(x, y):
    """ calculate the derivative of y with respect to x using the point before and after"""
    dy = y[2:] - y[:-2]
    dx = x[2:] - x[:-2]
    deriv_mid = dy / dx
    # add the first and last point
    deriv = np.zeros(len(x))
    deriv[1:-1] = deriv_mid
    deriv[0] = (y[1] - y[0]) / (x[1] - x[0])
    deriv[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return deriv

def select_near_1d(sim_tree, X, Y, Z, point, delta, coord):
    """ Find (within the tree) the nearest cell along one direction to the one chosen. 
     Parameters
     -----------
     sim_tree: tree.
        Simualation points. 
     X, Y, Z: arrays.
        Points coordinates.
     point: array.
        Chosen point.
     delta: float.
        Step you do from your chosen point. It has to be positive!
     coord: str.
        coordinates along which you want to move.
     Returns:
     -----------
     idx: int.
        Tree index of the queried nearest cell.
    """
    x_point = point[0]
    y_point = point[1]
    z_point = point[2]

    # move in the choosen direction till you query in the tree a point different from the starting one.
    # (i.e. its distance from the starting point is not 0)
    k = 0.6
    distance = 0
    while np.abs(distance)<1e-5:
        if coord == 'x':
                new_point = [x_point + k * delta, y_point, z_point]
        elif coord == 'y':
                new_point = [x_point, y_point +  k * delta, z_point]
        elif coord == 'z':
                new_point = [x_point, y_point, z_point +  k * delta]
        _, idx  = sim_tree.query(new_point)
        check_point = np.array([X[idx], Y[idx], Z[idx]])
        distance = math.dist(point, check_point)
        k += 0.1
        # check if you're going too long with these iterations. Exit from the loop (and you'll discard that point)
        if k > 100:
            print(f'lots of iterations for div/grad in {coord} for point {point}. Skip')
            distance = 1
    
    return idx

def select_neighbours(sim_tree, X, Y, Z, point, delta, select):
    """ Find the previous (next) points in the 3 cartesian directions.
    Parameters
    -----------
    sim_tree, X, Y, Z, point, delta: as select_near_1d.
    select: str.
        If 'before' --> you search the previous points respectively in x,y,z direction
        otherwise --> you search the next points respectively in x,y,z direction
    Returns
    -----------
    idxx, idyy, idzz: int.
        (Tree) indexes of the previous (next) points searched.
    """
    # Choose if you want to find the prevoius or the next one
    # Possible improvement: use different delta for x,y,z
    if select == 'before':
        step = - delta
    elif select == 'after':
        step = delta

    idxx = select_near_1d(sim_tree, X, Y, Z, point, step, coord = 'x')
    idxy = select_near_1d(sim_tree, X, Y, Z, point, step, coord = 'y')
    idxz = select_near_1d(sim_tree, X, Y, Z, point, step, coord = 'z')
    
    return idxx, idxy, idxz


def calc_div(sim_tree, X, Y, Z, fx_tree, fy_tree, fz_tree, point, delta):
    """ Compute the divergence.
    Parameters
    -----------
    sim_tree, X, Y, Z, point, delta: as select_near_1d.
    fx_tree, fy_tree, fz_tree: arrays of len=len(X).
            Components of the quantity f of the tree.
    kind_info: str.
            Tell if points is given in cartesian coordinates ('point') or if you have its tree index ('idx')
    Returns
    -----------
    div_f: float.
            Divergence of f.
    """
    # Find tree indexes of the previous and next neighbours in all the directions.
    prex, prey, prez = select_neighbours(sim_tree, X, Y, Z, point, delta, 'before')
    postx, posty, postz = select_neighbours(sim_tree, X, Y, Z, point, delta, 'after')

    # Find the coordinate and the values of f in these points.
    pre_xcoord = X[prex]
    fpre_x = fx_tree[prex]
    post_xcoord = X[postx]
    fpost_x = fx_tree[postx]

    pre_ycoord = Y[prey]
    fpre_y = fy_tree[prey]
    post_ycoord = Y[posty]
    fpost_y = fy_tree[posty]

    pre_zcoord = Z[prez]
    fpre_z = fz_tree[prez]
    post_zcoord = Z[postz]
    fpost_z = fz_tree[postz]

    delta_fx = (fpost_x-fpre_x) / (post_xcoord-pre_xcoord)
    delta_fy = (fpost_y-fpre_y)/ (post_ycoord-pre_ycoord)
    delta_fz = (fpost_z-fpre_z) / (post_zcoord-pre_zcoord)

    div_f = delta_fx + delta_fy + delta_fz
    return div_f

    
def calc_grad(sim_tree, X, Y, Z, f_tree, point, delta):
    """ Compute the gradient.
    Parameters
    -----------
    As the ones of calc_div except
    f_tree: array of len=len(X).
            Quantity f of the tree.
    Returns
    -----------
    grad: array.
        Gradient of f.
    """
    # Find tree indexes of the previous and next neighbours in all the directions.
    prex, prey, prez = select_neighbours(sim_tree, X, Y, Z, point, delta, 'before')
    postx, posty, postz = select_neighbours(sim_tree, X, Y, Z, point, delta, 'after')

    # Find the coordinate and the values of f in these points.
    pre_xcoord = X[prex]
    fpre_x = f_tree[prex]
    post_xcoord = X[postx]
    fpost_x = f_tree[postx]

    pre_ycoord = Y[prey]
    fpre_y = f_tree[prey]
    post_ycoord = Y[posty]
    fpost_y = f_tree[posty]

    pre_zcoord = Z[prez]
    fpre_z = f_tree[prez]
    post_zcoord = Z[postz]
    fpost_z = f_tree[postz]

    delta_fx = (fpost_x-fpre_x) / (post_xcoord-pre_xcoord)
    delta_fy = (fpost_y-fpre_y)/ (post_ycoord-pre_ycoord)
    delta_fz = (fpost_z-fpre_z) / (post_zcoord-pre_zcoord)

    grad = np.array([delta_fx, delta_fy, delta_fz])
    return grad

def calc_multiple_grad(sim_tree, X, Y, Z, f_array, point, delta):
    """ Find gradients of all the quantities you need."""
    # Find tree indexes of the previous and next neighbours in all the directions.
    prex, prey, prez = select_neighbours(sim_tree, X, Y, Z, point, delta, 'before')
    postx, posty, postz = select_neighbours(sim_tree, X, Y, Z, point, delta, 'after')

    # Find the coordinates in these points.
    pre_xcoord = X[prex]
    post_xcoord = X[postx]
    pre_ycoord = Y[prey]
    post_ycoord = Y[posty]
    pre_zcoord = Z[prez]
    post_zcoord = Z[postz]

    # Find the values of f in these points.
    gradients = []
    for f_tree in f_array:
        fpre_x = f_tree[prex]
        fpost_x = f_tree[postx]

        fpre_y = f_tree[prey]
        fpost_y = f_tree[posty]

        fpre_z = f_tree[prez]
        fpost_z = f_tree[postz]

        delta_fx = (fpost_x-fpre_x) / (post_xcoord-pre_xcoord)
        delta_fy = (fpost_y-fpre_y)/ (post_ycoord-pre_ycoord)
        delta_fz = (fpost_z-fpre_z) / (post_zcoord-pre_zcoord)

        grad = np.array([delta_fx, delta_fy, delta_fz])
        gradients.append(grad)

    return gradients

if __name__ == '__main__':
    from Utilities.sections import make_slices
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = 'HiResNewAMR'
    compton = 'Compton'
    snap = 0
    print('Computing curl for snapshot', snap, flush=True)

    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    data = make_tree(path, snap, energy = True)
    cut = data.Den > 1e-19
    X, Y, Z, vol, vx, vy, vz= \
        make_slices([data.X, data.Y, data.Z, data.Vol, data.VX, data.VY, data.VZ], cut)
    curl = compute_curl(X, Y, Z, vol, vx, vy, vz)
    np.save(f'{path}/curl_{snap}.npy', curl)


    