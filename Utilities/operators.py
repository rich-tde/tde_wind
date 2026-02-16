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

from matplotlib import cm
sys.path.append('/Users/paolamartire/shocks')

from Utilities.isalice import isalice
alice, plot = isalice()

import numpy as np
import matplotlib.pyplot as plt
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

def draw_line(x_arr, params, what):
    """ Draw a line in the x-y plane with slope tg(alpha).
    Parameters
    ----------
    x_arr: array.
        x coordinates of the points where you want to draw the line.
    params: array or float.
        parameters of the line. If what == 'line', params is the angle alpha. 
        If what == 'powerlaw', params is an array with the constant and the exponent of the power law.
    Returns
    -------
    y_arr: array.
        y coordinates of the points where you want to draw the line.
    """
    if what == 'line':
        alpha = params
        y_arr = np.tan(alpha) * x_arr
    if what == 'powerlaw':
        const, alpha = params
        y_arr = const * np.power(x_arr, alpha)
    return y_arr

def to_spherical_coordinate(x, y, z, r_frame = 'math'):
    """ Transform the components of a vector from cartesian to spherical coordinates 
    lat in [0, pi] with North pole at 0, orbital plane at pi/2
    if r_frame == 'math': long in [0, 2pi] with direction of positive x at 0 and y at pi/2 (as usual)
    if r_frame == 'us': long in [-pi, pi] clockwise with direction of positive x at 0 and y at -pi/2.
    """
    # Accept both scalars and arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z) 
    r = np.sqrt(x**2 + y**2 + z**2)
    if np.logical_and(x==0, np.logical_and(y==0, z==0)).any():
        lat = np.pi/2
        long = 0
    else:
        lat = np.arccos(z/r) # in [0, pi]
        long = np.arctan2(y, x) # in [-pi, pi]. 
        if r_frame == 'math':
            long = np.where(long < 0, long + 2*np.pi, long) # in [0, 2pi] counterclockwise with direction of positive x at 0 and y at pi/2 (as usual)
        if r_frame == 'us':
            long = -long # in [-pi, pi] clockwise with direction of positive x at 0 and y at -pi/2. 
    return r, lat, long

def to_spherical_components(vec_x, vec_y, vec_z, x, y, z):
    """ Transform the components of a vector from cartesian to spherical coordinates."""
    _, lat, long = to_spherical_coordinate(x, y, z, r_frame = 'math')
    # Accept both scalars and arrays
    lat_arr = np.asarray(lat)
    long_arr = np.asarray(long)
    if np.logical_and(x==0, np.logical_and(y==0, z==0)).any():
        vec_r = np.sqrt(vec_x**2 + vec_y**2 + vec_z**2)
        vec_theta = 0
        vec_phi = 0

    else:
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

def choose_sections(X, Y, Z, choice):
    R_cyl = np.sqrt(X**2 + Y**2)
    if choice in ['left_right_z', 'in_out_z', 'left_right_in_out_z']:
        if choice == 'left_right_in_out_z':
            alpha_pole = np.arcsin(2/3)
        else:
            alpha_pole = np.pi/6 
        slope = np.tan(alpha_pole)  
        cond_Npole = np.logical_and(np.abs(Z) > slope *  R_cyl, Z > 0)
        cond_Spole = np.logical_and(np.abs(Z) > slope *  R_cyl, Z < 0)
        north = {'cond': cond_Npole, 'label': r'north pole', 'color': 'dodgerblue', 'line': 'dotted'}
        south = {'cond': cond_Spole, 'label': r'south pole', 'color': 'deepskyblue', 'line': 'dotted'}

    if choice == 'all':
        cond_all = np.abs(X) != 1  # all True
        all = {'cond': cond_all, 'label': r'all', 'color': 'leftviolet', 'line': 'solid'}
        sec = {'all': all}

    if choice == 'left_right_z':
        cond_left = np.logical_and(X < 0, np.abs(Z) <= slope *  R_cyl)
        cond_right = np.logical_and(X >= 0, np.abs(Z) < slope * R_cyl)
        left = {'cond': cond_left, 'label': r'left', 'color': 'r', 'line': 'solid'}
        right = {'cond': cond_right, 'label': r'right', 'color': 'sandybrown', 'line': 'dashed'}
        sec = {'left': left, 'right': right, 'north': north, 'south': south}
    
    if choice == 'in_out_z': 
        cond_in = np.logical_and(Y > 0, np.abs(Z) <= slope * R_cyl)
        cond_out = np.logical_and(Y <= 0, np.abs(Z) <= slope *  R_cyl)
        ins = {'cond': cond_in, 'label': r'in', 'color': 'r', 'line': 'solid'}
        out = {'cond': cond_out, 'label': r'out', 'color': 'sandybrown', 'line': 'dashed'}
        sec = {'in': ins, 'out': out, 'north': north, 'south': south}

    if choice == 'left_right_in_out_z':
        cond_left_in = np.logical_and(X < 0, np.logical_and(Y >= 0, np.abs(Z) <= slope *  R_cyl))
        cond_right_in = np.logical_and(X >= 0, np.logical_and(Y >= 0, np.abs(Z) <= slope * R_cyl))
        cond_left_out = np.logical_and(X < 0, np.logical_and(Y < 0, np.abs(Z) <= slope *  R_cyl))
        cond_right_out = np.logical_and(X >= 0, np.logical_and(Y < 0, np.abs(Z) <= slope * R_cyl))
        left_in = {'cond': cond_left_in, 'label': r'left in', 'color': 'r', 'line': 'solid'}
        right_in = {'cond': cond_right_in, 'label': r'right in', 'color': 'sandybrown', 'line': 'dashed'}
        left_out = {'cond': cond_left_out, 'label': r'left out', 'color': 'forestgreen', 'line': 'solid'}
        right_out = {'cond': cond_right_out, 'label': r'right out', 'color': 'yellowgreen', 'line': 'dashed'}
        sec = {'left_in': left_in, 'right_in': right_in, 'left_out': left_out, 'right_out': right_out, 'north': north, 'south': south}
    
    if choice == 'thirties': 
        cond_right_030 = np.logical_and(X >= 0, np.abs(Z) < np.tan(np.pi/6) * R_cyl)
        cond_right_3060 = np.logical_and(X >= 0, np.logical_and(np.abs(Z) >= np.tan(np.pi/6) * R_cyl, np.abs(Z) < np.tan(np.pi/3) * R_cyl))
        cond_right_6090 = np.logical_and(X >= 0, np.abs(Z) >= np.tan(np.pi/3) * R_cyl)
        cond_left_030 = np.logical_and(X < 0, np.abs(Z) < np.tan(np.pi/6) * R_cyl)
        cond_left_3060 = np.logical_and(X < 0, np.logical_and(np.abs(Z) >= np.tan(np.pi/6) * R_cyl, np.abs(Z) < np.tan(np.pi/3) * R_cyl))
        cond_left_6090 = np.logical_and(X < 0, np.abs(Z) >= np.tan(np.pi/3) * R_cyl)
        
        right_030 = {'cond': cond_right_030, 'label': r'$X>0, |Z|<\tan(\pi/6)R$', 'color': 'sandybrown', 'line': 'solid'}
        right_3060 = {'cond': cond_right_3060, 'label': r'$X>0, \tan(\pi/6)R \le |Z| < \tan(\pi/3)R$', 'color': 'firebrick', 'line': 'solid'}
        right_6090 = {'cond': cond_right_6090, 'label': r'$X>0, |Z| \ge \tan(\pi/3)R$', 'color': 'orchid', 'line': 'solid'}
        left_030 = {'cond': cond_left_030, 'label': r'$X<0, |Z|<\tan(\pi/6)R$', 'color': 'yellowgreen', 'line': 'dashed'}
        left_3060 = {'cond': cond_left_3060, 'label': r'$X<0, \tan(\pi/6)R \le |Z| < \tan(\pi/3)R$', 'color': 'forestgreen', 'line': 'dashed'}
        left_6090 = {'cond': cond_left_6090, 'label': r'$X<0, |Z| \ge \tan(\pi/3)R$', 'color': 'deepskyblue', 'line': 'dashed'}
        sec = {'right_030': right_030, 'right_3060': right_3060, 'right_6090': right_6090, 'left_030': left_030, 'left_3060': left_3060, 'left_6090': left_6090}
    
    if choice == 'tenths': 
        cm = plt.get_cmap('tab20')       
        ncolors = cm.N
        sec = {}
        step = 10
        for i, alpha in enumerate(np.arange(0, 180, step)):
            slope = np.tan(alpha * np.pi/180) 
            slope_next = np.tan((alpha + step) * np.pi/180)
            if alpha < 90:
                cond = np.logical_and(X >= 0, np.logical_and(np.abs(Z) >= slope * R_cyl, np.abs(Z) < slope_next * R_cyl))
                sec[f'right_{alpha}-{alpha + step}'] = {'cond': cond, 'label': f'{alpha}-{alpha + step}', 'line': 'solid', 'color': cm(i % ncolors)}
            else: 
                cond = np.logical_and(X < 0, np.logical_and(np.abs(Z) >= np.abs(slope_next) * R_cyl, np.abs(Z) < np.abs(slope) * R_cyl))
                sec[f'left_{alpha}-{alpha + step}'] = {'cond': cond, 'label': f'{alpha}-{alpha +step}', 'line': 'dashed', 'color': cm(i % ncolors)}

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
    all_idx_obs = np.arange(len(x_obs))
    if choice == 'left_right_z' or choice == 'in_out_z' or choice == 'arch' or choice == 'left_right_in_out_z' or choice == 'all' or choice == 'tenths':
        indices_sorted = []
        sections_ph = choose_sections(x_obs, y_obs, z_obs, choice = choice)
        label_obs = []
        colors_obs = []
        lines_obs = []
        for key in sections_ph.keys(): 
            cond_single = sections_ph[key]['cond'] 
            label_obs.append(sections_ph[key]['label'])
            colors_obs.append(sections_ph[key]['color'])
            lines_obs.append(sections_ph[key]['line'])
        if choice == 'left_right_z' or choice == 'in_out_z':
            if choice == 'left_right_z':
                first_key = 'right'
                second_key = 'left'
            else:
                first_key = 'out'
                second_key = 'in'
            x_obs_right = x_obs[sections_ph[first_key]['cond']]
            x_obs_north = x_obs[sections_ph['north']['cond']]
            x_obs_south = x_obs[sections_ph['south']['cond']]
            if len(x_obs_right) != len(x_obs_north):
                print('Adjusting observers number')
                y_obs_north = y_obs[sections_ph['north']['cond']]
                z_obs_north = z_obs[sections_ph['north']['cond']]
                indices_north = all_idx_obs[sections_ph['north']['cond']]
                # find distances from the pole and all the one who has the maximum distance
                distances = np.sqrt(x_obs_north**2 + y_obs_north**2 + (z_obs_north - 1)**2)
                indices_to_change = np.where(np.isclose(distances, np.max(distances)))[0]
                indices_to_change_left = indices_to_change[x_obs_north[indices_to_change] < 0 if choice == 'left_right_z' else y_obs_north[indices_to_change] > 0]
                indices_to_change_left = indices_north[indices_to_change_left]
                indices_to_change_left = indices_to_change_left[::2]
                sections_ph['north']['cond'][indices_to_change_left] = False
                sections_ph[second_key]['cond'][indices_to_change_left] = True
                indices_to_change_right = indices_to_change[x_obs_north[indices_to_change] >= 0 if choice == 'left_right_z' else y_obs_north[indices_to_change] <= 0]
                indices_to_change_right = indices_north[indices_to_change_right]
                indices_to_change_right = indices_to_change_right[::2]
                sections_ph['north']['cond'][indices_to_change_right] = False
                sections_ph[first_key]['cond'][indices_to_change_right] = True
                # same for south
                y_obs_south = y_obs[sections_ph['south']['cond']]
                z_obs_south = z_obs[sections_ph['south']['cond']]
                indices_south = all_idx_obs[sections_ph['south']['cond']]
                # find distances from the pole and all the one who has the maximum distance
                distances = np.sqrt(x_obs_south**2 + y_obs_south**2 + (z_obs_south + 1)**2)
                indices_to_change = np.where(np.isclose(distances, np.max(distances)))[0]
                indices_to_change_left = indices_to_change[x_obs_south[indices_to_change] < 0 if choice == 'left_right_z' else y_obs_south[indices_to_change] > 0]
                indices_to_change_left = indices_south[indices_to_change_left]
                indices_to_change_left = indices_to_change_left[::2]
                sections_ph['south']['cond'][indices_to_change_left] = False
                sections_ph[second_key]['cond'][indices_to_change_left] = True
                indices_to_change_right = indices_to_change[x_obs_south[indices_to_change] >= 0 if choice == 'left_right_z' else y_obs_south[indices_to_change] <= 0]
                indices_to_change_right = indices_south[indices_to_change_right]
                indices_to_change_right = indices_to_change_right[::2]
                sections_ph['south']['cond'][indices_to_change_right] = False
                sections_ph[first_key]['cond'][indices_to_change_right] = True
        
        if choice == 'left_right_in_out_z': # here the problem are the observers with y = 0
            y_obs_right_in = y_obs[sections_ph['right_in']['cond']]
            y_obs_right_out = y_obs[sections_ph['right_out']['cond']]
            if len(y_obs_right_in) != len(y_obs_right_out):
                print('Adjusting observers number right')
                indices_in = all_idx_obs[sections_ph['right_in']['cond']]
                zero_idx = np.where(y_obs_right_in < 1e-10)[0]
                indices_to_change = indices_in[zero_idx]
                indices_to_change = indices_to_change[::2]
                sections_ph['right_in']['cond'][indices_to_change] = False
                sections_ph['right_out']['cond'][indices_to_change] = True
            # same with left
            y_obs_left_in = y_obs[sections_ph['left_in']['cond']]
            y_obs_left_out = y_obs[sections_ph['left_out']['cond']]
            if len(y_obs_left_in) != len(y_obs_left_out):
                print('Adjusting observers number left')
                indices_in = all_idx_obs[sections_ph['left_in']['cond']]
                zero_idx = np.where(y_obs_left_in < 1e-10)[0]
                indices_to_change = indices_in[zero_idx]
                indices_to_change = indices_to_change[::2]
                sections_ph['left_in']['cond'][indices_to_change] = False
                sections_ph['left_out']['cond'][indices_to_change] = True

        for key in sections_ph.keys(): 
            cond_single = sections_ph[key]['cond']
            indices_sorted.append(all_idx_obs[cond_single])
    
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
        # colors_obs = ['r', 'firebrick', 'plum', 'leftviolet', 'dosgerblue', 'deepskyblue', 'palegreen', 'r', 'dodgerblue', 'b'] # colors for \pm z
        colors_obs = ['r', 'firebrick', 'plum', 'leftviolet', 'dosgerblue', 'deepskyblue', 'yellowgreen', 'sandybrown', 'r', 'b']

        lines_obs = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
        # label_obs = ['x', 'y', 'z', 'xz', '-xz']
        # colors_obs = ['k', 'plum', 'dosgerblue', 'r', 'b']

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
        colors_obs = ['dodgerblue', 'plum', 'r', 'magenta', 'dosgerblue', 'deepskyblue']
        lines_obs = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']
    
    # if choice == 'equal_right_right_z': # 3D cartesian axis
    #     wanted_obs_x = [(1,0,0), (-1,0,0)]  
    #     wanted_obs_z = [(0,0,1), (0,0,-1)] 

    #     tree_obs = KDTree(observers_xyz.T) # shape is N,3
    #     _, indices_sorted_x = tree_obs.query(np.array(wanted_obs_x), k = 64) # shape: (len(wanted_obs),k)
    #     x_obs_nox, y_obs_nox, z_obs_nox, all_idx_obs_nox = x_obs[~indices_sorted_x.flatten()], y_obs[~indices_sorted_x.flatten()], z_obs[~indices_sorted_x.flatten()], all_idx_obs[~indices_sorted_x.flatten()]
    #     tree_obs_nox = KDTree(np.array([x_obs_nox, y_obs_nox, z_obs_nox]).T)
    #     print(len(x_obs_nox))
    #     _, indices_sorted_z_temp = tree_obs_nox.query(np.array(wanted_obs_z), k = 32) # shape: (len(wanted_obs),k)
    #     indices_sorted_z = all_idx_obs_nox[indices_sorted_z_temp.flatten()]
    #     indices_sorted = indices_sorted_x #[indices_sorted_x, indices_sorted_z]
    #     label_obs = ['x+', 'x-', 'z+', 'z-']
    #     colors_obs = ['dodgerblue', 'r', 'dosgerblue', 'deepskyblue']
    #     lines_obs = ['solid', 'solid', 'solid', 'solid']
 
    if choice == 'quadrants ': # 8 3d-quadrants 
        # Cartesian view    
        indices1 = all_idx_obs[np.logical_and(z_obs>=0, np.logical_and(x_obs >= 0, y_obs >= 0))]
        indices2 = all_idx_obs[np.logical_and(z_obs>=0, np.logical_and(x_obs < 0, y_obs >= 0))]
        indices3 = all_idx_obs[np.logical_and(z_obs>=0, np.logical_and(x_obs < 0, y_obs < 0))]
        indices4 = all_idx_obs[np.logical_and(z_obs>=0, np.logical_and(x_obs >= 0, y_obs < 0))]
        indices5 = all_idx_obs[np.logical_and(z_obs<0, np.logical_and(x_obs >= 0, y_obs >= 0))]
        indices6 = all_idx_obs[np.logical_and(z_obs<0, np.logical_and(x_obs < 0, y_obs >= 0))]
        indices7 = all_idx_obs[np.logical_and(z_obs<0, np.logical_and(x_obs < 0, y_obs < 0))]
        indices8 = all_idx_obs[np.logical_and(z_obs<0, np.logical_and(x_obs >= 0, y_obs < 0))]
        indices_sorted = [indices1, indices2, indices3, indices4, indices5, indices6, indices7, indices8]
        label_obs = ['+x+y+z', '-x+y+z', '-x-y+z', '+x-y+z',
                    '+x+y-z', '-x+y-z', '-x-y-z', '+x-y-z',]
        colors_obs = plt.cm.rainbow(np.linspace(0, 1, len(indices_sorted)))

    if choice == 'chunky_axis': # centered on the cartesian axes
        indices1 = all_idx_obs[np.logical_and(np.abs(z_obs) < np.abs(x_obs), np.logical_and(x_obs < 0, np.abs(y_obs) < np.abs(x_obs)))]
        indices2 = all_idx_obs[np.logical_and(np.abs(z_obs) < np.abs(x_obs), np.logical_and(x_obs >= 0, np.abs(y_obs) < x_obs))]
        indices3 = all_idx_obs[np.logical_and(np.abs(z_obs) < np.abs(y_obs), np.logical_and(y_obs < 0, np.abs(y_obs) > np.abs(x_obs)))]
        indices4 = all_idx_obs[np.logical_and(np.abs(z_obs) < np.abs(y_obs), np.logical_and(y_obs >= 0, y_obs > np.abs(x_obs)))]
        
        indices5 = all_idx_obs[np.logical_and(z_obs<0, np.logical_and(np.abs(z_obs) > np.abs(y_obs), np.abs(z_obs) > np.abs(x_obs)))]
        indices6 = all_idx_obs[np.logical_and(z_obs>=0, np.logical_and(z_obs > np.abs(y_obs), z_obs > np.abs(x_obs)))]

        indices_sorted = [indices1, indices2, indices3, indices4, indices5, indices6]#, indices7, indices8]
        colors_obs = ['r', 'sandybrown', 'magenta', 'plum', 'deepskyblue', 'dosgerblue']
        label_obs = ['-x','+x', '-y', '+y', 'z-', 'z+']
        lines_obs = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']
    
    if choice == '':
        indices_sorted = [np.arange(len(x_obs))]
        label_obs = ['']
        colors_obs = ['leftviolet']
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
        # x_line = np.arange(-4, 4, dtype=complex)
        # for a, alpha in enumerate(np.arange(0, 180, 10)):
        #     line = draw_line(x_line, alpha*np.pi/180, 'line')
        #     ax2.plot(x_line, line, c = 'k', ls = 'dashed')
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
    snap = 109

    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    # data = make_tree(path, snap, energy = True)
    # cut = data.Den > 1e-19
    # X, Y, Z, vol, vx, vy, vz= \
    #     make_slices([data.X, data.Y, data.Z, data.Vol, data.VX, data.VY, data.VZ], cut)
    
    # check radial velocity
    XYZ = [[1, 0, 1]]
    X, Y, Z = np.array(XYZ).T
    r, lat, lon = to_spherical_coordinate(X, Y, Z) 
    print('Position in spherical coord:', r, lat, lon)
    V_XYZ = [[0, 0, -1]] 
    vx, vy, vz = np.array(V_XYZ).T
    v_rad, _, _ = to_spherical_components(vx, vy, vz, X, Y, Z)
    print(v_rad)
    plt.figure()
    plt.scatter(X, Y)
    plt.quiver(X, Y, vx, vy, color='k', angles='xy', scale_units='xy', scale=1)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


    