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

def area_spherical_zone(r, theta):
    area = 4 * np.pi * r**2 * np.cos(theta) 
    return area

def area_spherical_cal(r, theta):
    area = 2 * np.pi * r**2 * (1 - np.cos(theta))
    return area

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
    # angles are defined considering alpha=0 at the orbital plane and alpha increasing towards the North pole.
    R_cyl = np.sqrt(X**2 + Y**2)
    if choice in ['left_right_z', 'in_out_z', 'left_right_in_out_z', 'funnel', '3d_arch', 'split_stream']:
        if choice == 'left_right_in_out_z':
            alpha_pole = np.arcsin(2/3)
        elif choice == 'funnel' or choice == '3d_arch':
            alpha_pole = np.pi/3
        elif choice == 'split_stream':
            alpha_pole = 7*np.pi/18
        else:
            alpha_pole = np.pi/6 
        slope = np.tan(alpha_pole)  
        cond_Npole = np.logical_and(np.abs(Z) > slope *  R_cyl, Z > 0)
        cond_Spole = np.logical_and(np.abs(Z) > slope *  R_cyl, Z < 0)
        north = {'cond': cond_Npole, 'label': r'North pole', 'color': '#56cfe1', 'line': 'solid', 'marker': 'X'}
        south = {'cond': cond_Spole, 'label': r'South pole', 'color': 'cornflowerblue', 'line': 'dashed', 'marker': 'X'}

    if choice == 'all':
        cond_all = np.abs(X) != 1  # all True
        all = {'cond': cond_all, 'label': r'all', 'color': 'darkviolet', 'line': 'solid'}
        sec = {'all': all}

    if choice == 'chunky_axes': # modulo rotation, you treat all the axis to be the same
        overture = np.tan(np.pi/12)  # overture single healpix obs = 4pi/192. If you want 4, you have to split by 4, so np/12
        alpha = np.pi/2-overture # since you measure from z
        slope_pole = np.tan(alpha) 
        R_yz = np.sqrt(Y**2 + Z**2)
        R_xz = np.sqrt(X**2 + Z**2)
        cond_Npole = np.logical_and(np.abs(Z) > slope_pole *  R_cyl, Z > 0)
        cond_Spole = np.logical_and(np.abs(Z) > slope_pole *  R_cyl, Z < 0)
        north = {'cond': cond_Npole, 'label': r'+$\hat{z}$', 'color': 'xkcd:sky blue', 'line': 'dotted'}
        south = {'cond': cond_Spole, 'label': r'-$\hat{z}$', 'color': 'cornflowerblue', 'line': 'dotted'}
        cond_xplus =  np.logical_and(np.abs(X) > slope_pole *  R_yz, X > 0)
        cond_xminus = np.logical_and(np.abs(X) > slope_pole *  R_yz, X < 0)
        cond_yplus = np.logical_and(np.abs(Y) > slope_pole *  R_xz, Y > 0)
        cond_yminus = np.logical_and(np.abs(Y) > slope_pole *  R_xz, Y < 0)
        xplus = {'cond': cond_xplus, 'label': r'+$\hat{x}$', 'color': 'xkcd:apple', 'line': 'solid'}
        xminus = {'cond': cond_xminus, 'label': r'-$\hat{x}$', 'color': 'forestgreen', 'line': 'dashed'}
        yplus = {'cond': cond_yplus, 'label': r'+$\hat{y}$', 'color': 'C1', 'line': 'solid'}
        yminus = {'cond': cond_yminus, 'label': r'-$\hat{y}$', 'color': 'xkcd:bubble gum pink', 'line': 'dashed'}
        sec = {'xplus': xplus, 'xminus': xminus, 'yplus': yplus, 'yminus': yminus, 'north': north, 'south': south}

    if choice == 'left_right_z' or choice == 'funnel':
        cond_left = np.logical_and(X < 0, np.abs(Z) <= slope *  R_cyl)
        cond_right = np.logical_and(X >= 0, np.abs(Z) < slope * R_cyl)
        left = {'cond': cond_left, 'label': r'Stream side', 'color': 'xkcd:bubble gum pink', 'line': 'solid'}
        right = {'cond': cond_right, 'label': r'Pericentre side', 'color': 'xkcd:apple', 'line': 'dashed'}
        sec = {'left': left, 'right': right, 'north': north, 'south': south}
    
    if choice == '3d_arch':  
        slope_op = np.tan(np.pi/6)
        cond_left_op = np.logical_and(X < 0, np.abs(Z) <= slope_op *  R_cyl)
        cond_left_ml = np.logical_and(X < 0, np.logical_and(np.abs(Z) <= slope *  R_cyl, np.abs(Z) > slope_op *  R_cyl))
        cond_right_op = np.logical_and(X >= 0, np.abs(Z) < slope_op * R_cyl)
        cond_right_ml = np.logical_and(X >= 0, np.logical_and(np.abs(Z) < slope * R_cyl, np.abs(Z) >= slope_op * R_cyl))
        left_op = {'cond': cond_left_op, 'label': r'Stream side (orb.pl.)', 'color': 'darkviolet', 'line': 'solid'}
        left_ml = {'cond': cond_left_ml, 'label': r'Stream side (mid. lat.)', 'color': 'xkcd:bubble gum pink', 'line': 'solid'}
        right_op = {'cond': cond_right_op, 'label': r'Pericentre side (orb.pl.)', 'color': 'forestgreen', 'line': 'dashed'}
        right_ml = {'cond': cond_right_ml, 'label': r'Pericentre side (mid.lat.)', 'color': 'xkcd:apple', 'line': 'dashed'}
        sec = {'left_op': left_op, 'left_ml': left_ml, 'right_op': right_op, 'right_ml': right_ml, 'north': north, 'south': south}
    
    if choice == 'split_stream': 
        slope_opd = np.tan(np.pi/18)
        slope_opu = np.tan(2*np.pi/9)
        cond_left_op = np.logical_and(X < 0, np.abs(Z) <= slope_opd *  R_cyl)
        cond_left_mld = np.logical_and(X < 0, np.logical_and(np.abs(Z) <= slope_opu *  R_cyl, np.abs(Z) > slope_opd *  R_cyl))
        cond_left_mlu = np.logical_and(X < 0, np.logical_and(np.abs(Z) <= slope *  R_cyl, np.abs(Z) > slope_opu *  R_cyl))
        cond_right = np.logical_and(X >= 0, np.abs(Z) <= slope * R_cyl)
        left_op = {'cond': cond_left_op, 'label': 'Eccentric flow side', 'color': '#ffc2d1', 'line': 'solid', 'marker': 'H'} # r'Stream side $\theta\in[4\pi/9,\pi/2]$'
        left_mld = {'cond': cond_left_mld, 'label': 'Middle stream side', 'color': '#ff499e', 'line': 'solid', 'marker': 'o'} # r'Stream side $\theta\in[5\pi/18, 4\pi/9]$'
        left_mlu = {'cond': cond_left_mlu, 'label': 'High stream side', 'color': '#6e44ff', 'line': 'solid', 'marker': 'p'} # r'Stream side $\theta\in[\pi/9, 5\pi/18]$'
        right = {'cond': cond_right, 'label': r'Pericentre side', 'color': '#b5e48c', 'line': 'solid', 'marker': 's'} 
        sec = {'left_op': left_op, 'left_mld': left_mld, 'left_mlu': left_mlu, 'right': right, 'north': north, 'south': south}
    
    if choice == 'in_out_z': 
        cond_in = np.logical_and(Y > 0, np.abs(Z) <= slope * R_cyl)
        cond_out = np.logical_and(Y <= 0, np.abs(Z) <= slope *  R_cyl)
        ins = {'cond': cond_in, 'label': r'$y>0$', 'color': 'xkcd:bubble gum pink', 'line': 'solid'}
        out = {'cond': cond_out, 'label': r'$y<0$', 'color': 'xkcd:apple', 'line': 'dashed'}
        sec = {'in': ins, 'out': out, 'north': north, 'south': south}

    if choice == 'left_right_in_out_z':
        cond_left_in = np.logical_and(X < 0, np.logical_and(Y >= 0, np.abs(Z) <= slope *  R_cyl))
        cond_right_in = np.logical_and(X >= 0, np.logical_and(Y >= 0, np.abs(Z) <= slope * R_cyl))
        cond_left_out = np.logical_and(X < 0, np.logical_and(Y < 0, np.abs(Z) <= slope *  R_cyl))
        cond_right_out = np.logical_and(X >= 0, np.logical_and(Y < 0, np.abs(Z) <= slope * R_cyl))
        right_in = {'cond': cond_right_in, 'label': r'Pericentre in', 'color': 'forestgreen', 'line': 'solid'}
        right_out = {'cond': cond_right_out, 'label': r'Pericentre out', 'color': 'xkcd:apple', 'line': 'solid'}
        left_in = {'cond': cond_left_in, 'label': r'Stream in', 'color': 'r', 'line': 'dashed'}
        left_out = {'cond': cond_left_out, 'label': r'Stream out', 'color': 'xkcd:bubble gum pink' , 'line': 'dashed'}
        sec = {'left_in': left_in, 'right_in': right_in, 'left_out': left_out, 'right_out': right_out, 'north': north, 'south': south}
    
    if choice == 'tenths': 
        cm = plt.get_cmap('tab20')       
        ncolors = cm.N
        sec = {}
        step = 10
        for i, alpha in enumerate(np.arange(0, 180, step)):
            slope = np.tan(alpha * np.pi/180) if np.abs(alpha) > 1e-5 else 0
            slope_next = np.tan((alpha + step) * np.pi/180) if np.abs(alpha + step - 180) > 1e-5 else 0
            if alpha < 90: 
                cond = np.logical_and(X >= 0, np.logical_and(np.abs(Z) >= slope * R_cyl, np.abs(Z) < slope_next * R_cyl))
                sec[f'{alpha}-{alpha + step}'] = {'cond': cond, 'label': f'{alpha}-{alpha + step}', 'line': 'solid', 'color': cm(i % ncolors)}
            else: 
                cond = np.logical_and(X <= 0, np.logical_and(np.abs(Z) >= np.abs(slope_next) * R_cyl, np.abs(Z) < np.abs(slope) * R_cyl))
                sec[f'{alpha}-{alpha + step}'] = {'cond': cond, 'label': f'{alpha}-{alpha +step}', 'line': 'dashed', 'color': cm(i % ncolors)}
    
    if choice == 'azimuthal': 
        cm = plt.get_cmap('tab20')       
        theta = np.arctan2(Y, X)  # range [-pi, pi]
        theta[theta<0] += 2 * np.pi  # range [0, 2pi]
        theta_deg = (theta * 180 / np.pi) 

        step = 30
        angles = np.arange(0, 360, step)
        color_sec = cm(np.linspace(0, 1, len(angles)))
        sec = {}
        for i, alpha in enumerate(angles):
            alpha_next = alpha + step
            cond = np.logical_and(theta_deg >= alpha,
                                theta_deg < alpha_next)
            sec[f'{alpha}-{alpha_next}'] = {
                'cond': cond,
                'label': f'{alpha}-{alpha_next}',
                'line': 'solid',
                'color': color_sec[i]
            }

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

    if choice in ['left_right_z', 'in_out_z', 'left_right_in_out_z', 'all', 'tenths', 'chunky_axes', 'azimuthal', 'funnel', '3d_arch', 'split_stream']:
        indices_sorted = []
        sections_ph = choose_sections(x_obs, y_obs, z_obs, choice = choice)

        # print(f"Choice: {choice}")
        # print("Sections:", list(sections_ph.keys()))
        # coverage = np.sum([np.sum(sec['cond']) for sec in sections_ph.values()])
        # print(f"Total coverage: {coverage}/{len(x_obs)}")
        # overlaps = np.sum(np.any([sec['cond'] for sec in sections_ph.values()], axis=0) > 1)
        # print(f"Overlaps: {overlaps}")

        indices_sorted = [all_idx_obs[sections_ph[key]['cond']] for key in sections_ph]
        label_obs = [sections_ph[key]['label'] for key in sections_ph]
        colors_obs = [sections_ph[key]['color'] for key in sections_ph]
        lines_obs = [sections_ph[key]['line'] for key in sections_ph]
        markers_obs = [sections_ph[key].get('marker', None) for key in sections_ph] # get marker if exists, else None

        if choice in ['left_right_z', 'in_out_z']:
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

        # for key in sections_ph.keys(): 
        #     cond_single = sections_ph[key]['cond']
        #     indices_sorted.append(all_idx_obs[cond_single])
        if choice in ['left_right_z', 'in_out_z', 'left_right_in_out_z']:
            indices_sorted = [all_idx_obs[sections_ph[key]['cond']] for key in sections_ph]
    
    if choice == '':
        indices_sorted = [np.arange(len(x_obs))]
        label_obs = ['']
        colors_obs = ['darkviolet']
        lines_obs = ['solid']

    if choice == 'single_axes':
        cart_axis = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        tree = KDTree(observers_xyz.T)
        indices_sorted = []
        for axis in cart_axis:
            _, idx = tree.query([axis], k=1)
            indices_sorted.append(idx[0])
        label_obs = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']
        colors_obs = ['xkcd:apple green',  'C1', 'xkcd:sky blue', 'gold', 'xkcd:bubble gum pink', 'forestgreen']
        lines_obs = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']
    
    if choice == 'arch':
        observers_xyz = np.array(observers_xyz)
        wanted_obs = [[1, 0, 0], [1/np.sqrt(2), 0,  1/np.sqrt(2)], [0, 0, 1], [-1/np.sqrt(2), 0,  1/np.sqrt(2)],[-1, 0, 0]]
        tree = KDTree(observers_xyz.T)
        indices_sorted = []
        for axis in wanted_obs:
            _, idx = tree.query([axis], k=4)
            indices_sorted.append(np.concatenate(idx))
        label_obs = [r'$\hat{x}$', r'$(\hat{x}+\hat{z})/\sqrt{2}$', r'$\hat{z}$', r'$(-\hat{x}+\hat{z})/\sqrt{2}$', r'$-\hat{x}$']
        colors_obs = ['xkcd:apple green',  '#073b4c', 'xkcd:sky blue', '#ffd166', 'xkcd:bubble gum pink']
        lines_obs = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']


    if plot:
        import matplotlib.pyplot as plt
        fig_obs, (ax1_obs, ax2_obs) = plt.subplots(1, 2, figsize=(11, 5))
        for j, idx_list in enumerate(indices_sorted):
            # print(idx_list)
            ax1_obs.scatter(x_obs[idx_list], y_obs[idx_list], s = 20, c = colors_obs[j], label = label_obs[j])
            ax2_obs.scatter(x_obs[idx_list], z_obs[idx_list], s = 20, c = colors_obs[j], label = label_obs[j])
        for ax in [ax1_obs, ax2_obs]:
            ax.set_xlabel(r'$X$')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
        x_line = np.arange(-4, 4, dtype=complex)
        # for a, alpha in enumerate(np.arange(0, 180, 10)):
        #     line = draw_line(x_line, alpha*np.pi/180, 'line')
        #     ax2_obs.plot(x_line, line, c = 'k', ls = 'dashed')
        ax1_obs.set_ylabel(r'$Y$')
        ax2_obs.set_ylabel(r'$Z$')
        plt.suptitle(f'Selected observers', fontsize=15)
        # ax1_obs.legend(fontsize = 12)
        # put the legend outside
        plt.legend(fontsize = 12, loc='upper right', bbox_to_anchor=(1.5, 1), ncol=1)
        plt.tight_layout()
        # plt.show()

    return indices_sorted, label_obs, colors_obs, lines_obs, markers_obs

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
    def __init__(self, sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, time, IE, Rad, Diss, 
                 Eg0 = None, Eg1 = None, Eg2 = None, Eg3 = None, Eg4 = None, Eg5 = None, Eg6 = None, Eg7 = None, Eg8 = None, Eg9 = None):
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
        # self.Entropy = Entropy
        self.time = time
        self.Eg0 = Eg0
        self.Eg1 = Eg1
        self.Eg2 = Eg2
        self.Eg3 = Eg3
        self.Eg4 = Eg4
        self.Eg5 = Eg5
        self.Eg6 = Eg6
        self.Eg7 = Eg7
        self.Eg8 = Eg8
        self.Eg9 = Eg9

def make_tree(filename, snap, MG = False):
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
    Diss = np.load(f'{filename}/Diss_{snap}.npy') # Dissipation rate density [energy/time/volume]
    # Entropy = np.load(f'{filename}/Entropy_{snap}.npy')
    IE = np.load(f'{filename}/IE_{snap}.npy')
    Rad = np.load(f'{filename}/Rad_{snap}.npy')
    # convert from energy/mass to energy density
    IE *= Den  
    Rad *= Den

    if MG:
        Eg0 = np.load(f'{filename}/Eg_0_{snap}.npy')
        Eg0 *= Den
        Eg1 = np.load(f'{filename}/Eg_1_{snap}.npy')
        Eg1 *= Den
        Eg2 = np.load(f'{filename}/Eg_2_{snap}.npy')
        Eg2 *= Den
        Eg3 = np.load(f'{filename}/Eg_3_{snap}.npy')
        Eg3 *= Den
        Eg4 = np.load(f'{filename}/Eg_4_{snap}.npy')
        Eg4 *= Den
        Eg5 = np.load(f'{filename}/Eg_5_{snap}.npy')
        Eg5 *= Den
        Eg6 = np.load(f'{filename}/Eg_6_{snap}.npy')
        Eg6 *= Den
        Eg7 = np.load(f'{filename}/Eg_7_{snap}.npy')
        Eg7 *= Den
        Eg8 = np.load(f'{filename}/Eg_8_{snap}.npy')
        Eg8 *= Den
        Eg9 = np.load(f'{filename}/Eg_9_{snap}.npy')
        Eg9 *= Den
             
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

    if MG:
        data = data_snap(sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, time, IE, Rad, Diss, Eg0, Eg1, Eg2, Eg3, Eg4, Eg5, Eg6, Eg7, Eg8, Eg9)
    else:
        data = data_snap(sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, time, IE, Rad, Diss)
        
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


    