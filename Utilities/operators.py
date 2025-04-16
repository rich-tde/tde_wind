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
import astropy.coordinates as coord
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
import math
import numba
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
    theta_broadcasted = -theta_coord
    return theta_broadcasted, radius

def from_cylindric(theta, r):
    # we expect theta as from the function to_cylindric, i.e. clockwise. 
    # You have to mirror it to get the angle for the usual polar coordinates.
    theta = -theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def to_spherical_components(vec_x, vec_y, vec_z, lat, long):
    """ Transform the components of a vector from cartesian to spherical coordinates.
    NB: you need to pass the latitude and longitude (in radians) because 
    vec_x, vec_y, vec_z are not necessarily postions, thus you can't use them to derive the angless.
    lat (theta) has to be in [0,pi] and long (phi) in [0, 2pi]."""
    vec_r = np.sin(lat) * (vec_x * np.cos(long) + vec_y * np.sin(long)) + vec_z * np.cos(lat)
    vec_theta = np.cos(lat) * (vec_x * np.cos(long) + vec_y * np.sin(long)) - vec_z * np.sin(lat)
    vec_phi = - vec_x * np.sin(long) + vec_y * np.cos(long)
    return vec_r, vec_theta, vec_phi

def J_cart_in_sphere(lat, long):
    matrix = np.array([[np.sin(lat)*np.cos(long), np.cos(lat)*np.cos(long), -np.sin(long)],
                        [np.sin(lat)*np.sin(long), np.cos(lat)*np.sin(long), np.cos(long)],
                        [np.cos(lat), -np.sin(lat), 0]])
    return matrix

def rotate_coordinate(x , y, theta):
    """ Rotate the coordinates of an angle theta."""
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot

def Ryan_sampler(theta_arr):
    """ Function to sample the angle in the orbital plane so that you have more points also at apocenter."""
    # theta_shift = np.pi * np.sin(theta_arr/2)
    theta_shift =  np.pi * np.tanh(2*theta_arr/np.pi) / np.tanh(2)
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

# def sort_list(list_passive, leading_list):
#     """ Sort list_passive based on the order of leading_list. List_passive is a list of arrays."""
#     new_list = []
#     for arr_passive in list_passive:
#         zipped_pairs = zip(leading_list, arr_passive)
#         z = [x for _, x in sorted(zipped_pairs)]
#         new_list.append(np.array(z))
#     return new_list

def sort_list(list_passive, leading_list):
    """Sort list_passive based on the order of leading_list. 
       list_passive is a list of numpy arrays.
    """
    sort_indices = np.argsort(leading_list)  # Get indices that would sort leading_list
    return [arr[sort_indices] for arr in list_passive]  # Apply those indices to each sub-array

def find_ratio(L1, L2):
    """ Find the ratio between the two lists."""
    if type(L1) == list or type(L1) == np.ndarray: 
        L1 = np.array(L1)
        L2 = np.array(L2)
        ratio = np.zeros(len(L1))
        for i in range(len(L1)):
            ratio[i] = max(np.abs(L1[i]), np.abs(L2[i]))/min(np.abs(L1[i]), np.abs(L2[i]))
    else:
        ratio = max(np.abs(L1), np.abs(L2))/min(np.abs(L1), np.abs(L2))
    return ratio

# def median_array(points, window_size=7):
#     n = len(points)
#     half_window = window_size // 2
#     medians = np.copy(points)
#     if half_window != 0:
#         for i in range(half_window, n-half_window): #I don't care of the first/last points
#             window = points[i-half_window:i+window_size]
#             median = np.median(window)
#             medians[i]= median
#     return np.array(medians)

def average_array(values, w, window_size=7):
    """ Compute the weighted average of the values in a window of size window_size."""
    n = len(values)
    half_window = window_size // 2
    averages = np.copy(values) 
    if half_window != 0:
        for i in range(half_window, n-half_window): #I don't care of the first/last points
            window = values[i-half_window:i+window_size]
            distances = np.abs(w[i-half_window:i+window_size]-w[i])
            weights = np.power(distances,2)
            average = np.average(window, weights = weights)
            averages[i]= average
    return np.array(averages)
class data_snap:
    # create a class to be used in make_tree so that it gives just one output.
    def __init__(self, sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, IE = None, Rad =None, Diss = None, Entropy = None):
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
        data = data_snap(sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, IE, Rad, Diss)
    else: 
        data = data_snap(sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T)
    return data

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
    """
    gridded_tocast = np.zeros((len(radii)))
    # check if weights is an integer
    if type(weights) != str:
        gridded_weights = np.zeros((len(radii)))
    R = R.reshape(-1, 1) # Reshaping to 2D array with one column
    tree = KDTree(R) 
    for i in range(len(radii)):
        radius = np.array([radii[i]]).reshape(1, -1) # reshape to match the tree
        if i == 0:
            width = radii[1] - radii[0]
        elif i == len(radii)-1:
            width = radii[-1] - radii[-2]
        else:
            width = (radii[i+1] - radii[i-1])/2
        width *= 2 # make it slightly bigger to smooth things
        # indices = tree.query_ball_point(radius, width) #if KDTree from scipy
        indices = tree.query_radius(radius, width) #if KDTree from sklearn
        indices = np.concatenate(indices)
        # if len(indices) < 2 :
        #     print('small sample of indices in single_branch', flush=True)
        #     sys.stdout.flush()
        #     # gridded_tocast[i] = 0
        # else:    
        indices = [int(idx) for idx in indices]
        if type(weights) != str:
            gridded_tocast[i] = np.sum(tocast[indices] * weights[indices])
            gridded_weights[i] = np.sum(weights[indices])
        else:
            if type(weights) == 'mean':
                gridded_tocast[i] = np.mean(tocast[indices])
            if type(weights) == 'sum':
                gridded_tocast[i] = np.sum(tocast[indices])
    if type(weights) != str:
        gridded_weights += 1e-20 # avoid division by zero
        final_casted = np.divide(gridded_tocast, gridded_weights)
    else:
        final_casted = gridded_tocast
    if keep_track:
        return final_casted, indices
    else:
        return final_casted

def multiple_branch(radii, R, tocast_matrix, weights_matrix, sumORmean_matrix = [], keep_track = False):
    """ Casts quantities down to a smaller size vector.
    Parameters
    ----------
    radii : arr,
        Array of radii we want to cast to.
    R : arr,
        Coordinates' data from simulation to be casted according to radii.
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
        radius = np.array([radii[i]]).reshape(1, -1) # reshape to match the tree
        if i == 0:
            width = radii[1] - radii[0]
        elif i == len(radii)-1:
            width = radii[-1] - radii[-2]
        else:
            width = (radii[i+1] - radii[i-1])/2
        width *= 2 # make it slightly bigger to smooth things
        # indices = tree.query_ball_point(radius, width) #if KDTree from scipy
        indices = tree.query_radius(radius, width) #if KDTree from sklearn
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
            print(sumORmean, flush=True) 
            sys.stdout.flush()
        for j in range(len(radii)):
            indices = indices_foradii[j]
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
            print(gridded_tocast[j])
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