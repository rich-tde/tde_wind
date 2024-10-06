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

def sort_list(list_passive, leading_list):
    """ Sort list_passive based on the order of leading_list. If list_passive is a list of arrays."""
    new_list = []
    for arr_passive in list_passive:
        zipped_pairs = zip(leading_list, arr_passive)
        z = [x for _, x in sorted(zipped_pairs)]
        new_list.append(np.array(z))
    return new_list

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
    def __init__(self, sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, IE = None, Rad =None, Entropy = None):
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
        # self.Diss = Diss
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
        # Diss = np.load(f'{filename}/Diss_{snap}.npy')
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
    sim_tree = KDTree(sim_value) 

    if energy:
        data = data_snap(sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T, IE, Rad)
    else: 
        data = data_snap(sim_tree, X, Y, Z, Vol, VX, VY, VZ, Mass, Den, P, T)
    return data

def single_branch(radii, xaxis, R, tocast, weights, keep_track = False):
    """ Casts a quantity down to a smaller size vector
    Parameters
    ----------
    radii : arr,
        Array of radii/angles we want to cast to.
    xaxis : str,
        'angles' or 'radii'.
    R : arr,
        Coordinates' data from simulation to be casted.
    sim_coord : Narr (N,3),
        Coordinates' data from simulation to be casted. 
    tocast: arr,
        Simulation data to cast.
    weights: arr,
        Weights to use in the casting.
    keep_track: bool,
        If True, returns the indices of the points used in the casting.
    Returns
    -------
    final_casted: arr
        Casted down version of tocast
    """
    if keep_track:
        indices = np.arange(len(tocast))
        cells_used = []
    gridded_tocast = np.zeros((len(radii)))
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
        if xaxis == 'angles':
            width = width
        elif xaxis == 'radii':
            width *= 2 # make it slightly bigger to smooth things
        # indices = tree.query_ball_point(radius, width)
        indices = tree.query_radius(radius, width)
        indices = np.concatenate(indices)
        if len(indices) < 2 :
            gridded_tocast[i] = 0
            if keep_track:
                cells_used.append([])
        else:    
            indices = [int(idx) for idx in indices]
            gridded_tocast[i] = np.sum(tocast[indices] * weights[indices])
            gridded_weights[i] = np.sum(weights[indices])
            if keep_track:
                cells_used.append(indices)

    gridded_weights += 1e-20 # avoid division by zero
    final_casted = np.divide(gridded_tocast, gridded_weights)
    if keep_track:
        return final_casted, cells_used
    else:
        return final_casted

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