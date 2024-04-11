"""
Recurrent operators.
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import h5py
import math
import Utilities.prelude as prel

def make_tree(filename, snap, is_tde, energy = False):
    """ Load data from simulation and build the tree. """
    X = np.load(f'{filename}/CMx_{snap}.npy')
    Y = np.load(f'{filename}/CMy_{snap}.npy')
    Z = np.load(f'{filename}/CMz_{snap}.npy')
    Vol = np.load(f'{filename}/Vol_{snap}.npy')
    VX = np.load(f'{filename}/Vx_{snap}.npy')
    VY = np.load(f'{filename}/Vy_{snap}.npy')
    VZ = np.load(f'{filename}/Vz_{snap}.npy')
    Den = np.load(f'{filename}/Den_{snap}.npy')
    # Mass = np.load(f'{filename}/Mass_{snap}.npy')
    if energy:
        IE = np.load(f'{filename}/IE_{snap}.npy')
        # convert from energy/mass to energy density
        IE *= Den 
        # if is_tde:
        #     IE *= prel.en_den_converter
        Diss = np.load(f'{filename}/Diss_{snap}.npy')
             
    P = np.load(f'{filename}/P_{snap}.npy')
    T = np.load(f'{filename}/T_{snap}.npy')
    if all(T) == 0:
        print('all T=0, bro. Compute by myself!')
        T = P/Den
    if is_tde:
        #Den *= prel.den_converter
        Star = np.load(f'{filename}/Star_{snap}.npy')
        for i,rho in enumerate(Den):
            cell_star = Star[i]
            if ((1-cell_star) > 1e-3):
                rho = 0 

    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of shape (number_points, 3)
    sim_tree = KDTree(sim_value) 

    if energy:
        return sim_tree, X, Y, Z, Vol, VX, VY, VZ, IE, Den, P, T, Diss
    else: 
        return sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, P, T


def f_interp(sim_tree, f_tree, X, Y, Z, point):
    """ Interpolate a function using the 4 nearest points. """
    idx_neigh, _ = sim_tree.query(point, k = 4)
    x_tointerp = X[idx_neigh]
    y_tointerp = Y[idx_neigh]
    z_tointerp = Z[idx_neigh]
    points_tointerp = np.array([x_tointerp, y_tointerp, z_tointerp]).T
    f_tointerp = f_tree[idx_neigh]

    new_f = griddata(points_tointerp, f_tointerp, point)
    
    return new_f


def calc_grad(sim_tree, fx_tree, fy_tree, fz_tree, X, Y, Z, point, delta):
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
    # Coordinates of your point.
    x_point = point[0]
    y_point = point[1]
    z_point = point[2]
    
    # Find the coordinate and the values of f in the neighbour points (in the 3 cartisian directions)
    pre_xcoord = x_point - delta
    fpre_x = f_interp(sim_tree, fx_tree, X, Y, Z, [pre_xcoord, y_point, z_point])
    post_xcoord = x_point + delta
    fpost_x = f_interp(sim_tree, fx_tree, X, Y, Z, [post_xcoord, y_point, z_point])

    pre_ycoord = y_point - delta
    fpre_y = f_interp(sim_tree, fy_tree, X, Y, Z, [x_point, pre_ycoord, z_point])    
    post_ycoord = y_point + delta
    fpost_y = f_interp(sim_tree, fy_tree, X, Y, Z, [x_point, post_ycoord, z_point])

    pre_zcoord = z_point - delta
    fpre_z = f_interp(sim_tree, fz_tree, X, Y, Z, [x_point, y_point, pre_zcoord])    
    post_zcoord = z_point + delta
    fpost_z = f_interp(sim_tree, fz_tree, X, Y, Z, [x_point, y_point, post_zcoord])

    delta_fx = (fpost_x-fpre_x) / (2*delta)
    delta_fy = (fpost_y-fpre_y)/ (2*delta)
    delta_fz = (fpost_z-fpre_z) / (2*delta)

    grad = np.array([delta_fx, delta_fy, delta_fz])

    return grad

