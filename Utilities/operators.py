"""
Recurrent operators.
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from scipy.spatial import KDTree
import h5py

def mask(X, Y, Z, xlim, ylim, zlim):
    """ Mask the data to avoid some cells (eg near the boarders of the simulation box)."""
    Xmask = X[np.logical_and(np.logical_and(Y > - ylim, Y < ylim), (X > - xlim, X < xlim))]
    Ymask = Y[np.logical_and(Y > - ylim, Y < ylim)]
    Zmask = Z[np.logical_and(Z > - zlim, Z > zlim)]
    return Xmask, Ymask, Zmask

def make_tree():
    """ Load data from simulation and build the tree. """
    X = np.load('data_sim/CMx.npy')
    Y = np.load('data_sim/CMy.npy')
    Z = np.load('data_sim/CMz.npy')
    VX = np.load('data_sim/Vx.npy')
    VY = np.load('data_sim/Vy.npy')
    VZ = np.load('data_sim/Vz.npy')
    Den = np.load('data_sim/Den.npy')
    P = np.load('data_sim/P.npy')
    T = np.load('data_sim/T.npy')
    if all(T)==0:
        print('all T=0, bro. CHANGE!')
        T = P/Den
    
    # Mask to avoid values at the borders.
    # Xmask, Ymask, Zmask = mask(X, Y, Z, xlim, ylim, zlim)
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 

    return sim_tree, X, Y, Z, VX, VY, VZ, Den, P, T

def select_neighbours(sim_tree, point, delta, select):
    """ Find the prevoius/next point in one (cartesian) direction.
    Parameters
    -----------
    sim_tree: nDarray.
            Tree where to search the point
    point: array.
            Coordinates [x,y,z] of the starting point.
    delta: float.
            Step between 2 neighbours.
    select: str.
            If 'pre' --> you search the previous points respectively in x,y,z direction
            If 'post' --> you search the next points respectively in x,y,z direction
    Returns
    -----------
    idxx, idyy, idzz: int.
                     (Tree) indexes of the previus/next points searched.
    """
    if select == 'pre':
        delta = - delta

    x_point = point[0]
    y_point = point[1]
    z_point = point[2]
    
    # new points 
    neigh_x = [x_point+delta, y_point, z_point]
    neigh_y = [x_point, y_point+delta, z_point]
    neigh_z = [x_point, y_point, z_point+delta]

    _, idxx  = sim_tree.query(neigh_x)
    _, idxy  = sim_tree.query(neigh_y)
    _, idxz  = sim_tree.query(neigh_z)
    
    return idxx, idxy, idxz


def calc_div(sim_tree, X, Y, Z, fx_tree, fy_tree, fz_tree, point, delta, kind_info = 'point'):
    """ Compute the divergence.
    Parameters
    -----------
    sim_tree: nDarray.
            Tree where to search the point.
    X, Y, Z: arrays.
            Coordinates of the points of the tree.
    fx_tree, fy_tree, fz_tree: arrays.
                                Components of the quantity f of the tree.
    point: array.
            Starting point.
    delta: float.
            Step between 2 neighbours.
    kind_info: str.
                Tell you if points is given in cartesian coordinates ('point') or if you have its tree index ('idx')
    Returns
    -----------
    div_f: float.
            Divergence of f.
      """
    if kind_info == 'idx':
        _, idx = sim_tree.query(point)
        x_point = X[idx]
        y_point = Y[idx]
        z_point = Z[idx]
        point = [x_point, y_point, z_point]
    
    # Find tree indexes of the pre/post neighbours.
    prex, prey, prez = select_neighbours(sim_tree, point, delta, 'pre')
    postx, posty, postz = select_neighbours(sim_tree, point, delta, 'post')

    # Find the values of/in these points.
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

    
def calc_grad(sim_tree, X, Y, Z, f_tree, point, delta, kind_info = 'point'):
    """ Compute the gradient.
    Parameters
    -----------
    As the ones of calc_div except
    f_tree: array
            quantity f of the tree.
    Returns
    -----------
    grad: array.
        Gradient of f.
    """
    if kind_info == 'idx':
        _, idx = sim_tree.query(point)
        x_point = X[idx]
        y_point = Y[idx]
        z_point = Z[idx]
        point = [x_point, y_point, z_point]

    # Find tree indexes of the pre/post neighbours.
    prex, prey, prez = select_neighbours(sim_tree, point, delta, 'pre')
    postx, posty, postz = select_neighbours(sim_tree, point, delta, 'post')

    # Find the values of/in these points.
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

    grad = [delta_fx, delta_fy, delta_fz]
    return grad
