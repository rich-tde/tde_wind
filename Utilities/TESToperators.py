"""
Recurrent operators.
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from scipy.spatial import KDTree
import h5py
import math


def make_tree(filename):
    """ Load data from simulation and build the tree. """
    X = np.load(f'{filename}/CMx.npy')
    Y = np.load(f'{filename}/CMy.npy')
    Z = np.load(f'{filename}/CMz.npy')
    VX = np.load(f'{filename}/Vx.npy')
    VY = np.load(f'{filename}/Vy.npy')
    VZ = np.load(f'{filename}/Vz.npy')
    Den = np.load(f'{filename}/Den.npy')
    P = np.load(f'{filename}/P.npy')
    T = np.load(f'{filename}/T.npy')
    if all(T) == 0:
        print('all T=0, bro. Compute by myself!')
        T = P/Den
    Vol = np.load(f'{filename}/Vol.npy')

    # Mask to avoid values at the borders.
    # Xmask, Ymask, Zmask = mask(X, Y, Z, lim, kind = 'slice', choose_coord)
        
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of shape (number_points, 3)
    sim_tree = KDTree(sim_value) 

    return sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, P, T

def select_near(sim_tree, X, Y, Z, point, delta, choice):
     x_point = point[0]
     y_point = point[1]
     z_point = point[2]
     k = 0.5
     distance = 0
     while distance == 0:
        if choice == 'x':
                new_point = [x_point + k * delta, y_point, z_point]
        elif choice == 'y':
                new_point = [x_point, y_point +  k * delta, z_point]
        elif choice == 'z':
                new_point = [x_point, y_point, z_point +  k * delta]
        _, idx  = sim_tree.query(new_point)
        check_point = np.array([X[idx], Y[idx], Z[idx]])
        distance = math.dist(point, check_point)
        k += 0.1

     return idx

def select_neighbours(sim_tree, X, Y, Z, point, deltax, deltay, deltaz, select):
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
            otherwise --> you search the next points respectively in x,y,z direction
    Returns
    -----------
    idxx, idyy, idzz: int.
         (Tree) indexes of the previous/next point searched.
    """
    if select == 'before':
        deltax = - deltax
        deltay = - deltay
        deltaz = - deltaz

    idxx = select_near(sim_tree, X, Y, Z, point, deltax, 'x')
    idxy = select_near(sim_tree, X, Y, Z, point, deltay, 'y')
    idxz = select_near(sim_tree, X, Y, Z, point, deltaz, 'z')
    
    return idxx, idxy, idxz


def calc_div(sim_tree, X, Y, Z, fx_tree, fy_tree, fz_tree, point, deltax, deltay, deltaz, kind_info = 'point'):
    """ Compute the divergence.
    Parameters
    -----------
    sim_tree: nDarray.
            Tree where to search the point.
    X, Y, Z: arrays (all of the SAME lenght)
            Coordinates of the points of the tree.
    fx_tree, fy_tree, fz_tree: arrays of len=len(X).
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
    # Search the coordinates of the point if you don't have it.
    if kind_info == 'idx':
        _, idx = sim_tree.query(point)
        x_point = X[idx]
        y_point = Y[idx]
        z_point = Z[idx]
        point = [x_point, y_point, z_point]
    
    # Find tree indexes of the pre/post neighbours.
    
    prex, prey, prez = select_neighbours(sim_tree, X, Y, Z, point, deltax, deltay, deltaz, 'before')
    postx, posty, postz = select_neighbours(sim_tree, X, Y, Z, point, deltax, deltay, deltaz, 'after')

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

    
def test_calc_grad(sim_tree, X, Y, Z, f_tree, point, deltax, deltay, deltaz, kind_info = 'point'):
    """ Compute the gradient.
    Parameters
    -----------
    As the ones of calc_div except
    f_tree: array of len=len(X)-
            Quantity f of the tree.
    Returns
    -----------
    grad: array.
        Gradient of f.
    """
    # Search the coordinates of the point if you don't have it.
    if kind_info == 'idx':
        _, idx = sim_tree.query(point)
        x_point = X[idx]
        y_point = Y[idx]
        z_point = Z[idx]
        point = [x_point, y_point, z_point]

    # Find tree indexes of the pre/post neighbours.
    prex, prey, prez = select_neighbours(sim_tree, X, Y, Z, point, deltax, deltay, deltaz, 'before')
    postx, posty, postz = select_neighbours(sim_tree, X, Y, Z, point, deltax, deltay, deltaz, 'after')

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

    grad = np.array([delta_fx, delta_fy, delta_fz])
    return grad
