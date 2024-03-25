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
import k3match

def make_tree(filename, snap):
    """ Load data from simulation and build the tree. """
    X = np.load(f'{filename}/CMx_{snap}.npy')
    Y = np.load(f'{filename}/CMy_{snap}.npy')
    Z = np.load(f'{filename}/CMz_{snap}.npy')
    VX = np.load(f'{filename}/Vx_{snap}.npy')
    VY = np.load(f'{filename}/Vy_{snap}.npy')
    VZ = np.load(f'{filename}/Vz_{snap}.npy')
    Den = np.load(f'{filename}/Den_{snap}.npy')
    P = np.load(f'{filename}/P_{snap}.npy')
    T = np.load(f'{filename}/T_{snap}.npy')
    if all(T) == 0:
        print('all T=0, bro. Compute by myself!')
        T = P/Den
    Vol = np.load(f'{filename}/Vol_{snap}.npy')

    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of shape (number_points, 3)
    sim_tree = KDTree(sim_value) 

    return sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, P, T

def old_select_neighbours(sim_tree, point, delta, select):
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
        delta = - delta

    x_point = point[0]
    y_point = point[1]
    z_point = point[2]
    
    # new points 
    neigh_x = [x_point + delta, y_point, z_point]
    neigh_y = [x_point, y_point + delta, z_point]
    neigh_z = [x_point, y_point, z_point + delta]

    _, idxx  = sim_tree.query(neigh_x)
    _, idxy  = sim_tree.query(neigh_y)
    _, idxz  = sim_tree.query(neigh_z)
    
    return idxx, idxy, idxz

    
def old_calc_grad(sim_tree, X, Y, Z, f_tree, point, delta, kind_info = 'point'):
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
    prex, prey, prez = old_select_neighbours(sim_tree, point, delta, 'before')
    postx, posty, postz = old_select_neighbours(sim_tree, point, delta, 'after')

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

#####################
def select_near(sim_tree, X, Y, Z, point, delta, choice):
     x_point = point[0]
     y_point = point[1]
     z_point = point[2]
     k = 0.6
     distance = 0
     while np.abs(distance)<1e-5:
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

def select_neighbours(sim_tree, X, Y, Z, point, deltax, select):
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
    deltay = deltax
    deltaz = deltax
    if select == 'before':
        deltax = - deltax
        deltay = - deltay
        deltaz = - deltaz

    idxx = select_near(sim_tree, X, Y, Z, point, deltax, 'x')
    idxy = select_near(sim_tree, X, Y, Z, point, deltay, 'y')
    idxz = select_near(sim_tree, X, Y, Z, point, deltaz, 'z')
    
    return idxx, idxy, idxz

def calc_grad(sim_tree, X, Y, Z, f_tree, point, deltax, kind_info = 'point'):
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
    prex, prey, prez = select_neighbours(sim_tree, X, Y, Z, point, deltax, 'before')
    postx, posty, postz = select_neighbours(sim_tree, X, Y, Z, point, deltax, 'after')

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

def only_twocells(sim_tree, X, Y, Z, f_tree, point):
     _, idx = sim_tree.query(point, k = 2)
     idxpost = idx[0]
     idxpre = idx[1]

     delta_fx = (f_tree[idxpost]-f_tree[idxpre]) / (X[idxpost]-X[idxpre])
     delta_fy = (f_tree[idxpost]-f_tree[idxpre])/ (Y[idxpost]-Y[idxpre])
     delta_fz = (f_tree[idxpost]-f_tree[idxpre]) / (Z[idxpost]-Z[idxpre])
     grad = np.array([delta_fx, delta_fy, delta_fz])
     
     return grad

# def interpol_grad(sim_tree, X, Y, Z, f_tree, idx, delta):
#      x_point = X[idx]
#      y_point = Y[idx]
#      z_point = Z[idx]

#      x = [[x_point + delta, y_point, z_point], [x_point - delta, y_point, z_point]]
#      y = [[x_point, y_point +  delta, z_point], [x_point, y_point -  delta, z_point]]
#      z = [[x_point, y_point, z_point + delta], [x_point, y_point, z_point - delta]]

#      indeces, _, _ = k3match.cartesian(X,Y,Z, x_point, y_point, z_point, 3*delta)

#      x_interp = np.zeros(len(indeces))
#      y_interp = np.zeros(len(indeces))
#      z_interp = np.zeros(len(indeces))
#      f_interp = np.zeros(len(indeces))

#      for i, idx in enumerate(idx):
#         x_interp[i] = X[idx]
#         y_interp[i] = Y[idx]
#         z_interp[i] = Z[idx]
#         f_interp[i] = f_tree[idx]

#      f_x = griddata((x_interp, y_interp, z_interp), f_interp, x)
#      f_x = griddata((x_interp, y_interp, z_interp), f_interp, x)
#           f_x = griddata((x_interp, y_interp, z_interp), f_interp, x)


          
