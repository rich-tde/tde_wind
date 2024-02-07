import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from scipy.spatial import KDTree

def the_nearest(x_array, y_array, z_array, xyz):
    """ Gives the coordinates of the nearest point to the selected point of coordinates xyz=[x,y,z]."""
    # find the coordinates indexes of the selected point 
    i = np.argmin(np.abs(x_array-xyz[0]))
    j = np.argmin(np.abs(y_array-xyz[1]))
    k = np.argmin(np.abs(z_array-xyz[2]))

    return i,j,k 

def calc_div(x_array, y_array, z_array, fx_grid, fy_grid, fz_grid, point, kind_info):
    """ Compute the divergence """
    if kind_info == 'coord':
        i,j,k = the_nearest(x_array, y_array, z_array, point)
    elif kind_info == 'idx':
        i = point[0]
        j = point[1]
        k = point[2]

    delta_fpost = np.array([fx_grid[i+1, j, k], fy_grid[i, j+1, k],  fz_grid[i, j, k+1]])
    delta_fpre = np.array([fx_grid[i-1, j, k], fy_grid[i, j-1, k] , fz_grid[i, j, k-1] ])
    delta_f =  np.array(delta_fpost) - np.array(delta_fpre)

    xyz_post = [x_array[i+1], y_array[j+1], z_array[k+1]]
    xyz_pre = [x_array[i-1], y_array[j-1], z_array[k-1]]
    delta_coord = np.array(xyz_post) - np.array(xyz_pre)

    delta_fx = delta_f[0] / delta_coord[0]
    delta_fy = delta_f[1] / delta_coord[1]
    delta_fz = delta_f[2] / delta_coord[2]

    div_v = delta_fx + delta_fy + delta_fz
    return div_v

    
def calc_grad(x_array, y_array, z_array, f_grid, point, kind_info):
    """ Compute the gradient """
    if kind_info == 'coord':
        i,j,k = the_nearest(x_array, y_array, z_array, point)
    elif kind_info == 'idx':
        i = point[0]
        j = point[1]
        k = point[2]

    delta_fpost = np.array([f_grid[i+1, j, k], f_grid[i, j+1, k],  f_grid[i, j, k+1]])
    delta_fpre = np.array([f_grid[i-1, j, k], f_grid[i, j-1, k] , f_grid[i, j, k-1] ])
    delta_f = np.array(delta_fpost) - np.array(delta_fpre)

    xyz_post = [x_array[i+1], y_array[j+1], z_array[k+1]]
    xyz_pre = [x_array[i-1], y_array[j-1], z_array[k-1]]
    delta_coord = np.array(xyz_post) - np.array(xyz_pre)

    delta_fx = delta_f[0] / delta_coord[0]
    delta_fy = delta_f[1] / delta_coord[1]
    delta_fz = delta_f[2] / delta_coord[2]

    grad = [delta_fx, delta_fy, delta_fz]
    return grad


def zero_interpolator(x_array, y_array, z_array, f_coord, xyz):
    """  
    Piecewise-constant interpolation: returns the field value at the nearest available point.
    """
    i,j,k = the_nearest(x_array, y_array, z_array, xyz)
    f_value = f_coord[i,j,k]
    return f_value

def tree_interpolator(pre, post):
    X = np.load('data/CMx.npy')
    Y = np.load('data/CMy.npy')
    Z = np.load('data/CMz.npy')
    Den = np.load('data/Den.npy')
    P = np.load('data/P.npy')

    # make a tree
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 

    _, idx_pre = sim_tree.query(pre)
    Den_pre = Den[idx_pre]
    P_pre = P[idx_pre]
    T_pre = P_pre/Den_pre

    _, idx_post = sim_tree.query(post)
    Den_post = Den[idx_post]
    P_post = P[idx_post]
    T_post = P_post/Den_post
    
    return T_pre, P_pre, T_post, P_post
