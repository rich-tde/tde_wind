import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from scipy.spatial import KDTree

def the_nearest(x_array, y_array, z_array, xyz, dim_cell, sel_output = 'idx'):
    """ Gives the coordinates of the nearest point to the selected point of coordinates xyz=[x,y,z]."""
    # filter only bunch of nearer neighbours otherwise the for loop lasts forever
    indexes_x = np.where(np.abs(x_array-xyz[0])< 2*dim_cell)
    indexes_x = indexes_x[0]
    indexes_y = np.where(np.abs(y_array-xyz[1])< 2*dim_cell)
    indexes_y = indexes_y[0]
    indexes_z = np.where(np.abs(z_array-xyz[2])< 2*dim_cell)
    indexes_z = indexes_z[0]
    # find the lowest distance to take the nearest point
    temp = 1e8
    for idxi in indexes_x:
        for idxj in indexes_y:
            for idxk in indexes_z:
                dist = np.linalg.norm(xyz - [x_array[idxi], y_array[idxj], z_array[idxk]])
                if dist < temp:
                    temp = dist
                    indexes = [idxi, idxj, idxk]
    if temp == 1e8:
        print('no near point')

    i = indexes[0]
    j = indexes[1]
    k = indexes[2]
    point = np.array([x_array[i], y_array[j], z_array[k]])

    if sel_output == 'point':
        return i,j,k,point
    else:
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
    dx = (x_array[-1]-x_array[0])/len(x_array)
    i,j,k = the_nearest(x_array, y_array, z_array, xyz, dx)
    f_value = f_coord[i,j,k]
    return f_value

def tree_interpolator(pre, post):
    X = np.load('data_sim/CMx.npy')
    Y = np.load('data_sim/CMy.npy')
    Z = np.load('data_sim/CMz.npy')
    Den = np.load('data_sim/Den.npy')
    P = np.load('data_sim/P.npy')

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
