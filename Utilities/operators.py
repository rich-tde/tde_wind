import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from scipy.spatial import KDTree


def select_neighbours(sim_tree, point, select):
    """ Find the prevoius/next point in one (cartesian) direction """
    delta = 0.05
    if select == 'pre':
        delta = - delta

    x_point = point[0]
    y_point = point[1]
    z_point = point[2]
    
    neigh_x = [x_point+delta, y_point, z_point]
    neigh_y = [x_point, y_point+delta, z_point]
    neigh_z = [x_point, y_point, z_point+delta]

    _, idxx  = sim_tree.query(neigh_x)
    _, idxy  = sim_tree.query(neigh_y)
    _, idxz  = sim_tree.query(neigh_z)
    
    return idxx, idxy, idxz


def calc_div(sim_tree, X, Y, Z, fx_tree, fy_tree, fz_tree, point, kind_info = 'point'):
    """ Compute the divergence """
    if kind_info == 'idx':
        _, idx = sim_tree.query(point)
        x_point = X[idx]
        y_point = Y[idx]
        z_point = Z[idx]
        point = [x_point, y_point, z_point]

    prex, prey, prez = select_neighbours(sim_tree, point, 'pre')
    postx, posty, postz = select_neighbours(sim_tree, point, 'post')

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

    div_v = delta_fx + delta_fy + delta_fz
    return div_v

    
def calc_grad(sim_tree, X, Y, Z, f_tree, point, kind_info = 'point'):
    """ Compute the gradient """
    if kind_info == 'idx':
        _, idx = sim_tree.query(point)
        x_point = X[idx]
        y_point = Y[idx]
        z_point = Z[idx]
        point = [x_point, y_point, z_point]

    prex, prey, prez = select_neighbours(sim_tree, point, 'pre')
    postx, posty, postz = select_neighbours(sim_tree, point, 'post')

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
