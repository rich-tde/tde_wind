import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from scipy.ndimage import gaussian_filter1d
from Utilities.operators import from_cylindric

def make_slices(all_data, condition):
    """
    For each array of data, slice it according to a condition.
    """
    return [data[condition] for data in all_data]

def radial_plane(x_data, y_data, dim_data, theta_chosen):
    x_chosen, y_chosen = from_cylindric(theta_chosen, 1)
    # Take the orthogonal vector to radial direction 
    orthog_vector = np.array([-y_chosen, x_chosen])
    r_data = np.transpose(np.array([x_data, y_data]))
    # and take the points orthogonal to that
    condition_plane = np.abs(np.dot(r_data, orthog_vector)) < dim_data
    # take points in the same semi-plane as the chosen point
    condition_y = np.dot(r_data, np.array([x_chosen, y_chosen])) > 0
    condition_coord = np.logical_and(condition_plane, condition_y)

    return condition_coord

def tangent_versor(x_orbit_in, y_orbit_in, idx, orbit = False):
    """ Find the components of the tangent vector to the orbit at the chosen point (idx)."""
    # Smooth the orbit
    x_orbit = gaussian_filter1d(x_orbit_in, 2)
    y_orbit = gaussian_filter1d(y_orbit_in, 2)
    # Compute the tangent vector (take care of boundaries)
    if idx == 0:
        x_coord_tg = x_orbit[1] - x_orbit[idx]
        y_coord_tg = y_orbit[1] - y_orbit[idx]
    elif idx == len(y_orbit)-1:
        x_coord_tg = x_orbit[idx] - x_orbit[-2] 
        y_coord_tg = y_orbit[idx] - y_orbit[-2]
    else:
        x_coord_tg = x_orbit[idx+1] - x_orbit[idx-1]
        y_coord_tg = y_orbit[idx+1] - y_orbit[idx-1]
    vec_tg = np.array([x_coord_tg, y_coord_tg])
    vers_tg = vec_tg / np.linalg.norm(vec_tg)
    if orbit:
        return vers_tg, x_orbit, y_orbit
    else:
        return vers_tg
        
def transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx, coord = False):
    """
    Find the transverse plane to the orbit at the chosen point.
    If coord is True, it returns the coordinates of the data in the plane with respect to the new coordinates system,
    i.e. the one that have versor (tg_vect, Zhat X tg_vect, Zhat).
    """
    vers_tg = tangent_versor(x_orbit, y_orbit, idx)
    # Find the points orthogonal to the tangent vector at the chosen point
    x_chosen, y_chosen = x_orbit[idx], y_orbit[idx]
    x_data_trasl = x_data - x_chosen
    y_data_trasl = y_data - y_chosen
    data_trasl = np.transpose([x_data_trasl, y_data_trasl])
    #data_trasl /= np.linalg.norm(data_trasl)
    dot_product = np.dot(data_trasl, vers_tg) #that's the projection of the data on the tangent vector
    condition_tra = np.abs(dot_product) < dim_data 
    r_data = np.transpose(np.array([x_data, y_data]))
    # take points in the same semi-plane as the chosen point
    condition_y = np.dot(r_data, np.array([x_chosen, y_chosen])) > 0

    condition_coord = np.logical_and(condition_tra, condition_y)
    if coord:
        #Find the versors in 3D
        zhat = np.array([0,0,1])
        yRhat = np.array([vers_tg[0], vers_tg[1],0]) # points in the direction of the orbit
        xRhat = np.cross(zhat, vers_tg) # points outwards
        xRhat /= np.linalg.norm(xRhat)
        vers_norm = np.array([xRhat[0], xRhat[1]])
        #Find the coordinates of the data in the new system
        x_onplane = np.dot(data_trasl[condition_coord], vers_norm)
        # y_onplane = np.dot(data_trasl[condition_coord], vers_tg)
        x0 = np.min(np.abs(x_onplane))
        return condition_coord, x_onplane, x0
    else:
        return condition_coord
    
# def tangent_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx):
#     # Search where you are in the orbit to then find the previous point
#     x_chosen, y_chosen = x_orbit[idx], y_orbit[idx]
#     vers_tg = tangent_versor(x_orbit, y_orbit, idx)
#     mx, my = vers_tg[0], vers_tg[1]
#     ideal_y_value = y_chosen + my * (x_data-x_chosen) / mx
#     condition_coord = np.abs(y_data - ideal_y_value) < dim_data
#     return condition_coord

# def tangent_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx):
#     # Search where you are in the orbit to then find the previous point
#     x_chosen, y_chosen = x_orbit[idx], y_orbit[idx]
#     if idx == 0:
#         m = (y_orbit[1] - y_chosen) / (x_orbit[1] - x_chosen)
#     elif idx == len(y_orbit)-1:
#         m = (y_chosen - y_orbit[-2]) / (x_chosen - x_orbit[-2])
#     elif np.abs(y_chosen)<0.1: #i.e. x=Rchosen, which should be pericenter --> draw vertical line
#         r_chosen = np.sqrt(x_chosen**2 + y_chosen**2)
#         condition_coord = np.abs(x_data-r_chosen) < dim_data 
#     else:
#         m = (y_orbit[idx+1] - y_orbit[idx-1]) / (x_orbit[idx+1] - x_orbit[idx-1])
#         print(idx, y_orbit[idx]) 
#         print(idx+1, x_orbit[idx+1], y_orbit[idx+1])
#         print(idx-1, x_orbit[idx-1], y_orbit[idx-1])
#     ideal_y_value = y_chosen + m * (x_data-x_chosen)
#     condition_coord = np.abs(y_data - ideal_y_value) < dim_data
#      return condition_coord


if __name__ == '__main__':
    import sys
    sys.path.append('/Users/paolamartire/shocks')
    from Utilities.operators import make_tree
    import matplotlib.pyplot as plt
    from src.orbits import follow_the_stream
    G = 1
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = 'Low'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
    snap = '164'
    path = f'TDE/{folder}{check}/{snap}'
    Rt = Rstar * (Mbh/mstar)**(1/3)
    theta_lim = np.pi
    step = 0.1
    theta_params = [-theta_lim, theta_lim, step]
    theta_arr = np.arange(*theta_params)
    data = make_tree(path, snap, is_tde = True, energy = False)
    dim_cell = data.Vol**(1/3)

    cm, lower_tube_w, upper_tube_w, lower_tube_h, upper_tube_h, w_params, h_params  = follow_the_stream(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, threshold=1/3)
    #%%
    for idx in range(4, len(theta_arr), 8):
        condition_rad = radial_plane(data.X, data.Y, dim_cell, theta_arr[idx])
        X_rad, Y_rad, Z_rad = \
            make_slices([data.X, data.Y, data.Z], condition_rad)
        X_rad_midplane = X_rad[np.abs(Z_rad) < dim_cell[condition_rad]]
        Y_rad_midplane = Y_rad[np.abs(Z_rad) < dim_cell[condition_rad]]

        vec_tg, x_orbit, y_orbit = tangent_versor(cm[:,0,] ,cm[:,1], idx, orbit = True)
        # condition_tg = tangent_plane(data.X, data.Y, dim_cell, x_orbit, y_orbit, idx)
        # X_tg, Y_tg, Z_tg = \
        #     make_slices([data.X, data.Y, data.Z], condition_tg)
        # X_tg_midplane = X_tg[np.abs(Z_tg) < dim_cell[condition_tg]]
        # Y_tg_midplane = Y_tg[np.abs(Z_tg) < dim_cell[condition_tg]]
                             
        plt.figure(figsize=(10,6))
        plt.plot(cm[:,0], cm[:,1], c='r')
        plt.scatter(X_rad_midplane, Y_rad_midplane, s=.1,alpha = 0.5, c = 'b', label = 'Radial plane')
        # plt.scatter(X_tg_midplane, Y_tg_midplane, s=.1,alpha = 0.5, c = 'g', label = 'Tangent plane')
        plt.scatter([0,cm[idx,0,]], [0,cm[idx,1]], c='r', s=10)
        plt.quiver(cm[idx,0], cm[idx,1], vec_tg[0], vec_tg[1], scale=0.1,scale_units='xy', color='k')
        plt.xlim(-200,20)
        plt.ylim(-60,60)
        plt.legend()
        plt.show()

        