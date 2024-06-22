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
    # Take points in the same quadrant as the chosen point
    condition_pos = (y_data * y_chosen > 0) & (x_data * x_chosen > 0)
    condition_coord = np.logical_and(condition_plane, condition_pos)

    return condition_coord

def tangent_versor(x_orbit, y_orbit, idx, smooth_orbit = False):
    """ Find the components of the tangent vector to the orbit at the chosen point (idx)."""
    # Smooth the orbit
    x_orbit_sm = gaussian_filter1d(x_orbit, 4)
    y_orbit_sm = gaussian_filter1d(y_orbit, 4)
    # Compute the tangent vector (take care of boundaries)
    if idx == 0:
        x_coord_tg = x_orbit_sm[1] - x_orbit_sm[idx]
        y_coord_tg = y_orbit_sm[1] - y_orbit_sm[idx]
    elif idx == len(y_orbit_sm)-1:
        x_coord_tg = x_orbit_sm[idx] - x_orbit_sm[-2] 
        y_coord_tg = y_orbit_sm[idx] - y_orbit_sm[-2]
    else:
        x_coord_tg = x_orbit_sm[idx+1] - x_orbit_sm[idx-1]
        y_coord_tg = y_orbit_sm[idx+1] - y_orbit_sm[idx-1]
    vec_tg = np.array([x_coord_tg, y_coord_tg])
    vers_tg = vec_tg / np.linalg.norm(vec_tg)
    if smooth_orbit:
        return vers_tg, x_orbit_sm, y_orbit_sm
    else:
        return vers_tg
        
def transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx, coord = False):
    """
    Find the transverse plane to the orbit at the chosen point.
    If coord == True, it returns the coordinates of the data in the plane with respect to the new coordinates system,
    i.e. the one that have versors (tg_vect, Zhat X tg_vect, Zhat).
    """
    # Put the data in the reference system of the chosen point
    x_chosen, y_chosen = x_orbit[idx], y_orbit[idx]
    x_data_trasl = x_data - x_chosen
    y_data_trasl = y_data - y_chosen
    data_trasl = np.transpose([x_data_trasl, y_data_trasl])
    # Find the tg versor at the chosen point and the points orthogonal to it
    vers_tg = tangent_versor(x_orbit, y_orbit, idx)
    dot_product = np.dot(data_trasl, vers_tg) #that's the projection of the data on the tangent vector
    condition_tra = np.abs(dot_product) < dim_data 
    # Take points in the same quadrant as the chosen point
    condition_pos = (y_data * y_chosen > 0) & (x_data * x_chosen > 0) 

    condition_coord = np.logical_and(condition_tra, condition_pos)
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
        print(f'Check if you have the 0 on the transverse line. x0: {x0}')
        return condition_coord, x_onplane, x0
    else:
        return condition_coord
    
def tangent_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx):
    """ Find the tangent plane the chosen point (idx) in the SMOOTHED orbit."""
    # Put the data in the reference system of the chosen point
    x_chosen, y_chosen = x_orbit[idx], y_orbit[idx]
    x_data_trasl = x_data - x_chosen
    y_data_trasl = y_data - y_chosen
    # Find the versors in 3D
    vers_tg = tangent_versor(x_orbit, y_orbit, idx)
    zhat = np.array([0,0,1])
    xRhat = np.cross(zhat, vers_tg) # points outwards
    xRhat /= np.linalg.norm(xRhat)
    vers_norm = np.array([xRhat[0], xRhat[1]])
    # and take the points orthogonal to the vector orthogonal to the tangent vector in the orbital plane (i.e parallel to the tg)
    r_data_trasl = np.transpose(np.array([x_data_trasl, y_data_trasl]))
    condition_coord = np.abs(np.dot(r_data_trasl, vers_norm)) < dim_data

    return condition_coord

if __name__ == '__main__':
    import sys
    sys.path.append('/Users/paolamartire/shocks')
    from Utilities.operators import make_tree
    import matplotlib.pyplot as plt
    from src.orbits import follow_the_stream, make_cfr
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
    step = 0.02
    theta_params = [-theta_lim, theta_lim, step]
    theta_arr = np.arange(*theta_params)
    xcfr, ycfr, cfr = make_cfr(Rt)
    data = make_tree(path, snap, is_tde = True, energy = False)
    dim_cell = data.Vol**(1/3)

    cm, lower_tube_w, upper_tube_w, lower_tube_h, upper_tube_h, w_params, h_params  = follow_the_stream(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, threshold=1/3)
    #%%
    for idx in range(3, len(theta_arr), 8):
        condition_rad = radial_plane(data.X, data.Y, dim_cell, theta_arr[idx])
        X_rad, Y_rad, Z_rad = \
            make_slices([data.X, data.Y, data.Z], condition_rad)
        X_rad_midplane = X_rad[np.abs(Z_rad) < dim_cell[condition_rad]]
        Y_rad_midplane = Y_rad[np.abs(Z_rad) < dim_cell[condition_rad]]

        vec_tg, x_orbit_sm, y_orbit_sm = tangent_versor(cm[:,0,] ,cm[:,1], idx, smooth_orbit = True)

        condition_tra, x_onplane, x0 = transverse_plane(data.X, data.Y, dim_cell, cm[:,0,] ,cm[:,1], idx, coord = True)
        X_tra, Y_tra, Z_tra = \
            make_slices([data.X, data.Y, data.Z], condition_tra)
        X_tra_midplane = X_tra[np.abs(Z_tra) < dim_cell[condition_tra]]
        Y_tra_midplane = Y_tra[np.abs(Z_tra) < dim_cell[condition_tra]]
                               
        condition_tg = tangent_plane(data.X, data.Y, dim_cell, cm[:,0,] ,cm[:,1], idx)
        X_tg, Y_tg, Z_tg = \
            make_slices([data.X, data.Y, data.Z], condition_tg)
        X_tg_midplane = X_tg[np.abs(Z_tg) < dim_cell[condition_tg]]
        Y_tg_midplane = Y_tg[np.abs(Z_tg) < dim_cell[condition_tg]]
                             
        plt.figure(figsize=(10,5))
        plt.plot(cm[:,0], cm[:,1], '--', c = 'orange', label = 'Orbit')
        plt.contour(xcfr, ycfr, cfr, [0], linestyles = 'dotted', colors = 'k')
        plt.plot(x_orbit_sm, y_orbit_sm, c = 'orange', label = 'Smoothed orbit')
        plt.scatter(X_rad_midplane, Y_rad_midplane, s=.1, alpha = 0.8, c = 'b', label = 'Radial plane')
        plt.scatter(X_tra_midplane, Y_tra_midplane, s=.1, alpha = 0.8, c = 'r', label = 'Transverse plane')
        plt.scatter(X_tg_midplane, Y_tg_midplane, s=.1, alpha = 0.8, c = 'g', label = 'Tangent plane')
        plt.scatter([0,cm[idx,0]], [0,cm[idx,1]], c = 'r', s=10)
        plt.quiver(cm[idx,0], cm[idx,1], vec_tg[0], vec_tg[1], width = 2e-3, scale = 0.1, scale_units='xy', color='k')
        plt.xlim(-200,20)
        plt.ylim(-60,60)
        plt.legend()
        plt.show()

        