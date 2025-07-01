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
    # Take the orthogonal versor to radial direction 
    orthog_vector = np.array([-y_chosen, x_chosen]) # 1x2 vector
    r_data = np.transpose(np.array([x_data, y_data])) # Nx2 vector
    # and take the points orthogonal to that
    condition_plane = np.abs(np.dot(r_data, orthog_vector)) < dim_data
    # Take points in the same quadrant as the chosen point (i.e. select half plane)
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
    norm_tg = np.max(np.linalg.norm(vec_tg), 1e-20)
    vers_tg = vec_tg / norm_tg
    if smooth_orbit:
        return vers_tg, x_orbit_sm, y_orbit_sm
    else:
        return vers_tg
         
# def transverse_plane(x_data, y_data, z_data, dim_data, x_orbit, y_orbit, z_orbit, idx, step_ang, coord = False):
def transverse_plane(x_data, y_data, z_data, dim_data, x_orbit, y_orbit, z_orbit, idx, just_plane = True):
    """
    Find the transverse plane to the orbit at the chosen point.
    If coord == True, it returns the coordinates of the data in the plane with respect to the new coordinates system,
    i.e. the one that have versors (That =: Zhat X tg_vect, Zhat, tg_vect).
    """
    # Put the data in the reference system of the chosen point
    x_chosen, y_chosen, z_chosen = x_orbit[idx], y_orbit[idx], z_orbit[idx]
    x_data_trasl = x_data - x_chosen
    y_data_trasl = y_data - y_chosen
    data_trasl = np.transpose([x_data_trasl, y_data_trasl])
    # Find the tg versor at the chosen point and the points orthogonal to it
    vers_tg = tangent_versor(x_orbit, y_orbit, idx)
    dot_product = np.dot(data_trasl, vers_tg) #that's the projection of the data on the tangent vector
    condition_tra = np.abs(dot_product) < dim_data 

    # Slice as wide as the widest --> more cells in the center 
    # Consider just the cells nearby (both in 2D and 3D)
    # R_data_mod = np.sqrt(x_data**2 + y_data**2+ z_data**2)
    # R_chosen_mod = np.sqrt(x_chosen**2 + y_chosen**2 + z_chosen**2)
    # r_data_mod = np.sqrt(x_data**2 + y_data**2)
    # r_chosen_mod = np.sqrt(x_chosen**2 + y_chosen**2)
    s = 0.5 #[R_star] since in BonnerotLu22 the planes have thickness 1R_star. If you use step_ang: 2*step_ang * r_chosen_mod
    condition_x = np.abs(x_data - x_chosen) < s
    condition_y = np.abs(y_data - y_chosen) < s
    condition_z = np.abs(z_data - z_chosen) < s
    condition_coord = np.logical_and(condition_x, np.logical_and(condition_y, condition_z))
    # Find the widest cell
    if (condition_tra&condition_coord).any() != False:
        max_dim = np.max(dim_data[condition_tra&condition_coord])#_R&condition_r])
        # Change the condition_tra
        condition_tra = np.abs(dot_product) < max_dim

    # Find the coordinates of the data in the new system and use them to cut the plane
    # Find the versors in 3D
    zhat = np.array([0,0,1])
    kRhat = np.array([vers_tg[0], vers_tg[1],0]) # points in the direction of the orbit
    xRhat = np.cross(zhat, vers_tg) # points outwards
    xRhat /= np.linalg.norm(xRhat)
    vers_norm = np.array([xRhat[0], xRhat[1]])
    # New x (T) coordinate
    x_onplaneall = np.dot(data_trasl, vers_norm)
    # condition_cut = np.abs(x_onplaneall) < 100
    # condition_tra = np.logical_and(condition_tra, condition_cut)
    if just_plane:
        x_onplane = x_onplaneall[condition_tra]
        # y_onplane = np.dot(data_trasl[condition_coord], vers_tg)
        # find the T of xchosen (won't be 0 because stream points are among simulation data)
        rad_trasl = np.sqrt(x_data_trasl**2 + y_data_trasl**2)
        x0 = x_onplaneall[np.argmin(rad_trasl)]
        return condition_tra, x_onplane, x0
    else:
        return condition_tra, x_onplaneall
    
def tangent_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx):
    """ Find the tangent plane the chosen point (idx) in the SMOOTHED orbit."""
    # Put the data in the reference system of the chosen point
    x_chosen, y_chosen = x_orbit[idx], y_orbit[idx]
    x_data_trasl = x_data - x_chosen
    y_data_trasl = y_data - y_chosen
    # Find the versors in 3D: (T,Z, K=tg_vec)
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
    check = ''
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}Compton'
    snap = '164'
    path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
    Rt = Rstar * (Mbh/mstar)**(1/3)
    theta_lim = np.pi
    step = 0.02
    params = [Mbh, Rstar, mstar, beta]
    xcfr, ycfr, cfr = make_cfr(Rt)
    data = make_tree(path, snap, energy = False)
    dim_cell = data.Vol**(1/3)

    stream = np.load(f'/Users/paolamartire/shocks/data/{folder}/stream_{check}{snap}.npy' )
    theta_arr, x_stream, y_stream, z_stream = stream[0], stream[1], stream[2], stream[3]
      
    # idx =  100
    # condition_rad = radial_plane(data.X, data.Y, dim_cell, theta_arr[idx])
    # X_rad, Y_rad, Z_rad = \
    #     make_slices([data.X, data.Y, data.Z], condition_rad)
    # X_rad_midplane = X_rad[np.abs(Z_rad) < dim_cell[condition_rad]]
    # Y_rad_midplane = Y_rad[np.abs(Z_rad) < dim_cell[condition_rad]]

    # vec_tg, x_orbit_sm, y_orbit_sm = tangent_versor(x_stream, y_stream, idx, smooth_orbit = True)

    plt.figure(figsize = (12,4))
    for idx in range(5,200):
        step_ang = theta_arr[idx+1]-theta_arr[idx]
        # condition_tra, x_onplane, x0 = transverse_plane(data.X, data.Y, data.Z, dim_cell, x_stream, y_stream, z_stream, idx, step_ang, coord = True)
        condition_tra, x_onplane, x0 = transverse_plane(data.X, data.Y, data.Z, dim_cell, x_stream, y_stream, z_stream, idx, coord = True)
        X_tra, Y_tra, Z_tra = \
            make_slices([data.X, data.Y, data.Z], condition_tra)
        X_tra_midplane = X_tra[np.abs(Z_tra) < dim_cell[condition_tra]]
        Y_tra_midplane = Y_tra[np.abs(Z_tra) < dim_cell[condition_tra]]
                                
        condition_tg = tangent_plane(data.X, data.Y, dim_cell, x_stream, y_stream, idx)
        X_tg, Y_tg, Z_tg = \
            make_slices([data.X, data.Y, data.Z], condition_tg)
        X_tg_midplane = X_tg[np.abs(Z_tg) < dim_cell[condition_tg]]
        Y_tg_midplane = Y_tg[np.abs(Z_tg) < dim_cell[condition_tg]]
        
        # plt.scatter(X_rad_midplane, Y_rad_midplane, s=.1, alpha = 0.8, c = 'b', label = 'Radial plane')
        plt.scatter(X_tra_midplane, Y_tra_midplane, s=.1, alpha = 0.8,  label = 'Transverse plane')
        # plt.scatter(X_tg_midplane, Y_tg_midplane, s=.1, alpha = 0.8, c = 'g', label = 'Tangent plane')
        # plt.quiver(x_orbit_sm[idx], y_orbit_sm[idx], vec_tg[0], vec_tg[1], width = 2e-3, scale = 0.1, scale_units='xy', color='k')
    plt.plot(x_stream, y_stream,  c = 'r', label = 'Orbit')
    plt.contour(xcfr, ycfr, cfr, [0], linestyles = 'dotted', colors = 'k')
    plt.xlim(-300,20)
    plt.ylim(-60,60)
    # plt.legend()
    plt.title(r'Thickness planes $= 1R_\odot$', fontsize = 14)
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{check}/Transverse/transverseslice.png')
    plt.show()

            