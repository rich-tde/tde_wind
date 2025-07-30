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

def tangent_versor(x_stream, y_stream, idx, smooth_stream = False):
    """ Find the components of the tangent vector to the stream at the chosen point (idx)."""
    # Smooth the stream
    x_stream_sm = gaussian_filter1d(x_stream, 4)
    y_stream_sm = gaussian_filter1d(y_stream, 4)
    # Compute the tangent vector (take care of boundaries)
    if idx == 0:
        x_coord_tg = x_stream_sm[1] - x_stream_sm[idx]
        y_coord_tg = y_stream_sm[1] - y_stream_sm[idx]
    elif idx == len(y_stream_sm)-1:
        x_coord_tg = x_stream_sm[idx] - x_stream_sm[-2] 
        y_coord_tg = y_stream_sm[idx] - y_stream_sm[-2]
    else:
        x_coord_tg = x_stream_sm[idx+1] - x_stream_sm[idx-1]
        y_coord_tg = y_stream_sm[idx+1] - y_stream_sm[idx-1]
    vec_tg = np.array([x_coord_tg, y_coord_tg])
    norm_tg = max(np.linalg.norm(vec_tg), 1e-20)
    vers_tg = vec_tg / norm_tg
    if smooth_stream:
        return vers_tg, x_stream_sm, y_stream_sm
    else:
        return vers_tg


def change_coordinates(x_data, y_data, x_stream, y_stream, idx, z_datas = None):
    """ 2D-Rototranslation of the reference system in the chosen point 
    with orientation given by vers_tg.
    Parameters:
    x_data, y_data: arrays
        X, Y coordinates of the data points.
    x_chosen, y_chosen: float
        X, Y coordinates of the chosen point.
    vers_tg: array_like
        Tangent vector at the chosen point.
    z_datas: list, optional
        If provided, it contains Z coordinates of the data points and the chosen point.
        If None, Z coordinates are not considered.
    Returns:
    x_onplaneall, y_onplaneall: array_like
        X, Y coordinates of the data points in the transverse plane.
    x_data_trasl, y_data_trasl: array_like
        X, Y coordinates of the data points after translation to the chosen point.
    z_data_trasl: array_like, optional
        Z coordinates of the data points after translation to the chosen point.
    """ 
    # Put the data in the reference system of the chosen point
    x_chosen, y_chosen = x_stream[idx], y_stream[idx]
    # Find the tg versor at the chosen point and the points orthogonal to it
    vers_tg = tangent_versor(x_stream, y_stream, idx)
    x_data_trasl = x_data - x_chosen
    y_data_trasl = y_data - y_chosen
    data_trasl = np.transpose([x_data_trasl, y_data_trasl])
    if z_datas is not None:
        z_data = z_datas[0]
        z_stream = z_datas[1]
        z_chosen = z_stream[idx]
        z_data_trasl = z_data - z_chosen
    zhat = np.array([0,0,1])
    # kRhat = np.array([vers_tg[0], vers_tg[1],0]) # points in the direction of the orbit
    xRhat = np.cross(zhat, vers_tg) # points outwards
    xRhat /= max(np.linalg.norm(xRhat), 1e-20) # avoid division by zero
    # kRhat = np.array([vers_tg[0], vers_tg[1],0]) # points in the direction of the orbit
    vers_norm = np.array([xRhat[0], xRhat[1]]) 
    x_onplaneall = np.dot(data_trasl, vers_norm)
    y_onplaneall = np.dot(data_trasl, vers_tg)
    if z_datas is not None:
        return x_onplaneall, y_onplaneall, x_data_trasl, y_data_trasl, z_data_trasl
    else:
        return x_onplaneall, y_onplaneall
             
def transverse_plane(x_data, y_data, z_data, dim_data, x_stream, y_stream, z_stream, idx, rstar, just_plane = True):
    """
    Parameters:
    x_data, y_data, z_data, dim_data: arrays 
        X, Y, Z coordinates, dimension of simulation points.
    x_stream, y_stream, z_stream: arrays
        X, Y, Z coordinates of the stream.
    idx : int
        Index of the chosen point along the stream
    rstar : float
        Stellar radius (used to define plane thickness = 0.5 * rstar)
    just_plane : bool, optional
        If True, return only data points within the transverse plane.
        If False, return T-coordinates for all simulation data. Default is True.
    ----     
    Returns:
    condition_tra : array_like (bool)
        Boolean mask indicating which data points lie within the transverse plane
    x_onplane or x_onplaneall : array_like
        T-coordinates of data points:
        - If just_plane=True: T-coordinates only for points in the transverse plane
        - If just_plane=False: T-coordinates for all simulation data
    x0 : float (only if just_plane=True)
        T-coordinate of the central point.
    """        
    # Rototrasl data
    x_onplaneall, y_onplaneall, x_data_trasl, y_data_trasl, z_data_trasl = \
        change_coordinates(x_data, y_data, x_stream, y_stream, idx, z_datas = [z_data, z_stream])
    condition_tra = np.abs(y_onplaneall) < dim_data 

    s = 0.5 * rstar  # so thickess is = R_star as in BonnerotLu22. If you use step_ang: 2*step_ang * r_chosen_mod
    condition_x = np.abs(x_data_trasl) < s
    condition_y = np.abs(y_data_trasl) < s
    condition_z = np.abs(z_data_trasl) < s
    condition_coord = np.logical_and(condition_x, np.logical_and(condition_y, condition_z))
    # keep only the points in the tg plane and, if possible, near the chosen point
    if (condition_tra&condition_coord).any() != False:
        max_dim = np.max(dim_data[condition_tra&condition_coord])#_R&condition_r])
        # Change the condition_tra
        condition_tra = np.abs(y_onplaneall) < max_dim

    if just_plane:
        x_onplane = x_onplaneall[condition_tra]
        # y_onplane = y_onplaneall[condition_tra]
        # find the T of xchosen (won't be 0 because stream points are among simulation data)
        rad_trasl = np.sqrt(x_data_trasl**2 + y_data_trasl**2)
        x0 = x_onplaneall[np.argmin(rad_trasl)] # since for now the stream is not in the data (u don't query it), it's not 0
        return condition_tra, x_onplane, x0
    else:
        return condition_tra, x_onplaneall
    
def tangent_plane(x_data, y_data, dim_data, x_stream, y_stream, idx):
    """ Find the tangent plane the chosen point (idx) in the SMOOTHED stream."""
    # Put the data in the reference system of the chosen point
    x_chosen, y_chosen = x_stream[idx], y_stream[idx]
    x_data_trasl = x_data - x_chosen
    y_data_trasl = y_data - y_chosen
    # Find the versors in 3D: (T,Z, K=tg_vec)
    vers_tg = tangent_versor(x_stream, y_stream, idx)
    zhat = np.array([0,0,1])
    xRhat = np.cross(zhat, vers_tg) # points outwards
    xRhat /= max(np.linalg.norm(xRhat), 1e-20) # avoid division by zero
    vers_norm = np.array([xRhat[0], xRhat[1]])
    # and take the points orthogonal to the vector orthogonal to the tangent vector in the orbital plane (i.e parallel to the tg)
    r_data_trasl = np.transpose(np.array([x_data_trasl, y_data_trasl]))
    condition_coord = np.abs(np.dot(r_data_trasl, vers_norm)) < dim_data

    return condition_coord

if __name__ == '__main__':
    from Utilities.operators import make_tree
    import matplotlib.pyplot as plt
    from src.orbits import make_cfr
    G = 1
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    snap = '162'
    check = 'NewAMR'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}Compton{check}'
    path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
    tfb = np.loadtxt(f'{path}/tfb_{snap}.txt')
    Rt = Rstar * (Mbh/mstar)**(1/3)
    params = [Mbh, Rstar, mstar, beta]
    xcfr, ycfr, cfr = make_cfr(Rt)
    data = make_tree(path, snap, energy = False)
    X, Y, Z, Vol = data.X, data.Y, data.Z, data.Vol
    dim_cell = Vol**(1/3)

    theta_arr, x_stream, y_stream, z_stream, _ = \
        np.load(f'/Users/paolamartire/shocks/data/{folder}/WH/stream/stream_{check}{snap}.npy' )
      
    # idx =  100
    # condition_rad = radial_plane(X, Y, dim_cell, theta_arr[idx])
    # X_rad, Y_rad, Z_rad = \
    #     make_slices([X, Y, Z], condition_rad)
    # X_rad_midplane = X_rad[np.abs(Z_rad) < dim_cell[condition_rad]]
    # Y_rad_midplane = Y_rad[np.abs(Z_rad) < dim_cell[condition_rad]]

    # vec_tg, x_stream_sm, y_stream_sm = tangent_versor(x_stream, y_stream, idx, smooth_stream = True)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,4), width_ratios=(1.5,.8))
    for idx in range(140,142):
    # idx = 120
        condition_tra, x_onplane, _ = transverse_plane(X, Y, Z, dim_cell, x_stream, y_stream, z_stream, idx, Rstar, just_plane = True)
        X_tra, Y_tra, Z_tra = \
            make_slices([X, Y, Z], condition_tra)
        X_tra_midplane = X_tra[np.abs(Z_tra) < dim_cell[condition_tra]]
        Y_tra_midplane = Y_tra[np.abs(Z_tra) < dim_cell[condition_tra]]
                                
        condition_tg = tangent_plane(X, Y, dim_cell, x_stream, y_stream, idx)
        X_tg, Y_tg, Z_tg = \
            make_slices([X, Y, Z], condition_tg)
        X_tg_midplane = X_tg[np.abs(Z_tg) < dim_cell[condition_tg]]
        Y_tg_midplane = Y_tg[np.abs(Z_tg) < dim_cell[condition_tg]]
        
        for ax in [ax1, ax2]:
            ax.scatter(X_tra_midplane, Y_tra_midplane, s=.1, alpha = 0.8,  label = 'Transverse plane')
            ax.scatter(X_tg_midplane, Y_tg_midplane, s=.1, alpha = 0.8, c = 'g', label = 'Tangent plane')
            ax.plot(x_stream, y_stream,  c = 'k', label = 'Stream')
            # ax.contour(xcfr, ycfr, cfr, [0], linestyles = 'dotted', colors = 'k')
            ax.set_xlabel(r'$X [R_\odot]$')
    ax1.set_xlim(-300,20)
    ax1.set_ylim(-60,60)
    ax2.set_xlim(0,15)
    ax2.set_ylim(-5,15)
    ax1.set_ylabel(r'$Y [R_\odot]$')
    plt.suptitle(f'Transverse and tangential plane at t = {tfb:.2f}' + r' $t_{\rm fb}$ for $\theta$ = ' + f'{theta_arr[idx]:.2f} rad', fontsize = 18)
    plt.tight_layout()
    plt.show()

            