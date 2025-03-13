abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(f'{abspath}')
import numpy as np
import numba
import Utilities.prelude
from Utilities.sections import transverse_plane, make_slices, radial_plane
import src.orbits as orb

def find_radial_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, R0):
    """ Find the maxima density points in the radial plane for all the thetas in theta_arr"""
    x_max = np.zeros(len(theta_arr))
    y_max = np.zeros(len(theta_arr))
    z_max = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        # Exclude points inside the smoothing lenght and find radial plane
        condition_distance = np.sqrt(x_data**2 + y_data**2 + z_data**2) > R0 
        condition_Rplane = radial_plane(x_data, y_data, dim_data, theta_arr[i])
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        x_plane, y_plane, z_plane, den_plane = make_slices([x_data, y_data, z_data, den_data], condition_Rplane)
        # Find and save the maximum density point
        idx_max = np.argmax(den_plane) 
        x_max[i] = x_plane[idx_max]
        y_max[i] = y_plane[idx_max]
        z_max[i] = z_plane[idx_max]
        
    return x_max, y_max, z_max    

def find_radial_com(x_data, y_data, z_data, dim_data, mass_data, theta_arr, R0):
    """ Find the maxima density points in the radial plane for all the thetas in theta_arr"""
    x_com = np.zeros(len(theta_arr))
    y_com = np.zeros(len(theta_arr))
    z_com = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        # Exclude points inside the smoothing lenght and find radial plane
        condition_distance = np.sqrt(x_data**2 + y_data**2 + z_data**2) > R0 
        condition_Rplane = radial_plane(x_data, y_data, dim_data, theta_arr[i])
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        x_plane, y_plane, z_plane, mass_plane = make_slices([x_data, y_data, z_data, mass_data], condition_Rplane)
        # Find and save the com
        x_com[i] = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_com[i] = np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_com[i] = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
        
    return x_com, y_com, z_com 

@numba.njit
def get_threshold(t_plane, z_plane, r_plane, mass_plane, dim_plane, R0):
    """ Find the T-Z threshold to cut the transverse plane (as a square) in width and height.
    Parameters
    ----------
    t_plane, z_planr, r_plane, mass_plane, dim_plane : array
        T, Z, radial (spherical) coordinates, mass and dimension of points in the TZ plane.
    R0 : float
        Smoothing radius.
    Returns
    -------
    C : float
        The (upper) threshold for t and z.
    """
    # First guess of C and find the mass enclosed in the initial boundaries
    C = 2 #np.min([not_toovercome, 2])
    condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
    mass = mass_plane[condition]
    total_mass = np.sum(mass)
    while True:
        # update 
        step = 2*np.mean(dim_plane[condition])#2*np.max(dim_plane[condition])
        C += step
        condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
        # Check that you add new points
        if len(mass_plane[condition]) == len(mass):
            C += 2
            # print(C)
        else:
            tocheck = r_plane[condition]-R0
            if tocheck.any()<0:
                C -= step
                print('overcome R0', C)
                break
            mass = mass_plane[condition]
            new_mass = np.sum(mass) 
            # old mass is > 95% of the new mass
            if np.logical_and(total_mass > 0.95 * new_mass, total_mass != new_mass): # to be sure that you've done a step
                break
            total_mass = new_mass
   
    return C

def find_transverse_com(x_data, y_data, z_data, dim_data, den_data, mass_data, theta_arr, params):
    """ Find the centres of mass in the transverse plane"""
    Mbh, Rstar, mstar, beta = params[0], params[1], params[2], params[3]
    Rt = Rstar * (Mbh/mstar)**(1/3)
    R0 = 0.6 * Rt
    apo = orb.apocentre(Rstar, mstar, Mbh, beta)
    # indeces = np.arange(len(x_data))
    # Cut a bit the data for computational reasons
    cutting = np.logical_and(np.abs(z_data) < 5*Rt, np.abs(y_data) < apo)
    x_cut, y_cut, z_cut, dim_cut, den_cut, mass_cut = \
        make_slices([x_data, y_data, z_data, dim_data, den_data, mass_data], cutting)
    # Find the radial maximum points to have a first guess of the stream (necessary for the threshold and the tg)
    x_stream_rad, y_stream_rad, z_stream_rad = find_radial_maximum(x_cut, y_cut, z_cut, dim_cut, den_cut, theta_arr, R0)
    print('radial done')

    # First iteration: find the center of mass of each transverse plane of the maxima-density stream
    # indeces_cmTR = np.zeros(len(theta_arr))
    x_cmTR = np.zeros(len(theta_arr))
    y_cmTR = np.zeros(len(theta_arr))
    z_cmTR = np.zeros(len(theta_arr))
    thresh_cmTR = np.zeros(len(theta_arr))
    for idx in range(len(theta_arr)):
        print(idx)
        # Find the transverse plane
        condition_T, x_T, _ = transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_stream_rad, y_stream_rad, z_stream_rad, idx, coord = True)
        x_plane, y_plane, z_plane, mass_plane, dim_plane = \
            make_slices([x_cut, y_cut, z_cut, mass_cut, dim_cut], condition_T)
        # Cut the TZ plane to not keep points too far away.
        r_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
        thresh = get_threshold(x_T, z_plane, r_plane, mass_plane, dim_plane, R0) #8 * Rstar * (r_stream_rad[idx]/Rp)**(1/2)
        condition_x = np.abs(x_T) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        x_plane, y_plane, z_plane, mass_plane = \
            make_slices([x_plane, y_plane, z_plane, mass_plane], condition)
        # Find the center of mass
        x_cmTR[idx] = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_cmTR[idx]= np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_cmTR[idx] = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
        thresh_cmTR[idx] = thresh

    #     x_com = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
    #     y_com = np.sum(y_plane * mass_plane) / np.sum(mass_plane)
    #     z_com = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
    #     # Search in the tree the closest point to the center of mass
    #     points = np.array([x_plane, y_plane, z_plane]).T
    #     tree = KDTree(points)
    #     _, idx_cm = tree.query([x_com, y_com, z_com])
    #     indeces_cm[idx]= indeces_plane[idx_cm]
    # indeces_cm = indeces_cm.astype(int)

    return x_cmTR, y_cmTR, z_cmTR, thresh_cmTR

if __name__ == '__main__':
    from Utilities.operators import make_tree, Ryan_sampler, from_cylindric
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm   

    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5 
    compton = 'Compton'
    check = ''
    snap = 164

    Rt = Rstar * (Mbh/mstar)**(1/3)
    Rp = beta * Rt
    R0 = 0.6 * Rt
    apo = orb.apocentre(Rstar, mstar, Mbh, beta)
    a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
    e_mb = orb.eccentricity(Rstar, mstar, Mbh, beta)
    params = [Mbh, Rstar, mstar, beta]
    theta_lim = np.pi
    step = 0.02
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    r_kepler = orb.keplerian_orbit(theta_arr, a_mb, Rp, ecc=e_mb)
    x_kepler, y_kepler = from_cylindric(theta_arr, r_kepler)
    
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    path = f'{abspath}/TDE/{folder}/{snap}'
    data = make_tree(path, snap, energy = False)
    X, Y, Z, Den, Mass = data.X, data.Y, data.Z, data.Den, data.Mass
    dim_cell = data.Vol**(1/3)
    cutden = Den > 1e-19
    X, Y, Z, dim_cell, Den, Mass = make_slices([X, Y, Z, dim_cell, Den, Mass], cutden)
    midplane = np.abs(Z) < dim_cell
    x_mid, y_mid, z_mid, dim_mid, den_mid, mass_mid = \
        make_slices([X, Y, Z, dim_cell, Den, Mass], midplane)

    #%% Check radial COM or density max
    cutting =  np.logical_and(np.abs(Z) < 5*Rt, np.abs(Y) < apo)
    x_cut, y_cut, z_cut, dim_cut, den_cut, mass_cut = \
        make_slices([X, Y, Z, dim_cell, Den, Mass], cutting)
    # Find the radial maximum points to have a first guess of the stream (necessary for the threshold and the tg)
    x_max_rad, y_max_rad, z_max_rad = find_radial_maximum(x_cut, y_cut, z_cut, dim_cut, den_cut, theta_arr, R0)
    x_com_rad, y_com_rad, z_com_rad = find_radial_com(x_cut, y_cut, z_cut, dim_cut, mass_cut, theta_arr, R0)

    #%% Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10,10))
    img = ax1.scatter(x_mid, y_mid, c = den_mid, cmap = 'viridis', s = 1, norm = LogNorm(vmin = 5e-10, vmax = 1e-6))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Density $[M_\odot/R_\odot^3$]')
    img = ax2.scatter(x_mid, y_mid, c = mass_mid, cmap = 'viridis', s = 1, norm = LogNorm(vmin = 1e-10, vmax = 1e-7))
    cbar = plt.colorbar(img)
    cbar.set_label(r'Cell mass $[M_\odot]$')
    for ax in [ax1, ax2]:
        ax.plot(x_max_rad, y_max_rad, c = 'k', linewidth = 2, label = 'Density max')
        ax.plot(x_com_rad, y_com_rad, c = 'r', linewidth = 2, label = 'COM')
        ax.set_xlim(-apo+1,20)
        ax.set_ylim(-80,80)
        ax.set_ylabel(r'Y [$R_\odot$]')
        ax.grid()
    ax1.legend(fontsize = 18)
    ax2.set_xlabel(r'X [$R_\odot$]')
    plt.show()  
    
    #%% Transverse com
    # if stream_{check}{snap} exists, open it, otherwise create it
    try:
        theta_arr, x_stream, y_stream, z_stream, thresh_cm = \
            np.load(f'/Users/paolamartire/shocks/src/stream/stream_{check}{snap}.npy')
    except:
        x_stream, y_stream, z_stream, thresh_cm = find_transverse_com(data.X, data.Y, data.Z, dim_cell, data.Den, data.Mass, theta_arr, params)
        np.save(f'/Users/paolamartire/shocks/src/stream/stream_{check}{snap}.npy', [theta_arr, x_stream, y_stream, z_stream, thresh_cm])

    #%% Plot
    plt.figure(figsize = (10,5))
    plt.plot(x_kepler, y_kepler, c = 'darkviolet', label = 'Kepler', alpha = 0.5)
    plt.plot(x_stream, y_stream, c = 'dodgerblue', label = 'COM')
    plt.xlim(-apo+1,20)
    plt.ylim(-80,80)
    plt.xlabel(r'X [$R_\odot$]')
    plt.ylabel(r'Y [$R_\odot$]')
    plt.grid()
    plt.legend(fontsize = 18)
    plt.show()  
# %%
