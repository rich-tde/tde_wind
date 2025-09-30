""" Define the stream. """
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    abspath = '/Users/paolamartire/shocks'
    import matplotlib.pyplot as plt
    from Utilities.basic_units import radians
    import matplotlib.colors as colors
    save = False

import numpy as np
import numba

import Utilities.prelude as prel
import Utilities.sections as sec
import src.orbits as orb
from Utilities.operators import make_tree, to_cylindric, Ryan_sampler
from Utilities.selectors_for_snap import select_snap, select_prefix

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
compton = 'Compton'
check = 'HiResNewAMR' 
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

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
    C = 2*np.min(dim_plane) #2 #np.min([not_toovercome, 2])
    condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
    # to be sure that you have points in the plane
    while len(mass_plane[condition]) == 0:
        print('No points found since the beginning, double C', np.round(C,2))
        C *= 2
        condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
    mass = mass_plane[condition]
    total_mass = np.sum(mass)
    while True:
        # update 
        step = 2*np.mean(dim_plane[condition]) 
        C += step
        condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
        # Check that you add new points
        if len(mass_plane[condition]) == len(mass):
            print(f'No new points added, increase C addinging {np.round(step,2)} to {np.round(C,2)}')
            print(f'len[condition]: {mass_plane[condition]}')
            C += step
        else:
            tocheck = r_plane[condition]-R0
            if tocheck.any()<0:
                C -= step
                print('overcome R0', C)
                break
            mass = mass_plane[condition]
            new_mass = np.sum(mass) 
            # old mass is > 95% of the new mass
            if np.logical_and(total_mass > 0.95 * new_mass, total_mass != new_mass): 
                break
            total_mass = new_mass
    return C

def find_radial_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, R0):
    """ Find the maxima density points in the radial plane for all the thetas in theta_arr 
    Parameters
    ----------
    x_data, y_data, z_data, dim_data, den_data : array
        X, Y, Z, coordinates, dimension and density of simualtion points.
    theta_arr : array
        Angles in radians.
    R0 : float
        Smoothing radius.
    Returns
    -------
    x_max, y_max, z_max : array
        X, Y, Z coordinates of the maximum density points in the radial plane of each angle in the sample.
    """
    x_max = np.zeros(len(theta_arr))
    y_max = np.zeros(len(theta_arr))
    z_max = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        # Exclude points inside the smoothing lenght and find radial plane
        condition_distance = np.sqrt(x_data**2 + y_data**2 + z_data**2) > R0 
        condition_Rplane = sec.radial_plane(x_data, y_data, dim_data, theta_arr[i])
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        x_plane, y_plane, z_plane, den_plane = sec.make_slices([x_data, y_data, z_data, den_data], condition_Rplane)
        # Find and save the maximum density point
        idx_max = np.argmax(den_plane) 
        x_max[i] = x_plane[idx_max]
        y_max[i] = y_plane[idx_max]
        z_max[i] = z_plane[idx_max]
        
        # if np.logical_and(alice == False, __name__ == '__main__'):
        #     if i == idx_forplot:
        #         r_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
        #         r_central = np.sqrt(x_max[i]**2 + y_max[i]**2 + z_max[i]**2)
        #         fig, ax1 = plt.subplots(1,1, figsize = (10,5))
        #         ax1.scatter(r_plane/Rt, den_plane, s = 10, c = 'k', label = 'Density')
        #         ax1.set_xlabel(r'R [$R_{\rm t}$]')
        #         ax1.set_ylabel(r'Density $[M_\odot/R_\odot^3]$')
        #         ax1.set_xlim(0.1, r_central/Rt+1)
        #         ax1.axvline(r_central/Rt, c = 'yellowgreen')
        #         ax1.set_title(r'$\theta$ = ' + f'{np.round(theta_arr[i],2)} rad', fontsize = 14)
        #         plt.show()

    return x_max, y_max, z_max    

def find_transverse_com(x_data, y_data, z_data, dim_data, den_data, mass_data, theta_arr, params, Rstar, all_iterations = False):
    """ Find the centres of mass in the transverse plane
    Parameters
    ----------
    x_data, y_data, z_data, dim_data, den_data, mass_data : array
        X, Y, Z, coordinates, dimension, density and mass of simualtion points.
    theta_arr : array
        Angles in radians.
    params : list
        List of parameters [Mbh, Rstar, mstar, beta].
    all_iterations : bool
        If True, returns (in the reversed order) all the iterations of the centers of mass and the radial maximum points.
    Returns
    -------
    thresh_cm: array
        Threshold for the transverse plane for each angle in the sample.
    x_cm, y_cm, z_cm, indeces_cm : array
        X, Y, Z coordinates, indices of x_data of the com points in the transverse plane of each angle in the sample.
    """
    Mbh, Rstar, mstar, beta = params
    Rt = orb.tidal_radius(Rstar, mstar, Mbh)
    R0 = 0.6 * (Rt/beta)
    apo = orb.apocentre(Rstar, mstar, Mbh, beta)
    # Cut a bit the data at the first step of everything for computational reasons
    cutting = np.logical_and(np.abs(z_data) < 100, np.abs(y_data) < 10*apo)
    x_cut, y_cut, z_cut, dim_cut, den_cut, mass_cut = \
        sec.make_slices([x_data, y_data, z_data, dim_data, den_data, mass_data], cutting)
    # Find the radial maximum points to have a first guess of the stream (necessary for the threshold and the tg)
    x_stream_rad, y_stream_rad, z_stream_rad = find_radial_maximum(x_cut, y_cut, z_cut, dim_cut, den_cut, theta_arr, R0)
    print('radial done', flush = True)

    # First iteration: find the center of mass of each transverse plane of the maxima-density stream
    x_cmTR = np.zeros(len(theta_arr))
    y_cmTR = np.zeros(len(theta_arr))
    z_cmTR = np.zeros(len(theta_arr))
    for idx in range(len(theta_arr)):
        # print(idx)
        # Find the transverse plane
        condition_T, x_T, _ = sec.transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_stream_rad, y_stream_rad, z_stream_rad, idx, Rstar, just_plane = True)
        x_plane, y_plane, z_plane, dim_plane, den_plane, mass_plane = \
            sec.make_slices([x_cut, y_cut, z_cut, dim_cut, den_cut, mass_cut], condition_T)
        # Cut the TZ plane to not keep points too far away.
        r_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
        thresh = get_threshold(x_T, z_plane, r_plane, mass_plane, dim_plane, R0) #8 * Rstar * (r_stream_rad[idx]/Rp)**(1/2)
        condition_x = np.abs(x_T) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        x_plane, y_plane, z_plane, mass_plane = \
            sec.make_slices([x_plane, y_plane, z_plane, mass_plane], condition)
        # Find the center of mass
        x_cmTR[idx] = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_cmTR[idx]= np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_cmTR[idx] = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
    print('Iteration radial-transverse done', flush = True)

    # Second iteration: find the center of mass of each transverse plane corresponding to COM stream
    x_cm = np.zeros(len(theta_arr))
    y_cm = np.zeros(len(theta_arr))
    z_cm = np.zeros(len(theta_arr))
    thresh_cm = np.zeros(len(theta_arr))
    for idx in range(len(theta_arr)):
        # print(idx)
        # Find the transverse plane
        condition_T, x_T, _ = sec.transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_cmTR, y_cmTR, z_cmTR, idx, Rstar, just_plane = True)
        x_plane, y_plane, z_plane, dim_plane, den_plane, mass_plane = \
            sec.make_slices([x_cut, y_cut, z_cut, dim_cut, den_cut, mass_cut], condition_T)
        # plot section at pericenter
        # if idx == np.argmin(np.abs(theta_arr)): 
        #     from matplotlib import colors
        #     fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,8))
        #     img = ax1.scatter(x_T, z_plane, c = den_plane, s = 10, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-13, vmax = 1e-6))
        #     cbar = plt.colorbar(img)
        #     cbar.set_label(r'Density $[M_\odot/R_\odot^3]$')
        #     ax1.set_ylabel(r'Z [$R_\odot$]')
        #     img = ax2.scatter(x_T, z_plane, c = mass_plane, s = 10, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-8))
        #     cbar = plt.colorbar(img)
        #     cbar.set_label(r'Cell mass $[R_\odot]$')
        #     for ax in [ax1, ax2]:
        #         ax.scatter(0,0, edgecolor= 'k', marker = 'o', facecolors='none', s=80)
        #         ax.set_xlim(-50, 30)
        #         ax.set_ylim(-10, 10)
        #         ax.set_xlabel(r'T [$R_\odot$]')
        #     plt.suptitle('(0,0) is the center of mass of the TZ plane of the max density point', fontsize = 14)
        #     plt.tight_layout()
        # Restrict the points to not keep points too far away.
        r_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
        thresh = get_threshold(x_T, z_plane, r_plane, mass_plane, dim_plane, R0) #8 * Rstar * (r_cmTR[idx]/Rp)**(1/2)
        condition_x = np.abs(x_T) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        x_plane, y_plane, z_plane, mass_plane = \
            sec.make_slices([x_plane, y_plane, z_plane, mass_plane], condition)
        # Find and save the center of mass
        x_cm[idx] = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_cm[idx]= np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_cm[idx] = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
        thresh_cm[idx] = thresh
    # Search in the tree the closest point to the center of mass
    # points = np.array([x_data, y_data, z_data]).T
    # tree = KDTree(points)
    # _, indeces_cm = tree.query(np.array([x_cm, y_cm, z_cm]).T, k=1)
    # indeces_cm = [ int(indeces_cm[i][0]) for i in range(len(indeces_cm))]
    if all_iterations:
        return x_cm, y_cm, z_cm, thresh_cm, x_cmTR, y_cmTR, z_cmTR, x_stream_rad, y_stream_rad, z_stream_rad
    else:
        return x_cm, y_cm, z_cm, thresh_cm


if __name__ == '__main__':
    theta_lim =  np.pi
    step = np.round((2*theta_lim)/200, 3)
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    # idx_forplot = 4 

    if alice:
        snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 
    else: 
        snaps = [238]
        # theta_arr = theta_arr[136:146]
    
    print('We are in res', check, flush = True)
    for i, snap in enumerate(snaps):
        print(f'Snap {snap}', flush = True)

        path = select_prefix(m, check, mstar, Rstar, beta, n, compton)
        if alice:
            path = f'{path}/snap_{snap}'
        else:
            path = f'{path}/{snap}'
        data = make_tree(path, snap, energy = False)
        X, Y, Z, Den, Mass, Vol = \
            data.X, data.Y, data.Z, data.Den, data.Mass, data.Vol
        cutden = Den >1e-19
        X, Y, Z, Den, Mass, Vol = \
            sec.make_slices([X, Y, Z, Den, Mass, Vol], cutden)
        R = np.sqrt(X**2 + Y**2 + Z**2)
        THETA, RADIUS_cyl = to_cylindric(X, Y)
        dim_cell = Vol**(1/3) 
        x_stream, y_stream, z_stream, thresh_cm = find_transverse_com(X, Y, Z, dim_cell, Den, Mass, theta_arr, params, Rstar)
        stream = [theta_arr, x_stream, y_stream, z_stream, thresh_cm]

        if save:
            np.save(f'{abspath}/data/{folder}/WH/stream/stream_{check}{snap}.npy', stream)
        
        if plot:
            fig, ax = plt.subplots(1, 1, figsize = (10, 5))
            ax.scatter(x_stream, y_stream, s = 1, c = 'k')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()