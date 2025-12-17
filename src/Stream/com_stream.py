""" If alice: Find the stream as line connecting COMs.
If not alice: plot on a scatter plot. """
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
from sklearn.neighbors import KDTree
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
compton = 'Compton'
check = 'HiResNewAMR' 
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

@numba.njit
def get_threshold(t_plane, z_plane, r_plane, mass_plane, dim_plane, R0):
    """ Find the T-Z threshold to cut the transverse plane (as a square) in width and height. 
    This is necessary to avoid to take points too far away from the center of mass (eg in 2 different semiplanes).
    Idea: expand the boundaries from the radial maximum the mass enclosed  doens't change much from the previous step.
    Parameters
    ----------
    t_plane: array
        T coordinates of points in the TZ plane of the radial maximum.
    z_plane, r_plane, mass_plane, dim_plane : array 
        Z and radial (spherical) coordinates in the cartesian reference frame, mass and dimension of points in the TZ plane.
    R0 : float
        Smoothing radius.
    Returns
    -------
    C : float
        The (upper) threshold for t and z.
    """
    # First guess of C and find the mass enclosed in the initial boundaries
    C = 2 * np.min(dim_plane) 
    condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
    # to be sure that you have points in the plane
    while len(mass_plane[condition]) == 0:
        # print(f'No points found since the beginning, inside C= {np.round(C,2)}. I double C.', flush = True)
        C *= 2
        condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
    mass = mass_plane[condition]
    total_mass = np.sum(mass)
    while True:
        # expand the boundaries 
        step = 2*np.mean(dim_plane[condition]) 
        C += step 
        condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
        # Check that you add new points
        # if len(mass_plane[condition]) == len(mass):
            # print(f'No new points added, increase C adding {np.round(step,2)} to {np.round(C,2)}', flush = True)
            # C += step

        if len(mass_plane[condition]) != len(mass):
            tocheck = r_plane[condition]-R0
            if tocheck.any()<0:
                C -= step
                # print('overcome R0, I stop', flush = True)
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
        #     r_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
        #     r_central = np.sqrt(x_max[i]**2 + y_max[i]**2 + z_max[i]**2)
        #     fig, ax1 = plt.subplots(1,1, figsize = (10,5))
        #     ax1.scatter(r_plane/Rt, den_plane, s = 10, c = 'k', label = 'Density')
        #     ax1.set_xlabel(r'R [$R_{\rm t}$]')
        #     ax1.set_ylabel(r'Density $[M_\odot/R_\odot^3]$')
        #     ax1.set_xlim(0.1, r_central/Rt+1)
        #     ax1.axvline(r_central/Rt, c = 'yellowgreen')
        #     ax1.set_title(r'$\theta$ = ' + f'{np.round(theta_arr[i],2)} rad', fontsize = 14)
        #     plt.show()

    return x_max, y_max, z_max    

def find_transverse_com(x_data, y_data, z_data, dim_data, den_data, mass_data, theta_arr, all_iterations = False):
    """ Find the centres of mass that defines the stream.
    Parameters
    ----------
    x_data, y_data, z_data, dim_data, den_data, mass_data : array
        X, Y, Z, coordinates, dimension, density and mass of simualtion points.
    theta_arr : array
        Angles in radians.
    Rstar : float
        Stellar radius.
    all_iterations : bool
        If True, returns (in the reversed order) all the iterations of the centers of mass and the radial maximum points.
    Returns
    -------
    x_cm, y_cm, z_cm : array
        X, Y, Z coordinates of the com points in the transverse plane of each angle in the sample. 
        NB: they don't correspond to simulation points
    thresh_cm: array
        Threshold for the transverse plane for each angle in the sample.
    if all_iterations:
        x_cmTR, y_cmTR, z_cmTR : array
            X, Y, Z coordinates of the com points in the transverse plane of each radial maximum point.
        x_stream_rad, y_stream_rad, z_stream_rad : array
            X, Y, Z coordinates of the radial maximum points of the stream.
    """ 
    global R0, apo, Rt, Rstar
    # Cut a bit the data at the first step of everything for computational reasons
    cutting = np.logical_and(np.abs(z_data) < apo, np.abs(y_data) < 2 * apo)
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
        # Find the transverse plane
        condition_T, x_T, _ = sec.transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_stream_rad, y_stream_rad, z_stream_rad, idx, Rstar, just_plane = True)
        x_plane, y_plane, z_plane, dim_plane, mass_plane = \
            sec.make_slices([x_cut, y_cut, z_cut, dim_cut, mass_cut], condition_T)
        # Cut the TZ plane to not keep points too far away.
        r_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
        thresh = get_threshold(x_T, z_plane, r_plane, mass_plane, dim_plane, R0) 
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
        # Find the transverse plane
        condition_T, x_T, _ = sec.transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_cmTR, y_cmTR, z_cmTR, idx, Rstar, just_plane = True)
        x_plane, y_plane, z_plane, dim_plane, mass_plane = \
            sec.make_slices([x_cut, y_cut, z_cut, dim_cut, mass_cut], condition_T)
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
        thresh = get_threshold(x_T, z_plane, r_plane, mass_plane, dim_plane, R0) 
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
    points = np.array([x_data, y_data, z_data]).T
    tree = KDTree(points)
    _, indeces_cm = tree.query(np.array([x_cm, y_cm, z_cm]).T, k=1)
    indeces_cm = [ int(indeces_cm[i][0]) for i in range(len(indeces_cm))]
    if all_iterations:
        return x_cm, y_cm, z_cm, thresh_cm, x_cmTR, y_cmTR, z_cmTR, x_stream_rad, y_stream_rad, z_stream_rad
    else:
        return thresh_cm, indeces_cm


if __name__ == '__main__':
    if alice:
        theta_lim =  np.pi
        step = np.round((2*theta_lim)/200, 3)
        theta_init = np.arange(-theta_lim, theta_lim, step)
        theta_arr = Ryan_sampler(theta_init)
        snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 
    else: 
        snaps = [80]
        # theta_arr = theta_arr[136:146]
    
    print('We are in res', check, flush = True)
    for i, snap in enumerate(snaps):
        print(f'Snap {snap}', flush = True)
        
        if alice:
            path = select_prefix(m, check, mstar, Rstar, beta, n, compton)
            path = f'{path}/snap_{snap}'
            data = make_tree(path, snap, energy = False)
            X, Y, Z, VX, VY, VZ, Den, Mass, Vol = \
                data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Den, data.Mass, data.Vol
            cutden = Den >1e-19
            X, Y, Z, VX, VY, VZ, Den, Mass, Vol = \
                sec.make_slices([X, Y, Z, VX, VY, VZ, Den, Mass, Vol], cutden)
            dim_cell = Vol**(1/3) 
            thresh_cm, indices_cm = find_transverse_com(X, Y, Z, dim_cell, Den, Mass, theta_arr)
            x_cm, y_cm, z_cm, vx_cm, vy_cm, vz_cm, den_cm, mass_cm = \
                X[indices_cm], Y[indices_cm], Z[indices_cm], \
                    VX[indices_cm], VY[indices_cm], VZ[indices_cm], \
                        Den[indices_cm], Mass[indices_cm]
            com = {
                'theta_arr': theta_arr,
                'thresh_cm': thresh_cm,
                'x_cm': x_cm,
                'y_cm': y_cm,
                'z_cm': z_cm,
                'vx_cm': vx_cm,
                'vy_cm': vy_cm,
                'vz_cm': vz_cm,
                'den_cm': den_cm,
                'mass_cm': mass_cm
            } 

            np.savez(f'{abspath}/data/{folder}/WH/stream/stream_{check}{snap}.npz', **com)

        else:
            data = np.load(f'{abspath}/data/{folder}/WH/stream/stream_{check}{snap}.npz')
            theta_arr, x_stream, y_stream, z_stream, thresh_cm = \
                data['theta_arr'], data['x_cm'], data['y_cm'], data['z_cm'], data['thresh_cm']
            # print(len(theta_arr))
            snaps, times = np.loadtxt(f'{abspath}/data/{folder}/slices/z/z0_time.txt')
            slice_data = np.load(f'{abspath}/data/{folder}/slices/z/z0slice_{snap}.npy')
            x_slice, y_slice, z_slice, dim_slice, den_slice = \
                slice_data[0], slice_data[1], slice_data[2], slice_data[3], slice_data[4]
            mass_slice = den_slice * dim_slice**3
            fig, ax = plt.subplots(1, 1, figsize = (14, 7))
            img = ax.scatter(x_slice/Rt, y_slice/Rt, s = 1, c = mass_slice, cmap = 'rainbow', norm = colors.LogNorm(vmin = np.percentile(mass_slice, 5), vmax = np.percentile(mass_slice, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'Mass [$M_\odot$]', fontsize = 16)
            ax.plot(x_stream/Rt, y_stream/Rt, c = 'k')
            ax.set_xlabel(r'X [$r_{\rm t}$]')
            ax.set_ylabel(r'Y [$r_{\rm t}$]')
            ax.set_xlim(-30, 2)
            ax.set_ylim(-7, 7)
            ax.set_title(f't = {np.round(times[np.argmin(np.abs(snaps-snap))], 2)}' + r'$t_{\rm fb}$', fontsize = 16)
            plt.show()