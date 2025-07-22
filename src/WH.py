""" Measures the stream width and heigh for all considered snaps. """
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
    compute = True
else:
    abspath = '/Users/paolamartire/shocks'
    import matplotlib.pyplot as plt
    from Utilities.basic_units import radians
    import matplotlib.colors as colors
    save = False
    compute = True

from datetime import datetime
import numpy as np
import numba
from scipy.optimize import brentq

import Utilities.prelude as prel
import Utilities.sections as sec
import src.orbits as orb
from Utilities.operators import make_tree, to_cylindric, Ryan_sampler, draw_line
from Utilities.selectors_for_snap import select_snap, select_prefix

#%%
## Parameters
#

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
check = 'HiResNewAMR'
compton = 'Compton'
if check not in ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']:
    folder = f'opacity_tests/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
else:
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

Mbh = 10**m
Rs = 2*prel.G*Mbh / prel.csol_cgs**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rp
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

##
# FUNCTIONS
##

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
        
        if np.logical_and(alice == False, __name__ == '__main__'):
            if i == idx_forplot:
                r_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
                r_central = np.sqrt(x_max[i]**2 + y_max[i]**2 + z_max[i]**2)
                fig, ax1 = plt.subplots(1,1, figsize = (10,5))
                ax1.scatter(r_plane/Rt, den_plane, s = 10, c = 'k', label = 'Density')
                ax1.set_xlabel(r'R [$R_{\rm t}$]')
                ax1.set_ylabel(r'Density $[M_\odot/R_\odot^3]$')
                ax1.set_xlim(0.1, r_central/Rt+1)
                ax1.axvline(r_central/Rt, c = 'yellowgreen')
                ax1.set_title(r'$\theta$ = ' + f'{np.round(theta_arr[i],2)} rad', fontsize = 14)
                plt.show()

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

def bound_mass(x, check_data, mass_data, m_thres):
    """ Function to use with root finding to find the coordinate threshold to respect the wanted mass enclosed.
    Parameters
    ----------
    x : float
        The threshold to find.
    check_data, mass_data : array
        The data to check the threshold against and the corresponding mass data.
    m_thres : float
        The threshold mass to respect.
    Returns
    -------
    float
        The difference between the total mass enclosed in the threshold and the wanted threshold mass.
    """
    condition = np.abs(check_data) < x # it's either x_T or Z
    mass = mass_data[condition]
    total_mass = np.sum(mass)
    return total_mass - m_thres

def find_single_boundaries(x_data, y_data, z_data, dim_data, mass_data, stream, idx, params, mass_percentage):
    """ Find the width and the height of the stream for a single theta 
    Parameters
    ----------
    x_data, y_data, z_data, dim_data, mass_data : array
        X, Y, Z, coordinates, dimension and mass of simualtion points.
    stream : array
        Stream data containing: theta, x, y, z coordinates, threshold.
    idx : int
        Index of the theta in the stream to consider.
    params : list
        List of parameters [Mbh, Rstar, mstar, beta].
    Returns
    -------
    indeces_boundary : array
        Indeces of the boundaries in the data. You have to use them to slice np.arange(len(x_data)).
    x_T_width : array
        X coordinates of the transverse plane boundaries.
    w_params : array
        Width parameters: [width, number of cells in width].
    h_params : array
        Height parameters: [height, number of cells in height].
    thresh : float  
        Threshold for the transverse plane.
    """
    Mbh, Rstar, mstar, beta = params
    Rt = Rstar * (Mbh/mstar)**(1/3)
    R0 = 0.6 * (Rt / beta) 
    indeces = np.arange(len(x_data))
    theta_arr, x_stream, y_stream, z_stream, thresh_stream = stream[0], stream[1], stream[2], stream[3], stream[4]
    # Find the transverse plane 
    condition_T, x_Tplane, _ = sec.transverse_plane(x_data, y_data, z_data, dim_data, x_stream, y_stream, z_stream, idx, Rstar, just_plane = True)
    x_plane, y_plane, z_plane, dim_plane, mass_plane, indeces_plane = \
        sec.make_slices([x_data, y_data, z_data, dim_data, mass_data, indeces], condition_T)
    r_spherical_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
    thresh = thresh_stream[idx]
    condition_x = np.abs(x_Tplane) < thresh
    condition_z = np.abs(z_plane) < thresh
    condition = condition_x & condition_z
    x_plane, x_Tplane, y_plane, z_plane, r_spherical_plane, dim_plane, mass_plane, indeces_plane = \
        sec.make_slices([x_plane, x_Tplane, y_plane, z_plane, r_spherical_plane, dim_plane, mass_plane, indeces_plane], condition)
    if np.min(r_spherical_plane)< R0:
        print(f'The threshold to cut the TZ plane in width is too broad: you overcome R0 at angle #{theta_arr[idx]} of the stream')
    
    mass_to_reach = (mass_percentage/2) * np.sum(mass_plane) #25% for each quarter
    # Find the threshold for x
    x_Tplanepositive, mass_planepositive, indices_planepositive = \
        x_Tplane[x_Tplane>=0], mass_plane[x_Tplane>=0], indeces_plane[x_Tplane>=0]
    x_Tplanenegative, mass_planenegative, indices_planenegative = \
        x_Tplane[x_Tplane<0], mass_plane[x_Tplane<0], indeces_plane[x_Tplane<0]

    try:
        contourTpositive = brentq(bound_mass, 0, thresh, args=(x_Tplanepositive, mass_planepositive, mass_to_reach))
        condition_contourTpositive = np.abs(x_Tplanepositive) < contourTpositive
        x_T_contourTpositive, indeces_contourTpositive = sec.make_slices([x_Tplanepositive, indices_planepositive], condition_contourTpositive)
    except ValueError as e:
        return None, None, None, None, None
    try:
        contourTnegative = brentq(bound_mass, 0, thresh, args=(x_Tplanenegative, mass_planenegative, mass_to_reach))
        condition_contourTnegative = np.abs(x_Tplanenegative) < contourTnegative
        x_T_contourTnegative, indeces_contourTnegative = sec.make_slices([x_Tplanenegative, indices_planenegative], condition_contourTnegative)
    except ValueError as e:
        return None, None, None, None, None
    # contourT = brentq(bound_mass, 0, thresh, args=(x_Tplane, mass_plane, mass_to_reach))
    # condition_contourT = np.abs(x_Tplane) < contourT
    # x_T_contourT, indeces_contourT = make_slices([x_Tplane, indeces_plane], condition_contourT)

    idx_before = np.argmin(x_T_contourTnegative)
    idx_after = np.argmax(x_T_contourTpositive)
    x_T_low, idx_low = x_T_contourTnegative[idx_before], indeces_contourTnegative[idx_before]
    x_T_up, idx_up = x_T_contourTpositive[idx_after], indeces_contourTpositive[idx_after]
    width = x_T_up - x_T_low
    # width = np.max([width, dim_stream[idx]]) # to avoid 0 width

    # Find the threshold for z 
    z_planepositive, mass_planepositive, indices_planepositive = \
        z_plane[z_plane>=0], mass_plane[z_plane>=0], indeces_plane[z_plane>=0]
    z_planenegative, mass_planenegative, indices_planenegative = \
        z_plane[z_plane<0], mass_plane[z_plane<0], indeces_plane[z_plane<0]

    try: 
        contourZpositive = brentq(bound_mass, 0, thresh, args=(z_planepositive, mass_planepositive, mass_to_reach))
        condition_contourZpositive = np.abs(z_planepositive) < contourZpositive
        z_contourZpositive, indeces_contourZpositive = sec.make_slices([z_planepositive, indices_planepositive], condition_contourZpositive)
    except ValueError as e:
        return None, None, None, None, None
    try:
        contourZnegative = brentq(bound_mass, 0, thresh, args=(z_planenegative, mass_planenegative, mass_to_reach))
        condition_contourZnegative = np.abs(z_planenegative) < contourZnegative
        z_contourZnegative, indeces_contourZnegative = sec.make_slices([z_planenegative, indices_planenegative], condition_contourZnegative)
    except ValueError as e:
        return None, None, None, None, None
    
    # contourZ = brentq(bound_mass, 0, thresh, args=(z_plane, mass_plane, mass_to_reach))
    # condition_contourZ = np.abs(z_plane) < contourZ
    # z_contourZ, indeces_contourZ = sec.make_slices([z_plane, indeces_plane], condition_contourZ)
    
    idx_before = np.argmin(z_contourZnegative)
    idx_after = np.argmax(z_contourZpositive)
    z_low, idx_low_h = z_contourZnegative[idx_before], indeces_contourZnegative[idx_before]
    z_up, idx_up_h = z_contourZpositive[idx_after], indeces_contourZpositive[idx_after]
    height = z_up - z_low
    # height = np.max([height, dim_stream[idx]]) # to avoid 0 height

    # Compute the number of cells in width and height using the cells in the rectangle
    cell_contour = np.logical_and(x_Tplane >= x_T_low, np.logical_and(x_Tplane <= x_T_up, np.logical_and(z_plane >= z_low, z_plane <= z_up)))
    dim_cell_mean = np.mean(dim_plane[cell_contour])
    ncells_w = np.round(width/dim_cell_mean, 0) # round to the nearest integer
    ncells_h = np.round(height/dim_cell_mean, 0) # round to the nearest integer

    indeces_boundary = np.array([idx_low, idx_up, idx_low_h, idx_up_h]).astype(int)
    w_params = np.array([width, ncells_w])
    h_params = np.array([height, ncells_h])

    if np.logical_and(alice == False, __name__ == '__main__'):
        if idx == idx_forplot: #np.abs(theta_arr[idx] - 1.5) == np.min(np.abs(theta_arr - 1.5)):
            # Compute weighted histogram data
            counts_T, bin_edges_T = np.histogram(x_Tplane, bins=50, weights=mass_plane)
            counts_Z, bin_edges_Z = np.histogram(z_plane, bins=50, weights=mass_plane)
            # Compute bin centers for plotting
            bin_centers_T = 0.5 * (bin_edges_T[:-1] + bin_edges_T[1:])
            bin_centers_Z = 0.5 * (bin_edges_Z[:-1] + bin_edges_Z[1:])
            # Plot as a continuous line
            fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize = (12,12))
            ax1.plot(bin_centers_T, counts_T, c = 'k')
            ax1.axvline(x=x_T_low, color='grey', linestyle = '--', label = 'Stream width')
            ax1.axvline(x=x_T_up, color='grey', linestyle = '--')
            # find standard deviation of the counts
            # ax1.axhline(np.percentile(counts_T, 50), color='red', linestyle = '--', label = 'std counts')
            # ax1.axhline(np.percentile(counts_T, 100-(mass_percentage*100)/2), color='red', linestyle = '--', label = 'std counts')
            ax1.legend()            

            ax1.set_ylabel(r'Mass [$M_\odot$]')
            ax1.set_xlabel(r'T [$R_\odot$]')

            img = ax2.scatter(x_Tplane, z_plane, c = mass_plane, s = 10, cmap = 'rainbow', norm = colors.LogNorm(vmin = 4e-10, vmax = 4e-8))
            cbar = plt.colorbar(img)
            cbar.set_label(r'Mass [$M_\odot$]')
            ax2.set_xlabel(r'T [$R_\odot$]')
            ax2.set_ylabel(r'Z [$R_\odot$]')
            ax2.axvline(x=x_T_low, color='grey', linestyle = '--')
            ax2.axvline(x=x_T_up, color='grey', linestyle = '--')
            ax2.axhline(y=z_low, color='grey', linestyle = '--')
            ax2.axhline(y=z_up, color='grey', linestyle = '--')
            #ticks
            original_ticks = ax1.get_xticks()
            mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
            new_ticks = np.concatenate((original_ticks, mid_ticks))
            for ax in [ax1, ax2]:
                ax.set_xticks(new_ticks)
                ax.set_xticks(new_ticks)
                labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]
                ax.set_xticklabels(labels)
                ax.set_xlim(x_T_low-2, x_T_up+2)
            ax2.set_ylim(z_low-1, z_up+1)

            ax3.plot(bin_centers_Z, counts_Z,  c = 'k')
            ax3.axvline(x=z_low, color='grey', linestyle = '--')
            ax3.axvline(x=z_up, color='grey', linestyle = '--')
            ax3.set_ylabel(r'Mass [$M_\odot$]')
            ax3.set_xlabel(r'Z [$R_\odot$]')    
            original_ticks = ax3.get_xticks()
            mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
            new_ticks = np.concatenate((original_ticks, mid_ticks))
            ax3.set_xticks(new_ticks)
            # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]
            # ax3.set_xticklabels(labels)
            ax3.set_xlim(z_low-1.5, z_up+1.5)

            for ax in [ax1, ax2, ax3]:
                ax.tick_params(axis='both', which='major', length=10)
            plt.tight_layout()
            plt.suptitle(r'Width and height at $\theta$' + f'= {np.round(theta_arr[idx], 2)} rad', fontsize = 16)

    return indeces_boundary, w_params, h_params

def follow_the_stream(x_data, y_data, z_data, dim_data, mass_data, stream, params, mass_percentage = 0.8):
    """ Find width and height all along the stream """
    # Find the stream (load it)
    theta_arr = stream[0]
    # Find the boundaries for each theta
    theta_wh = []
    indeces_boundary = []
    w_params = []
    h_params = []
    for i in range(len(theta_arr)):
        indeces_boundary_i, w_params_i, h_params_i = \
            find_single_boundaries(x_data, y_data, z_data, dim_data, mass_data, stream, i, params, mass_percentage)
        if indeces_boundary_i is None:
            print(f'Skipping theta {theta_arr[i]} due to ValueError in boundary finding.', flush = True)
            continue
        else:
            theta_wh.append(theta_arr[i])
            indeces_boundary.append(indeces_boundary_i)
            w_params.append(w_params_i)
            h_params.append(h_params_i)
    indeces_boundary = np.array(indeces_boundary).astype(int)
    w_params = np.transpose(np.array(w_params)) # line 1: width, line 2: ncells
    h_params = np.transpose(np.array(h_params)) # line 1: height, line 2: ncells
    return theta_wh, indeces_boundary, w_params, h_params

if __name__ == '__main__':
    from Utilities.operators import from_cylindric
    theta_lim =  np.pi
    step = 0.02
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    if not alice:
        theta_arr = theta_arr[136:146]
        idx_forplot = 4

    if alice:
        snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 
    else: 
        check = 'NewAMR'
        snaps = [238]
    
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
        theta_wh, indeces_boundary, w_params, h_params  = follow_the_stream(X, Y, Z, dim_cell, Mass, stream, params = params, mass_percentage = 0.8)
    
        if save:
            np.save(f'{abspath}/data/{folder}/WH/stream_{check}{snap}.npy', stream)
            with open(f'{abspath}/data/{folder}/WH/wh_{check}{snap}.txt','w') as file:
                # if file exist, save theta and date of execution
                file.write(f'# theta, done on {datetime.now()} \n')
                file.write((' '.join(map(str, theta_wh)) + '\n'))
                file.write(f'# Width \n')
                file.write((' '.join(map(str, w_params[0])) + '\n'))
                file.write(f'# Ncells width\n')
                file.write((' '.join(map(str, w_params[1])) + '\n'))
                file.write(f'# Height \n')
                file.write((' '.join(map(str, h_params[0])) + '\n'))
                file.write(f'# Ncells height \n')
                file.write((' '.join(map(str, h_params[1])) + '\n'))
            np.save(f'{abspath}/data/{folder}/WH/indeces_boundary_{check}{snap}.npy', indeces_boundary)
        
        if alice == False:
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15, 5))
            ax1.plot(theta_wh*radians, w_params[0], c = 'k', label = 'isodensity (80\% mass enclosed)')
            # print(f'N cells width and H: {w_params[1][idx_forplot]} {h_params_sp[1][idx_forplot]}')
            ax1.legend(fontsize = 16)
            ax1.set_ylabel(r'Width [$R_\odot$]')
            ax1.set_ylim(1.1, 10)
            ax2.plot(theta_wh*radians, h_params[0], c = 'k')
            ax2.set_ylabel(r'Height [$R_\odot$]')
            ax2.set_ylim(.2, 10)
            for ax in [ax1, ax2]:
                ax.set_yscale('log')
                ax.set_xlabel(r'$\theta$')
                ax.set_xlim(-2.5, 2.5)
                ax.tick_params(axis='both', which='major', length = 8, width = 1.2)
                ax.tick_params(axis='both', which='minor', length = 4, width = 1)
            plt.tight_layout()