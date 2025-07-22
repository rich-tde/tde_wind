""" Measures the stream width and heigh for all considered snaps. """
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    from Utilities.selectors_for_snap import select_snap
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
from Utilities.operators import make_tree, Ryan_sampler
from Utilities.selectors_for_snap import select_prefix

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
check = 'NewAMR'
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

def bound_mass_density(density_threshold, density_data, mass_data, m_thres):
    """
    Function to use with root finding to find the density threshold to respect the wanted mass enclosed.
    
    Parameters
    ----------
    density_threshold : float
        The density threshold to find.
    density_data : array
        The density data to check the threshold against.
    mass_data : array
        The corresponding mass data.
    m_thres : float
        The threshold mass to respect.
        
    Returns
    -------
    float
        The difference between the total mass enclosed in the density threshold and the wanted threshold mass.
    """
    condition = density_data >= density_threshold  # Points with density >= threshold are "inside"
    mass = mass_data[condition]
    total_mass = np.sum(mass)
    return total_mass - m_thres

def find_single_boundaries_isodensity(x_data, y_data, z_data, dim_data, density_data, mass_data, stream, idx, params, mass_percentage):
    """
    Find the boundaries using isodensity contours for a single theta.
    
    Parameters
    ----------
    x_data, y_data, z_data, dim_data, density_data, mass_data : array
        X, Y, Z coordinates, dimension, density, and mass of simulation points.
    stream : array
        Stream data containing: theta, x, y, z coordinates, threshold.
    idx : int
        Index of the theta in the stream to consider.
    params : list
        List of parameters [Mbh, Rstar, mstar, beta].
    mass_percentage : float
        Target mass percentage to enclose.
        
    Returns
    -------
    density_threshold : float
        The density threshold that encloses the target mass percentage.
    indeces_enclosed : array
        Indices of points inside the isodensity contour.
    contour_stats : dict
        Dictionary containing statistics about the contour.
    """
    Mbh, Rstar, mstar, beta = params
    Rt = orb.tidal_radius(Rstar, mstar, Mbh)
    R0 = 0.6 * (Rt / beta)
    indeces = np.arange(len(x_data))
    theta_arr, x_stream, y_stream, z_stream, thresh_stream = stream[0], stream[1], stream[2], stream[3], stream[4]
    # Find the transverse plane
    condition_T, x_Tplane, _ = sec.transverse_plane(x_data, y_data, z_data, dim_data, x_stream, y_stream, z_stream, idx, Rstar, just_plane=True)
    x_plane, y_plane, z_plane, dim_plane, density_plane, mass_plane, indeces_plane = \
        sec.make_slices([x_data, y_data, z_data, dim_data, density_data, mass_data, indeces], condition_T)
    
    r_spherical_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
    thresh = thresh_stream[idx]
    condition_x = np.abs(x_Tplane) < thresh
    condition_z = np.abs(z_plane) < thresh
    condition = condition_x & condition_z
    
    x_plane, x_Tplane, y_plane, z_plane, r_spherical_plane, dim_plane, density_plane, mass_plane, indeces_plane = \
        sec.make_slices([x_plane, x_Tplane, y_plane, z_plane, r_spherical_plane, dim_plane, density_plane, mass_plane, indeces_plane], condition)
    
    if np.min(r_spherical_plane) < R0:
        print(f'The threshold to cut the TZ plane in width is too broad: you overcome R0 at angle #{theta_arr[idx]} of the stream')
    
    # Calculate target mass
    total_mass = np.sum(mass_plane)
    mass_to_reach = mass_percentage * total_mass
    
    # Find density range for root finding
    min_density = np.min(density_plane)
    max_density = np.max(density_plane)
    
    # Use a slightly higher minimum to ensure we don't include all points
    density_range_min = min_density + 0.001 * (max_density - min_density)
    
    try:
        # Find the density threshold that encloses the target mass
        density_threshold = brentq(bound_mass_density, 
                                 density_range_min, max_density,
                                 args=(density_plane, mass_plane, mass_to_reach))
        
        # Get condition for points inside the contour
        condition_enclosed = density_plane >= density_threshold
        indeces_enclosed = indeces_plane[condition_enclosed]
        
        # Calculate statistics for the enclosed region
        x_T_enclosed = x_Tplane[condition_enclosed]
        z_enclosed = z_plane[condition_enclosed]
        
        # Calculate bounding box of the enclosed region
        idx_before = np.argmin(x_T_enclosed)
        idx_after = np.argmax(x_T_enclosed)
        xT_min, xT_max = x_T_enclosed[idx_before], x_T_enclosed[idx_after]
        idx_low, idx_up = indeces_enclosed[idx_before], indeces_enclosed[idx_after]
        idx_before = np.argmin(z_enclosed)
        idx_after = np.argmax(z_enclosed)
        z_min, z_max = z_enclosed[idx_before], z_enclosed[idx_after]
        idx_low_h, idx_up_h = indeces_enclosed[idx_before], indeces_enclosed[idx_after]
        indeces_boundary = np.array([idx_low, idx_up, idx_low_h, idx_up_h]).astype(int)

        width = xT_max - xT_min
        height = z_max - z_min
        
        # Calculate effective dimensions using mean cell size in enclosed region
        dim_cell_mean = np.mean(dim_plane[condition_enclosed])
        ncells_w = np.round(width / dim_cell_mean, 0)
        ncells_h = np.round(height / dim_cell_mean, 0)
        
        # Create contour statistics
        contour_stats = {
            'density_threshold': density_threshold,
            'x_bounds': [xT_min, xT_max],
            'width': width,
            'ncells_w': ncells_w,
            'z_bounds': [z_min, z_max],
            'height': height,
            'ncells_h': ncells_h,
            'indeces_boundary': indeces_boundary, 
            'mass_fraction': np.sum(mass_plane[condition_enclosed]) / total_mass if not alice else None,
            # 'total_mass': total_mass,
            # 'n_points_enclosed': len(indeces_enclosed),
            # 'n_points_total': len(indeces_plane)
        } 
        
        # Optional: Create visualization if this is the theta for plotting
        if np.logical_and(alice == False, idx == idx_forplot):
            plot_isodensity_results(x_Tplane, z_plane, mass_plane, dim_plane,
                                    condition_enclosed, contour_stats, 
                                    theta_arr[idx], x_data, y_data, indeces_enclosed, 
                                    indeces_plane)
        
        return density_threshold, indeces_enclosed, contour_stats
        
    except ValueError as e:
        print(f"Error finding isodensity contour at theta {theta_arr[idx]}: {e}")
        return None, None, None, None

def plot_isodensity_results(x_Tplane, z_plane, mass_plane, dim_plane,
                           condition_enclosed, contour_stats, 
                           theta_val, x_data, y_data, indeces_enclosed, 
                           indeces_plane):
    """
    Plot the isodensity contour results similar to the original plotting style.
    """
    # Get global variables (assuming they exist in your original code)
    # apo = globals().get('apo', 1.0)  # Default to 1 if not found
    x_T_low, x_T_up = contour_stats['x_bounds'][0], contour_stats['x_bounds'][1]
    z_low, z_up = contour_stats['z_bounds'][0], contour_stats['z_bounds'][1]
    idx_x_low, idx_x_up = contour_stats['indeces_boundary'][0], contour_stats['indeces_boundary'][1]
    x_low, x_up  = x_data[idx_x_low], x_data[idx_x_up]
    y_low, y_up  = y_data[idx_x_low], y_data[idx_x_up]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot 1: Mass histogram in T direction
    counts_T, bin_edges_T = np.histogram(x_Tplane, bins=50, weights=mass_plane)
    bin_centers_T = 0.5 * (bin_edges_T[:-1] + bin_edges_T[1:])
    ax1.plot(bin_centers_T, counts_T, c='k', label='Mass distribution')
    ax1.axvline(x=x_T_low, color='k', linestyle='--', label='Isodensity boundary')
    ax1.axvline(x=x_T_up, color='k', linestyle='--')
    ax1.legend()
    ax1.set_ylabel(r'Mass [$R_\odot$]')
    ax1.set_xlabel(r'T [$R_\odot$]')
    
    # Plot 2: Scatter plot with isodensity contour
    # Color points by whether they're inside or outside the contour
    ax2.scatter(x_Tplane, z_plane, s=10, edgecolor='k', facecolor='none') 
    img = ax2.scatter(x_Tplane[condition_enclosed], z_plane[condition_enclosed], c=mass_plane[condition_enclosed], cmap = 'rainbow',s = 12, norm =colors.LogNorm(vmin=np.percentile(mass_plane, 80), vmax=np.max(mass_plane)))
    plt.colorbar(img, ax=ax2, label=r'Mass [$M_\odot$]')
    ax2.axvline(x=x_T_low, color='k', linestyle='--', linewidth=2)
    ax2.axvline(x=x_T_up, color='k', linestyle='--', linewidth=2)
    ax2.axhline(y=z_low, color='k', linestyle='--', linewidth=2)
    ax2.axhline(y=z_up, color='k', linestyle='--', linewidth=2)
    ax2.set_xlabel(r'T [$R_\odot$]')
    ax2.set_ylabel(r'Z [$R_\odot$]')
    ax2.set_ylim(z_low-1, z_up+1)
    
    original_ticks = ax1.get_xticks()
    mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
    new_ticks = np.concatenate((original_ticks, mid_ticks))
    for ax in [ax1, ax2]:
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]
        ax.set_xticklabels(labels)
        ax.set_xlim(x_T_low-1, x_T_up+1)

    # Plot 3: Mass histogram in Z direction
    counts_Z, bin_edges_Z = np.histogram(z_plane, bins=50, weights=mass_plane)
    bin_centers_Z = 0.5 * (bin_edges_Z[:-1] + bin_edges_Z[1:])
    ax3.plot(bin_centers_Z, counts_Z, c='k')
    ax3.axvline(x=z_low, color='k', linestyle='--')
    ax3.axvline(x=z_up, color='k', linestyle='--')
    ax3.set_ylabel(r'Mass [$R_\odot$]')
    ax3.set_xlabel(r'Z [$R_\odot$]')
    original_ticks = ax3.get_xticks()
    mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
    new_ticks = np.concatenate((original_ticks, mid_ticks))
    ax3.set_xticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]
    ax3.set_xticklabels(labels)
    ax3.set_xlim(z_low-1, z_up+1)
    
    # Plot 4: Orbital plane view
    mid = np.abs(z_plane) < dim_plane
    if len(indeces_enclosed) > 0:
        # Map back to original coordinates
        x_enclosed = x_data[indeces_enclosed]
        y_enclosed = y_data[indeces_enclosed]
        mass_enclosed = mass_plane[condition_enclosed]
        z_enclosed = z_plane[condition_enclosed]
        dim_enclosed = dim_plane[condition_enclosed]
        mid_enclosed = np.abs(z_enclosed) < dim_enclosed
         
        vmin = np.percentile(mass_plane, 80)
        vmax = np.max(mass_plane)
        
        # Plot all points in the plane
        img = ax4.scatter(x_data[indeces_plane][mid], y_data[indeces_plane][mid], 
                         s=10, c=mass_plane[mid], cmap='rainbow', 
                         norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        
        # Highlight enclosed points
        ax4.scatter(x_enclosed[mid_enclosed], y_enclosed[mid_enclosed], 
                   s=20, c=mass_enclosed[mid_enclosed], edgecolor = 'k', cmap='rainbow', norm=colors.LogNorm(vmin=vmin, vmax=vmax), label='Enclosed points')
        
        cbar = plt.colorbar(img, ax=ax4, label=r'Mass [M$_\odot$]')
        ax4.scatter(0, 0, c='k', s=10)
        ax4.legend()
        ax4.set_xlabel(r'X [$R_\odot$]')
        ax4.set_ylabel(r'Y [$R_\odot$]')
        ax4.set_xlim(x_low-1, x_up+1)
        ax4.set_ylim(y_low-1, y_up+1)
        ax4.set_title('Orbital plane')
    
    # Add text box with statistics
    stats_text = f"Width: {contour_stats['width']:.2f}\nHeight: {contour_stats['height']:.2f}\n"
    stats_text += f"Cells W: {contour_stats['ncells_w']:.0f}\nCells H: {contour_stats['ncells_h']:.0f}\n"
    stats_text += f"Mass enclosed : {100*contour_stats['mass_fraction']:.0f}\% of trans. plane"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle(rf'$\theta$ = {np.round(theta_val, 2)} rad', fontsize=16)
    plt.show()

def follow_the_stream_isodensity(x_data, y_data, z_data, dim_data, density_data, mass_data, stream, params, mass_percentage=0.8):
    """
    Find isodensity contour boundaries all along the stream.
    
    Parameters
    ----------
    x_data, y_data, z_data, dim_data, density_data, mass_data : array
        Simulation data arrays.
    stream : array
        Stream data containing: theta, x, y, z coordinates, threshold.
    params : list
        List of parameters [Mbh, Rstar, mstar, beta].
    mass_percentage : float
        Target mass percentage to enclose (default 0.8 for 80%).
        
    Returns
    -------
    theta_wh : array
        Theta values where boundaries were successfully found.
    density_thresholds : array
        Density thresholds for each theta.
    indeces_enclosed : list
        List of indices arrays for enclosed points at each theta.
    contour_stats : list
        List of contour statistics dictionaries for each theta.
    """
    theta_arr = stream[0]
    
    # Initialize output arrays
    theta_wh = []
    density_thresholds = []
    indeces_enclosed = []
    contour_stats = []
    
    for i in range(len(theta_arr)):
        density_threshold, indeces_enclosed_i, contour_stats_i = \
            find_single_boundaries_isodensity(x_data, y_data, z_data, dim_data, density_data, mass_data, stream, i, params, mass_percentage)
        
        if density_threshold is None:
            print(f'Skipping theta {theta_arr[i]} due to ValueError in boundary finding.', flush=True)
            continue
        else:
            theta_wh.append(theta_arr[i])
            density_thresholds.append(density_threshold)
            indeces_enclosed.append(indeces_enclosed_i)
            contour_stats.append(contour_stats_i)
    
    return theta_wh, np.array(density_thresholds), indeces_enclosed, contour_stats

if __name__ == '__main__':
    snap = 238
    print(f'We are in folder {folder}, snap {snap}')
    theta_lim =  np.pi
    step = 0.02
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    theta_arr = theta_arr[136:146]  # Take every second element for better performance
    idx_forplot = 2
    
    path = select_prefix(m, check, mstar, Rstar, beta, n, compton)

    if alice:
        snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 
    else: 
        snaps = [238]
    for i, snap in enumerate(snaps):
        print(f'Snap {snap}', flush = True)

        path = select_prefix(m, check, mstar, Rstar, beta, n, compton)
        if alice:
            path = f'{path}/snap_{snap}'
        else:
            path = f'{path}/{snap}'

        tfb = np.loadtxt(f'{abspath}/TDE/{folder}/{snap}/tfb_{snap}.txt')
        data = make_tree(path, snap, energy = False)
        X, Y, Z, Den, Mass, Vol = \
            data.X, data.Y, data.Z, data.Den, data.Mass, data.Vol
        cutden = Den >1e-19
        X, Y, Z, Den, Mass, Vol = \
            sec.make_slices([X, Y, Z, Den, Mass, Vol], cutden)
        dim_cell = Vol**(1/3) 

        try:
            stream = np.load(f'{abspath}/data/{folder}/WHs/stream_{check}{snap}.npy', allow_pickle=True)
        except FileNotFoundError:
            from src.WH import find_transverse_com
            x_stream, y_stream, z_stream, thresh_cm = find_transverse_com(X, Y, Z, dim_cell, Den, Mass, theta_arr, params, Rstar)
            stream = [theta_arr, x_stream, y_stream, z_stream, thresh_cm]
            if save:
                np.save(f'{abspath}/data/{folder}/WH/stream_{check}{snap}.npy', stream)
        
        theta_wh, density_thresholds, indeces_enclosed, contour_stats = follow_the_stream_isodensity(X, Y, Z, dim_cell, Den, Mass, stream, params = params)
        w_params = np.array([[stats['width'] for stats in contour_stats],
                            [stats['ncells_w'] for stats in contour_stats]])
        h_params = np.array([[stats['height'] for stats in contour_stats],
                            [stats['ncells_h'] for stats in contour_stats]])
        indeces_boundary = np.array([stats['indeces_boundary'] for stats in contour_stats])

        if save:
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

        if not alice:
            theta_wh_sp, width_sp, N_width_sp, height_sp, N_height_sp = \
                np.loadtxt(f'{abspath}/data/{folder}/WH/wh_{check}{snap}.txt')
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15, 5))
            ax1.plot(theta_wh*radians, w_params[0], c = 'dodgerblue', label = 'isodensity (80\% mass enclosed)')
            ax1.plot(theta_wh_sp*radians, width_sp, c = 'k', label = 'spatial (\"80\%\" mass enclosed)')
            # print(f'N cells width and H: {w_params[1][idx_forplot]} {h_params_sp[1][idx_forplot]}')
            ax1.legend(fontsize = 16)
            ax1.set_ylabel(r'Width [$R_\odot$]')
            ax1.set_ylim(1.1, 10)
            ax2.plot(theta_wh*radians, h_params[0], c = 'dodgerblue')
            ax2.plot(theta_wh_sp*radians, height_sp, c = 'k')
            ax2.set_ylabel(r'Height [$R_\odot$]')
            ax2.set_ylim(.2, 10)
            for ax in [ax1, ax2]:
                ax.set_yscale('log')
                ax.set_xlabel(r'$\theta$')
                ax.set_xlim(-2.5, 2.5)
                ax.tick_params(axis='both', which='major', length = 8, width = 1.2)
                ax.tick_params(axis='both', which='minor', length = 4, width = 1)
            plt.suptitle(f'Width and height with different root-finder conditions at t = {np.round(tfb,2)} ' + r't$_{\rm fb}$', fontsize = 16)
            plt.tight_layout()
        
# %%
