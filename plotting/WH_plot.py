""" If comparison is True, it plots the WH and #cells for different res at different times.
If single_snap_behavior is True, it plots the WH and #cells for a single res at a single time.
If section is True, it plots the transverse plane, with H and W for a choosen angle.
"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import matplotlib.pyplot as plt
from Utilities.basic_units import radians
import numpy as np
import matplotlib.colors as colors
import Utilities.prelude as prel
import Utilities.sections as sec
import src.orbits as orb
from Utilities.operators import make_tree, draw_line, format_pi_frac
from plotting.paper.IHopeIsTheLast import split_data_red
from matplotlib.ticker import FuncFormatter

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
compton = 'Compton'
what = 'section' #section or comparison or max_compr or single_snap_behavior

Mbh = 10**m
Rs = 2*prel.G*Mbh / prel.csol_cgs**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rp
apo = orb.apocentre(Rstar, mstar, Mbh, beta)    
x_arr = np.arange(-2, 0.1, .1)
line3_4 = draw_line(x_arr, 3/4*np.pi)
lineminus3_4 = draw_line(x_arr, -3/4*np.pi)

if what == 'comparison':
    # line at 3/4 and -3/4 pi
    wanted_time = [0.52, 0.75, 1] # 1.2]
    checks = ['LowResNewAMR', 'NewAMR']
    compton = 'Compton'
    checks_name = ['Low', 'Fid']
    markers_sizes = [30, 10]
    linestyle_checks = ['solid', 'dashed']
    color_checks = [['k', 'grey'], ['dodgerblue', 'darkcyan']]

    for i, time in enumerate(wanted_time):
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = (10,12))
        ax1bis = ax1.twinx()
        ax2bis = ax2.twinx()

        for i, check in enumerate(checks):
            folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
            data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
            snaps = np.array([int(s) for s in data[:,0]])
            tfb = data[:, 1]
            idx_time = np.argmin(np.abs(tfb - time)) # find the closest snap to the wanted time
            snap = snaps[idx_time]
            if np.abs(tfb[idx_time]- time) > 0.05:
                print(f'Warning: the time {time} is not well represented in the data, closest is {tfb[idx_time]}')
            theta_arr, x_stream, y_stream, z_stream, _ = np.load(f'{abspath}/data/{folder}/WH/stream_{check}{snap}.npy', allow_pickle=True)
            wh = np.loadtxt(f'{abspath}/data/{folder}/WH/wh_{check}{snap}.txt')
            theta_wh, width, N_width, height, N_height = \
                wh[0], wh[1], wh[2], wh[3], wh[4]
            
            ax0.plot(x_stream/apo, y_stream/apo, c = color_checks[0][i], linestyle = linestyle_checks[i], label = f'{checks_name[i]}')
            ax0.plot(x_arr, line3_4, c = 'grey', alpha = 0.2)
            ax0.plot(x_arr, lineminus3_4, c = 'grey', alpha = 0.2)
            ax1.scatter(theta_wh * radians, N_width, c = color_checks[0][i], s = markers_sizes[i], label = f'{checks_name[i]}')
            ax2.scatter(theta_wh * radians, N_height, c = color_checks[0][i], s = markers_sizes[i], label = f'{checks_name[i]}')
            ax1bis.plot(theta_wh * radians, width, c = color_checks[1][i], linestyle = linestyle_checks[i], label = f'{checks_name[i]}')
            ax2bis.plot(theta_wh * radians, height, c = color_checks[1][i], linestyle = linestyle_checks[i], label = f'{checks_name[i]}')

        ax0.set_xlim(-2,0.2)
        ax0.set_ylim(-.2,.2)
        ax1.set_yscale('log')
        ax1.set_ylabel(r'N$_{\rm cells}$')
        ax1bis.set_yscale('log')
        ax1bis.set_ylabel(r'Width [$R_\odot$]', c = 'dodgerblue')
        ax1bis.tick_params(axis='y', which='major', length=10, width=1.5, color = 'dodgerblue')
        ax1bis.tick_params(axis='y', which='minor', length=5, width=1, color = 'dodgerblue')
        # ax1bis.set_ylim(.1,10)
        ax1bis.tick_params(axis='y', labelcolor='dodgerblue')
        ax2.set_yscale('log')
        ax2.set_ylabel(r'N$_{\rm cells}$')
        ax2bis.set_yscale('log')
        ax2bis.set_ylabel(r'Height [$R_\odot$]', c = 'dodgerblue')
        ax2bis.tick_params(axis='y', which='major', length=10, width=1.5, color = 'dodgerblue')
        ax2bis.tick_params(axis='y', which='minor', length=5, width=1, color = 'dodgerblue')
        # ax2bis.set_ylim(0.1,10)
        ax2bis.tick_params(axis='y', labelcolor='dodgerblue')
        for ax in [ax1, ax2]:
            ax.set_xlim(-2.5, 2.5)#-3/4*np.pi, 3/4*np.pi)
            ax.set_ylim(.9, 20)
            ax.grid()
            ax.tick_params(axis='both', which='major', length=10, width=1.5)
            ax.tick_params(axis='both', which='minor', length=5, width=1)
            ax.set_xlabel(r'$\theta$')
        
        ax0.legend(fontsize = 14)
        ax1.legend(fontsize = 14)
        ax1bis.legend(fontsize = 14)
        plt.suptitle(r't/t$_{fb}$ = ' + str(time), fontsize = 16)
        plt.tight_layout()
        plt.show()
        # Find the stream in the simulation data
        # tree_plot = KDTree(np.array([X, Y, Z]).T)
        # _, indeces_plot = tree_plot.query(np.array([x_stream, y_stream, z_stream]).T, k=1)

if what == 'single_snap_behavior' or what == 'section':
    check = 'NewAMR'
    snap = 162
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    path = f'{abspath}/TDE/{folder}/{snap}'
    # Load data snap
    tfb_single = np.loadtxt(f'{abspath}/TDE/{folder}/{snap}/tfb_{snap}.txt')
    data = make_tree(path, snap, energy = False)
    X, Y, Z, Den, Mass, Vol = \
        data.X, data.Y, data.Z, data.Den, data.Mass, data.Vol
    cutden = Den >1e-19
    dim_cell = Vol**(1/3)
    X, Y, Z, Den, Mass, dim_cell = \
        sec.make_slices([X, Y, Z, Den, Mass, dim_cell], cutden)
    midplane = np.abs(Z) < dim_cell
    X_midplane, Y_midplane, Z_midplane, dim_midplane, Den_midplane, Mass_midplane = \
        sec.make_slices([X, Y, Z, dim_cell, Den, Mass], midplane)
    # Load data stream
    theta_arr, x_stream, y_stream, z_stream, thresh_cm = np.load(f'{abspath}/data/{folder}/WH/stream_{check}{snap}.npy', allow_pickle=True)
    stream = [theta_arr, x_stream, y_stream, z_stream, thresh_cm]
    wh = np.loadtxt(f'{abspath}/data/{folder}/WH/wh_{check}{snap}.txt')
    theta_wh, width, N_width, height, N_height = wh[0], wh[1], wh[2], wh[3], wh[4]
    indeces_boundary = np.load(f'{abspath}/data/{folder}/WH/indeces_boundary_{check}{snap}.npy')
    indeces_boundary_lowX, indeces_boundary_upX, indeces_boundary_lowZ, indeces_boundary_upZ = \
        indeces_boundary[:,0], indeces_boundary[:,1], indeces_boundary[:,2], indeces_boundary[:,3]
    indeces_all = np.arange(len(X))
    x_low_width, y_low_width = X[indeces_boundary_lowX], Y[indeces_boundary_lowX]
    x_up_width, y_up_width = X[indeces_boundary_upX], Y[indeces_boundary_upX]

    if what == 'single_snap_behavior':
        # Plot stream (zoom out and in on theorbital plane) 
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,6), width_ratios= [1.5, .5])
        for ax in [ax1, ax2]:
            ax.scatter(X_midplane/apo, Y_midplane/apo, c = Den_midplane, s = 1, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-13, vmax = 1e-6))
            ax.plot(x_stream[:-70]/apo, y_stream[:-70]/apo, c = 'k')
            ax.plot(x_stream[-30:]/apo, y_stream[-30:]/apo, c = 'k')
            ax.plot([x_stream[0]/apo, x_stream[-1]/apo] , [y_stream[0]/apo, y_stream[-1]/apo], c = 'k')
            ax.plot(x_low_width/apo, y_low_width/apo, c = 'coral')
            ax.plot(x_up_width/apo, y_up_width/apo, c = 'coral')
            ax.plot(x_arr, line3_4, c = 'grey', linestyle = '--')
            ax.plot(x_arr, lineminus3_4, c = 'grey', linestyle = '--')
            ax.set_xlabel(r'$X [R_{\rm a}]$')
        ax1.set_ylabel(r'$Y [R_{\rm a}]$')
        ax1.set_xlim(-1, .1)
        ax1.set_ylim(-.3, .3)
        ax2.set_xlim(-.2, .05)
        ax2.set_ylim(-.1, .1)
        plt.suptitle(r'Stream at t/t$_{\rm fb}$ = ' + str(np.round(tfb_single,2)) + f', check: {check}', fontsize = 16)

        # Plot width and height and number of cells
        fig, (ax1,ax2) = plt.subplots(1,2, figsize = (18,6))
        ax1.plot(theta_arr * radians, width, c = 'k')
        img = ax1.scatter(theta_arr * radians, width, c = N_width, cmap = 'rainbow', vmin = 0, vmax = 15, marker = 's', label  = '80% mass')
        cbar = plt.colorbar(img)
        cbar.set_label(r'Ncells')
        ax1.set_ylabel(r'Width [$R_\odot$]')
        ax1.set_ylim(0.1,6)
        ax2.plot(theta_arr * radians, height, c = 'k')
        img = ax2.scatter(theta_arr * radians, height, c = N_height, cmap = 'rainbow', vmin = 0, vmax = 10, marker = 's', label  = '80% mass')
        cbar = plt.colorbar(img)
        cbar.set_label(r'Ncells')
        ax2.set_ylabel(r'Height [$R_\odot$]')
        ax2.set_ylim(0.1, 4)
        for ax in [ax1, ax2]:
            ax.set_xlabel(r'$\theta$')
            ax.set_xlim(-2.5, 2.5)#-3/4*np.pi, 3/4*np.pi)
            ax.grid()
        ax1.legend(fontsize = 14)
        plt.tight_layout()
        plt.suptitle(r't/t$_{fb}$ = ' + str(np.round(tfb_single,2)), fontsize = 16)

    if what == 'section':
        q_color = 'Mass'
        idx = np.argmin(height) #np.argmin(np.abs(theta_arr + 3*np.pi/4)) # find the index of the theta closest to 1.5
        chosentheta = theta_wh[idx]
        thetabefore = theta_wh[np.argmin(height)] - np.pi/2
        thetaafter = theta_wh[np.argmin(height)] + np.pi/2
        x_arr = np.arange(-.1, 1.1, .1)
        y_arr_chosentheta = draw_line(x_arr, chosentheta)
        y_arr_thetaafterbefore = draw_line(x_arr, thetabefore)

        indeces = np.arange(len(X))
        # find in the simulaion the boundary and the points inside
        condition_T, x_T = sec.transverse_plane(X, Y, Z, dim_cell, x_stream, y_stream, z_stream, idx, just_plane = False)
        x_plane, x_T_plane, y_plane, z_plane, dim_plane, mass_plane, den_plane, indeces_plane = \
            sec.make_slices([X, x_T, Y, Z, dim_cell, Mass, Den, indeces], condition_T)
        x_T_low, x_T_up = x_T[indeces_boundary_lowX[idx]], x_T[indeces_boundary_upX[idx]]
        z_low, z_up = Z[indeces_boundary_lowZ[idx]], Z[indeces_boundary_upZ[idx]]
        
        condition_thresh = np.logical_and(np.abs(x_T_plane) < thresh_cm[idx], np.abs(z_plane) < thresh_cm[idx])
        x_sec, x_T_sec, y_sec, z_sec, dim_sec, den_sec, mass_sec = \
            sec.make_slices([x_plane, x_T_plane, y_plane, z_plane, dim_plane, den_plane, mass_plane], condition_thresh)
        stream_cells_T = np.logical_and(x_T_sec >= x_T_low, x_T_sec <= x_T_up)
        stream_cells_Z = np.logical_and(z_sec >= z_low, z_sec <= z_up)
        stream_cells = np.logical_and(stream_cells_T, stream_cells_Z)

        print('Mass in the stream/mass section: ', np.sum(mass_sec[stream_cells])/np.sum(mass_sec))
        print(f'Mass cells betwwen +-T, no limit in Z: {int(100*np.sum(mass_sec[stream_cells_T]) / np.sum(mass_sec))}\% of total mass in the plane (below section threshold)')
        print(f'Mass cells betwwen +-Z, no limit in T: {int(100*np.sum(mass_sec[stream_cells_Z]) / np.sum(mass_sec))}\% of total mass in the plane (below section threshold)')

        # if q_color == 'Mass':
        #     vmax = [1e-7, 8e-9] # for 0.1 tfb: [1e-5, 5e-12] 
        #     vmin = 1e-13 #1e-14 
        #     q_points = Mass_midplane
        # elif q_color == 'Density':
        #     vmax = [1e-2, 2e-10] 
        #     vmin = 1e-12
        #     q_points = Den_midplane * prel.den_converter
        # fig, (ax0, ax1) = plt.subplots(1, 2,figsize = (18,6), width_ratios = [1.5, 1])
        # for i, ax in enumerate([ax0, ax1]):
        #     img = ax.scatter(X_midplane/apo, Y_midplane/apo, c = q_points, s = 1, cmap = 'rainbow', norm = colors.LogNorm(vmin = vmin, vmax = vmax[i]))
        #     cbar = plt.colorbar(img)
        #     if q_color == 'Mass':
        #         cbar.set_label(r'Mass [$M_\odot$]') 
        #     elif q_color == 'Density':
        #         cbar.set_label(r'Density [g/cm$^3$]') 
        #     # ax0.plot(x_stream/apo, y_stream/apo, c = 'k', ls = ':')
        #     ax.scatter(x_low_width/apo, y_low_width/apo, c = 'k', s = 5)
        #     ax.scatter(x_up_width/apo, y_up_width/apo, c = 'k', s = 5)
        #     ax.plot(x_arr, y_arr_chosentheta, c = 'b', label = r'$\alpha$')
        #     ax.plot(x_arr, y_arr_thetaafterbefore, c = 'r', label = r'$\alpha \pm \pi/2$')
        #     ax.set_xlabel(r'X [$R_{\rm a}$]')
        #     ax.scatter(0,0, c = 'k', s = 40)

        # ax0.set_xlim(-0.8, 0.05) #-1, 0.1)
        # ax0.set_ylim(-.3, .3) #-.3, .3)
        # ax0.set_ylabel(r'Y [$R_{\rm a}$]')
        # ax1.set_xlim(-0.1, 0.05) #-1, 0.1)
        # ax1.set_ylim(-.1, .1) #-.3, .3)
        # ax1.legend(fontsize = 14, loc = 'upper left')
        # plt.savefig(f'{abspath}/Figs/max_compr/{check}{snap}_{q_color}.png')

        if q_color == 'Mass':
            vmax = 1e-9 # for 0.1 tfb: [1e-5, 5e-12] 
            vmin = 1e-13 #1e-14 
            q_points = Mass_midplane
            q_plane = mass_plane
            q_sec = mass_sec
        elif q_color == 'Density':
            vmax = 1e-2
            vmin = 1e-12
            q_points = Den_midplane * prel.den_converter
            q_plane = den_plane * prel.den_converter
            q_sec = den_sec * prel.den_converter

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2,figsize = (18,14))
        img = ax0.scatter(X_midplane/apo, Y_midplane/apo, c = q_points, s = 1, cmap = 'rainbow', norm = colors.LogNorm(vmin = vmin, vmax = vmax))
        cbar = plt.colorbar(img, ax=ax0)
        if q_color == 'Mass':
            cbar.set_label(r'Mass [$M_\odot$]') 
        elif q_color == 'Density':
            cbar.set_label(r'Density [g/cm$^3$]')
        ax0.plot(x_stream/apo, y_stream/apo, c = 'k', ls = ':')
        ax0.scatter(x_low_width/apo, y_low_width/apo, c = 'k', s = 5)
        ax0.scatter(x_up_width/apo, y_up_width/apo, c = 'k', s = 5)
        ax0.plot(x_arr, y_arr_chosentheta, c = 'b')
        ax0.plot(x_arr, y_arr_thetaafterbefore, c = 'r')
        ax0.set_xlim(-1, 0.1)
        ax0.set_ylim(-.3, .3)
        ax0.set_xlabel(r'X [$R_{\rm a}$]')
        ax0.set_ylabel(r'Y [$R_{\rm a}$]')
        
        img = ax1.scatter(x_T_plane, z_plane, c = q_plane, s = 20, cmap = 'rainbow', norm = colors.LogNorm(vmin = vmin, vmax = vmax))
        cbar = plt.colorbar(img)
        img = ax1.scatter(x_T_sec[stream_cells], z_sec[stream_cells], c = q_sec[stream_cells], s = 40, edgecolor = 'k', cmap = 'rainbow', norm = colors.LogNorm(vmin = vmin, vmax = vmax))
        cbar.set_label(r'Mass [$M_\odot$]')
        ax1.set_xlabel(r'T [$R_\odot$]')
        ax1.set_ylabel(r'Z [$R_\odot$]')
        ax1.axvline(x=x_T_low, color='grey', linestyle = '--')
        ax1.axvline(x=x_T_up, color='grey', linestyle = '--')
        ax1.axhline(y=z_low, color='grey', linestyle = '--')
        ax1.axhline(y=z_up, color='grey', linestyle = '--')
        # put the lim now to find the middle ones u want
        ax1.set_xlim(x_T_low-2, x_T_up+2)
        ax1.set_ylim(z_low-1, z_up+1)
        # ticks
        original_ticks = ax1.get_xticks()
        mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
        all_ticks = np.concatenate((original_ticks, mid_ticks))
        ax1.set_xticks(all_ticks)
        ax1.set_xticklabels([f'{tick:.1f}' if tick in original_ticks else '' for tick in all_ticks])
        original_ticks = ax1.get_yticks()
        mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
        all_ticks = np.concatenate((original_ticks, mid_ticks))
        ax1.set_yticks(all_ticks)
        ax1.set_yticklabels([f'{tick:.1f}' if tick in original_ticks else '' for tick in all_ticks])
        # again limits to avoid extra ticks outside
        ax1.set_xlim(x_T_low-2, x_T_up+2)
        ax1.set_ylim(z_low-1, z_up+1)

        # Compute weighted histogram data
        bins = np.arange(-3.1, 3.1, .2)
        ax2.hist(x_T_sec, bins = bins, weights=mass_sec, color='orange', alpha = 0.5)
        ax2.axvline(x = x_T_low, color='grey', linestyle = '--', label = 'Stream width')
        ax2.axvline(x = x_T_up, color='grey', linestyle = '--')
        ax2.set_ylabel(r'Mass [$M_\odot$]')
        ax2.set_xlabel(r'T [$R_\odot$]')
        ax2.axvline(0, c = 'k')

        img = ax3.scatter(x_T_sec, den_sec, s = 20, c = z_sec, cmap = 'viridis', norm = colors.LogNorm(vmin = 1e-2, vmax = 10))
        cbar = plt.colorbar(img, ax=ax3)
        cbar.set_label(r'Z [$R_\odot$]')
        ax3.set_xlabel(r'T [$R_\odot$]')
        ax3.set_ylabel(r'Density [$M_\odot/R_\odot^3$]')

        plt.suptitle(r'$\alpha$ = ' + f'{np.round(theta_wh[idx], 2)}, {check}, t =  {np.round(tfb_single,2)}' + r' $t_{\rm fb}$', fontsize = 25)
        plt.tight_layout()

if what == 'max_compr':
    checks = ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']
    colors_checks = ['C1', 'yellowgreen', 'darkviolet']
    names_checks = ['Low', 'Fid', 'High']
    markers_checks = ['x', 'o', 's']
    ratio_of_pi = 2
    vmin_ang = -np.pi/8
    vmax_ang = np.pi/4

    # to look how the expansion/compression changes with time
    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (10,10))
    ax1.set_ylabel(r'$\Delta (\alpha-\pi$/' + f'{ratio_of_pi}) /' + r'$\Delta (\alpha+\pi$/' + f'{ratio_of_pi})')
    ax2.set_ylabel(r'H ($\alpha-\pi$/' + f'{ratio_of_pi}) /' + r'H($\alpha+\pi$/' + f'{ratio_of_pi})')
    for ax in [ax1, ax2]:
        ax.axhline(y=1, color='grey', linestyle = '--')
        ax.set_yscale('log')
        ax.grid()
    ax2.set_xlabel(r't [t$_{\rm fb}$]')

    # to plot where is the maximum compression
    fig_compr, ax1_compr = plt.subplots(1,1, figsize = (12,7))
    ax1_compr.set_xlabel(r't/t$_{\rm fb}$')
    ax1_compr.set_ylabel(r'H$_{\rm min} [R_\star]$')
    ax1_compr.grid()

    for j, check in enumerate(checks):
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        snaps, Lum, tfbs = split_data_red(check)
        len_arr = len(snaps[tfbs< 1.05])
        snaps, Lum, tfbs = snaps[:len_arr], Lum[:len_arr], tfbs[:len_arr]
        print(snaps[np.argmin(np.abs(tfbs-.1))])

        before_after_width = []
        before_after_h = []
        tfb_befaft = []
        theta_compr = np.zeros(len(snaps))
        height_compr = np.zeros(len(snaps))
        width_compr = np.zeros(len(snaps))
        Nheight_compr = np.zeros(len(snaps))
        Nwidth_compr = np.zeros(len(snaps))

        for i, tfb in enumerate(tfbs):
            snap = snaps[i]
            path = f'{abspath}/TDE/{folder}/{snap}'
            wh = np.loadtxt(f'{abspath}/data/{folder}/WH/wh_{check}{snap}.txt')
            theta_wh, width, N_width, height, N_height = wh[0], wh[1], wh[2], wh[3], wh[4]
            angle_check_idx = np.argmin(height)
            theta_compr[i] = theta_wh[angle_check_idx]
            height_compr[i] = height[angle_check_idx]
            width_compr[i] = width[angle_check_idx]
            Nheight_compr[i] = N_height[angle_check_idx]
            Nwidth_compr[i] = N_width[angle_check_idx]
            theta_inc = theta_compr[i] - np.pi/ ratio_of_pi
            theta_out = theta_compr[i] + np.pi/ ratio_of_pi

            idx_before = np.argmin(np.abs(theta_wh - theta_inc)) 
            idx_after = np.argmin(np.abs(theta_wh - theta_out))
            if snap == 98:
                print(theta_compr[i])

            if theta_inc < -np.pi or theta_inc > 0:
                continue
            before_after_width.append(width[idx_before] / width[idx_after])
            before_after_h.append(height[idx_before] / height[idx_after])
            tfb_befaft.append(tfb)
        
        ax1.plot(tfb_befaft, before_after_width, c = colors_checks[j], label = names_checks[j])
        ax2.plot(tfb_befaft, before_after_h, c = colors_checks[j])
    
        img_compr = ax1_compr.scatter(tfbs, height_compr/Rstar, c = theta_compr, marker = markers_checks[j], label = names_checks[j], vmin = vmin_ang, vmax = vmax_ang, cmap = 'rainbow', s = 50)
    
    cbar = plt.colorbar(img_compr)
    cbar.set_label(r'$\alpha$ [rad]')
    # Define the exact ticks you want:
    ticks = np.array([vmin_ang, 0, np.pi/8,  vmax_ang])
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_pi_frac))
    ax1_compr.legend(fontsize = 18)
    ax1_compr.axvline(0.1, c = 'k', ls = '--')
    ax1.legend(fontsize = 18)
    

        
                
