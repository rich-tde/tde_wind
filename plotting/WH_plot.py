""" If what = 'comparison' is True, it plots the WH and #cells as function of the anomaly for different res at different times.
If what = 'single_snap_behavior' is True, it plots the WH and #cells for a single res at a single time + a visualization of the stream in the orbital plane.
If what = 'onlysection' is True, it plots only the midplane with stream and width of the chosem res.
If what = 'section4' is True, it plots the midplane and transverse plane, mass distribution and T-density scatter.
If what = 'max_compr' is True, it plots the res comparison of H_min and ratio between before and after the compression.
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
from Utilities.operators import draw_line, format_pi_frac
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
what = 'section4' # 'onlysection' or 'section4' or 'comparison' or 'max_compr' or 'single_snap_behavior' 

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
x_arr = np.arange(-20, 0.1, .1)
line3_4 = draw_line(x_arr, 3/4*np.pi)
lineminus3_4 = draw_line(x_arr, -3/4*np.pi)

if what == 'comparison':
    # line at 3/4 and -3/4 pi
    wanted_time = [0.2, 0.3, 1, 1.5] #5, 0.75, 1]#, 1.5]
    checks = ['LowResNewAMR', 'NewAMR', 'HiResNewAMR', 'HiResStream','HiResStream2']
    compton = 'Compton'
    checks_name = ['Low', 'Middle', 'High', 'HiResStream', 'HiResStream2']
    markers_sizes = [40, 30, 20, 15, 10]
    linestyle_checks = ['solid', 'dashed', 'dotted', 'dashdot', 'dashdot']
    color_checks = ['C1', 'yellowgreen', 'darkviolet', 'magenta', 'dodgerblue']

    for i, time in enumerate(wanted_time): 
        fig_st, ax_st = plt.subplots(1, 1, figsize = (10,6))
        fig_wh, ((ax_w, ax_h), (ax_Nw, ax_Nh)) = plt.subplots(2, 2, figsize = (20,12))
        # fig_pp, (ax_pp1, ax_pp2) = plt.subplots(2,1, figsize = (10,10))
        ratio_of_pi = 2
        # ax_pp1.set_ylabel(r'$\Delta (\alpha+\pi$/' + f'{ratio_of_pi}) /' + r'$\Delta (\alpha-\pi$/' + f'{ratio_of_pi})')
        # ax_pp2.set_ylabel(r'H ($\alpha+\pi$/' + f'{ratio_of_pi}) /' + r'H($\alpha-\pi$/' + f'{ratio_of_pi})')

        # to plot where is the maximum compression
        # fig_compr, ax1_compr = plt.subplots(1,1, figsize = (12,7))
        # ax1_compr.set_ylabel(r'H$_{\rm min} [R_\star]$')

        for i, check in enumerate(checks):
            # pick the simulation
            folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
            data = np.loadtxt(f'{abspath}/data/{folder}/projection/Dentime_proj.txt', ndmin=2)
            snaps = data[0].astype(int)
            tfb = data[1] 
            idx_time = np.argmin(np.abs(tfb - time)) # find the closest snap to the wanted time
            snap = snaps[idx_time]
            if np.abs(tfb[idx_time]- time) > 0.05:
                print(f'Warning: the time {time} is not well represented in the data, closest is {tfb[idx_time]}')
                continue
            print(f'Check: {check}, time: {np.round(tfb[idx_time], 2)}', f'snap: {snap}')
            # Load the data
            data_stream = np.load(f'{abspath}/data/{folder}/WH/stream/stream_{check}_{snap}.npz', allow_pickle=True)
            x_stream = data_stream['x_cm']
            y_stream = data_stream['y_cm']
            z_stream = data_stream['z_cm']
  
            ax_st.plot(x_stream/apo, y_stream/apo, c = color_checks[i], linestyle = linestyle_checks[i], label = f'{checks_name[i]}')
            ax_st.plot(x_arr, line3_4, c = 'grey', alpha = 0.2)
            ax_st.plot(x_arr, lineminus3_4, c = 'grey', alpha = 0.2)
            
            data_width = np.load(f'{abspath}/data/{folder}/WH/wh_0.8{check}_{snap}.npz', allow_pickle=True)
            theta_wh = data_width['theta_wh']
            width = data_width['width']
            N_width = data_width['N_width']
            height = data_width['height']
            N_height = data_width['N_height']

            # if check == 'LowResNewAMR':
            #     continue
            ax_w.plot(theta_wh * radians, width, c = color_checks[i], linestyle = linestyle_checks[i], label = f'{checks_name[i]}')
            ax_Nw.plot(theta_wh * radians, N_width, c = color_checks[i], linestyle = linestyle_checks[i], label = f'{checks_name[i]}')
            ax_h.plot(theta_wh * radians, height, c = color_checks[i], linestyle = linestyle_checks[i], label = f'{checks_name[i]}')
            ax_Nh.plot(theta_wh * radians, N_height, c = color_checks[i], linestyle = linestyle_checks[i], label = f'{checks_name[i]}')

        ax_w.set_title('Width', fontsize = 20)
        ax_h.set_title('Height', fontsize = 20)
        ax_st.set_xlim(-1, 0.1)
        ax_st.set_ylim(-.2,.2)
        ax_st.set_xlabel(r'$X [r_{\rm a}]$')
        ax_st.set_ylabel(r'$Y [r_{\rm a}]$')
        ax_st.legend(fontsize = 16)

        ax_w.set_ylabel(r'Stream size [$R_\odot$]')
        ax_w.set_ylim(1, 30)
        ax_h.set_ylim(0.1, 20)
        ax_w.legend(fontsize = 16)
        ax_Nw.set_ylabel(r'N$_{\rm cells}$')
        ax_Nw.set_ylim(9, 100) #np.min(N_width)/2, np.max(N_width) + 1)
        ax_Nh.set_ylim(.9, np.max(N_height) + 1)
        for ax in [ax_w, ax_h, ax_Nw, ax_Nh]: 
            ax.set_xlim(-2.5, 2.5) #-3/4*np.pi, 3/4*np.pi)
            ax.grid()
            ax.tick_params(axis='both', which='major', length=10, width=1.5)
            ax.tick_params(axis='both', which='minor', length=7, width=1.2)
            ax.set_yscale('log')
            if ax in [ax_Nw, ax_Nh]:
                ax.set_xlabel(r'$\theta$ [rad]')
            else: 
                ax.set_xlabel('')
        
        fig_wh.suptitle(r't/t$_{fb}$ = ' + str(time), fontsize = 20)
        fig_wh.tight_layout()
        fig_st.suptitle(r't/t$_{fb}$ = ' + str(time), fontsize = 20)
        fig_st.tight_layout()
        # Find the stream in the simulation data
        # tree_plot = KDTree(np.array([X, Y, Z]).T)
        # _, indeces_plot = tree_plot.query(np.array([x_stream, y_stream, z_stream]).T, k=1)

if what == 'single_snap_behavior' or what == 'section4' or what == 'onlysection':
    check = 'HiResNewAMR'
    snap = 76
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    snaps, tfb = np.loadtxt(f'{abspath}/data/{folder}/projection/Dentime_proj.txt', ndmin=2)
    snaps = snaps.astype(int)
    tfb_single = tfb[np.argmin(np.abs(snaps-snap))]

    # Load data snap
    data_stream = np.load(f'{abspath}/data/{folder}/WH/stream/stream_{check}_{snap}.npz', allow_pickle=True)
    x_stream = data_stream['x_cm']
    y_stream = data_stream['y_cm']
    z_stream = data_stream['z_cm']
    data_width = np.load(f'{abspath}/data/{folder}/WH/wh_0.8{check}_{snap}.npz', allow_pickle=True)
    theta_wh = data_width['theta_wh']
    x_low_width = data_width['X_low_w']
    x_up_width = data_width['X_up_w']
    y_low_width = data_width['Y_low_w']
    y_up_width = data_width['Y_up_w']
    width = data_width['width']
    N_width = data_width['N_width']
    height = data_width['height']
    N_height = data_width['N_height']
    
    if what == 'single_snap_behavior': 
        # Plot stream (zoom out and in on theorbital plane) 
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,8), width_ratios= [1.2, .8])
        for ax in [ax1, ax2]: 
            # img =ax.scatter(X_midplane/Rt, Y_midplane/Rt, c = Den_midplane, s = 1, cmap = 'viridis', norm = colors.LogNorm(vmin = 1e-13, vmax = 1e-6))
            ax.scatter(x_stream/Rt, y_stream/Rt, s = 2, c = 'k')
            # ax.scatter(x_stream[-30:]/Rt, y_stream[-30:]/Rt, s = 2, c = 'k')
            ax.scatter([x_stream[0]/Rt, x_stream[-1]/Rt] , [y_stream[0]/Rt, y_stream[-1]/Rt], s = 2, c = 'k')
            ax.scatter(x_low_width/Rt, y_low_width/Rt, ls = '--', s = 2, c = 'k')
            ax.scatter(x_up_width/Rt, y_up_width/Rt, ls = '--', s = 2, c = 'k')
            ax.plot(x_arr, line3_4, c = 'grey', linestyle = '--')
            ax.plot(x_arr, lineminus3_4, c = 'grey', linestyle = '--')
            ax.set_xlabel(r'$X [r_{\rm a}]$')
        # cbar = plt.colorbar(img, ax=ax1, orientation = 'horizontal')
        # cbar.set_label(r'Density [$M_\odot/R_\odot^3$]')
        ax1.set_ylabel(r'$Y [r_{\rm a}]$')
        ax1.set_xlim(-10, 2)
        ax1.set_ylim(-5, 5)
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        plt.suptitle(r'Stream at t/t$_{\rm fb}$ = ' + str(np.round(tfb_single,2)) + f', check: {check}', fontsize = 16)

        # Plot width and height and number of cells
        fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (18,9))
        ax1.plot(theta_wh * radians, width, c = 'dodgerblue')
        img = ax3.scatter(theta_wh * radians, N_width, c = 'dodgerblue')#, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1, vmax = 15))
        # cbar = plt.colorbar(img)
        ax1.set_ylabel(r'Width [$R_\odot$]')
        ax3.set_ylabel(r'Ncells')
        ax2.plot(theta_wh * radians, height, c = 'orange')
        img = ax4.scatter(theta_wh * radians, N_height, c = 'orange')#, cmap = 'rainbow', norm = colors.LogNorm(vmin = 0.5, vmax = 10))
        # cbar = plt.colorbar(img)
        ax2.set_ylabel(r'Height [$R_\odot$]')
        # ax4.set_ylabel(r'Ncells')
        ax1.set_ylim(1.1, 10)
        ax2.set_ylim(.2, 10)
        ax3.set_ylim(8, 50)    
        ax4.set_ylim(0.9, 30)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_yscale('log')
            ax.set_xlabel(r'$\theta$')
            ax.set_xlim(-2.5, 2.5)
            ax.tick_params(axis='both', which='major', length = 8, width = 1.2)
            ax.tick_params(axis='both', which='minor', length = 4, width = 1)
        plt.tight_layout()

    if what == 'section4' or what == 'onlysection':
        from Utilities.selectors_for_snap import select_prefix
        from Utilities.operators import make_tree
        q_color = 'Density'
        idx = np.argmin(height) #np.argmin(np.abs(theta_arr + 3*np.pi/4)) # find the index of the theta closest to 1.5
        chosentheta = theta_wh[idx]
        thetabefore = chosentheta - np.pi/2
        thetaafter = chosentheta + np.pi/2
        x_arr = np.arange(-.1, 1.1, .1)
        y_arr_chosentheta = draw_line(x_arr, chosentheta)
        y_before = draw_line(x_arr, thetabefore)
        y_after = draw_line(x_arr, thetaafter)

        path = select_prefix(m, check, mstar, Rstar, beta, n, compton)
        path = f'{path}/{snap}'
        data = make_tree(path, snap, energy = True)
        X, Y, Z, Den, Vol = data.X, data.Y, data.Z, data.Den, data.Vol
        X, Y, Z, Den, Vol = sec.make_slices([X, Y, Z, Den, Vol], Den > 1e-19)
        dim_cell = Vol**(1/3)
        Mass = Den * Vol
        cut_mid =  np.abs(Z) < Vol**(1/3)
        x_midplane, y_midplane, Den_midplane, vol_midplane, Mass_midplane = \
            sec.make_slices([X, Y, Den, Vol, Mass], cut_mid)

        if q_color == 'Mass':
            q_points = Mass_midplane
        elif q_color == 'Density':
            q_points = Den_midplane * prel.den_converter

        if what == 'onlysection':
            fig, ax0 = plt.subplots(1,1, figsize = (12,7))
            # plt.suptitle(f't =  {np.round(tfb_single,2)}' + r' $t_{\rm fb}$', fontsize = 25)

        else:
            indeces_boundary = np.load(f'{abspath}/data/{folder}/WH/indeces_boundary_0.8{check}_{snap}.npy', allow_pickle=True)
            print(np.shape(indeces_boundary))
            idx_low, idx_up, idx_low_h, idx_up_h = indeces_boundary[idx,0], indeces_boundary[idx,1], indeces_boundary[idx,2], indeces_boundary[idx,3]
            thresh_cm = data_stream['thresh_cm']
            indeces = np.arange(len(X))
            # find in the simulaion the boundary and the points inside
            condition_T, x_T = sec.transverse_plane(X, Y, Z, dim_cell, x_stream, y_stream, z_stream, idx, Rstar, just_plane = False)
            x_plane, x_T_plane, y_plane, z_plane, dim_plane, mass_plane, den_plane, indeces_plane = \
                sec.make_slices([X, x_T, Y, Z, dim_cell, Mass, Den, indeces], condition_T)
            x_T_low, x_T_up = x_T[idx_low], x_T[idx_up]
            z_low, z_up = Z[idx_low_h], Z[idx_up_h]
            
            condition_thresh = np.logical_and(np.abs(x_T_plane) < thresh_cm[idx], np.abs(z_plane) < thresh_cm[idx])
            x_sec, x_T_sec, y_sec, z_sec, dim_sec, den_sec, mass_sec = \
                sec.make_slices([x_plane, x_T_plane, y_plane, z_plane, dim_plane, den_plane, mass_plane], condition_thresh)
            # stream_cells_T = np.logical_and(x_T_sec >= x_T_low, x_T_sec <= x_T_up)
            # stream_cells_Z = np.logical_and(z_sec >= z_low, z_sec <= z_up)
            # stream_cells = np.logical_and(stream_cells_T, stream_cells_Z)
            # print(f'Mass cells betwwen +-T, no limit in Z: {int(100*np.sum(mass_sec[stream_cells_T]) / np.sum(mass_sec))}% of total mass in the plane (below section threshold)')
            # print(f'Mass cells betwwen +-Z, no limit in T: {int(100*np.sum(mass_sec[stream_cells_Z]) / np.sum(mass_sec))}% of total mass in the plane (below section threshold)')
            enclosed_cells = np.load(f'{abspath}/data/{folder}/WH/enclosed/indeces_enclosed_0.8{check}_{snap}.npy', allow_pickle=True)
            stream_cells = enclosed_cells[idx]
            print('Mass in the stream/mass section: ', np.sum(Mass[stream_cells])/np.sum(mass_sec))
            if q_color == 'Mass':
                q_encl = Mass
                q_plane = mass_plane
                q_sec = mass_sec
            elif q_color == 'Density':
                q_encl = Den * prel.den_converter
                q_plane = den_plane * prel.den_converter
                q_sec = den_sec * prel.den_converter
            
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2,figsize = (20,14))
            img = ax1.scatter(x_T_plane, z_plane, c = q_plane, s = 20, cmap = 'rainbow', norm = colors.LogNorm(vmin = np.percentile(q_plane, 50), vmax = np.max(q_plane)))
            cbar = plt.colorbar(img)
            img = ax1.scatter(x_T[stream_cells], Z[stream_cells], c = q_encl[stream_cells], s = 40, edgecolor = 'k', cmap = 'rainbow', norm = colors.LogNorm(vmin = np.percentile(q_sec, 50), vmax = np.max(q_sec)))
            cbar.set_label(r'Mass [$M_\odot$]' if q_color == 'Mass' else r'Density [g/cm$^3$]')
            ax1.set_xlabel(r'N [$R_\odot$]')
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
            ax2.set_xlabel(r'N [$R_\odot$]')
            ax2.axvline(0, c = 'k')

            img = ax3.scatter(x_T_sec, den_sec, s = 20, c = z_sec, cmap = 'viridis', norm = colors.LogNorm(vmin = 1e-1, vmax = 10))
            cbar = plt.colorbar(img, ax=ax3)
            cbar.set_label(r'Z [$R_\odot$]')
            ax3.set_xlabel(r'N [$R_\odot$]')
            ax3.set_ylabel(r'Density [$M_\odot/R_\odot^3$]')

            plt.suptitle(r'$\alpha$ = ' + f'{np.round(theta_wh[idx], 2)}, {check}, t =  {np.round(tfb_single,2)}' + r' $t_{\rm fb}$, number cells $\Delta$ = ' + f'{N_width[idx]}, H = {N_height[idx]}', fontsize = 25)
        
        img = ax0.scatter(x_midplane/apo, y_midplane/apo, c = q_points, s = 1, cmap = 'rainbow', norm = colors.LogNorm(vmin = np.percentile(q_points, 20), vmax = np.percentile(q_points, 95)))
        cbar = plt.colorbar(img, ax=ax0)
        cbar.set_label(r'Mass [$M_\odot$]' if q_color == 'Mass' else r'Density [g/cm$^3$]', fontsize = 25)
        cbar.ax.tick_params(which='major', labelsize=25, width = .8, length = 8, pad = 10)
        cbar.ax.tick_params(which='minor',  width = .6, length = 5, pad = 10)
            
        ax0.plot(x_stream[:185]/apo, y_stream[:185]/apo, c = 'k', ls = ':')
        ax0.plot(x_low_width[:185]/apo, y_low_width[:185]/apo, c = 'k')
        ax0.plot(x_up_width[:185]/apo, y_up_width[:185]/apo, c = 'k')
        # ax0.plot(x_arr, y_arr_chosentheta, c = 'k', label = r'$\alpha$')
        # ax0.plot(x_arr, y_before, c = 'r', label = r'$\alpha \pm \pi/2$')
        # ax0.plot(x_arr, y_after, c = 'b', label = r'$\alpha + \pi/2$')
        ax0.set_xlim(-1, 0.1) 
        ax0.set_ylim(-.2, .2)
        ax0.set_xlabel(r'X [$r_{\rm a}$]')
        ax0.set_ylabel(r'Y [$r_{\rm a}$]')
        ax0.legend(fontsize = 16, loc = 'upper left')
        
        plt.tight_layout()


if what == 'max_compr':
    checks = ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']
    colors_checks = ['C1', 'yellowgreen', 'darkviolet']
    names_checks = ['Low', 'Middle', 'High']
    markers_checks = ['x', 'o', 's']
    ratio_of_pi = 2
    vmin_ang = -np.pi/8
    vmax_ang = np.pi/4

    # to look how the expansion/compression changes with time
    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (10,10))
    ax1.set_ylabel(r'$\Delta (\alpha+\pi$/' + f'{ratio_of_pi}) /' + r'$\Delta (\alpha-\pi$/' + f'{ratio_of_pi})')
    ax2.set_ylabel(r'H ($\alpha+\pi$/' + f'{ratio_of_pi}) /' + r'H($\alpha-\pi$/' + f'{ratio_of_pi})')

    # to plot where is the maximum compression
    fig_compr, ax1_compr = plt.subplots(1,1, figsize = (12,7))
    ax1_compr.set_ylabel(r'H$_{\rm min} [R_\star]$')

    for j, check in enumerate(checks):
        # if check != 'HiResNewAMR':
        #     continue
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        snaps, Lum, tfbs = split_data_red(check)
        # len_arr = np.argmin(np.abs(tfbs - 1.2))
        # print(check, snaps[len_arr])
        # snaps, Lum, tfbs = snaps[:len_arr], Lum[:len_arr], tfbs[:len_arr]

        after_before_width = []
        after_before_h = []
        tfb_befaft = []
        theta_compr = np.zeros(len(snaps))
        height_compr = np.zeros(len(snaps))
        width_compr = np.zeros(len(snaps))
        Nheight_compr = np.zeros(len(snaps))
        Nwidth_compr = np.zeros(len(snaps))

        for i, tfb in enumerate(tfbs):
            if tfb > 1.1:
                continue
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

            if theta_inc < -np.pi or theta_inc > 0:
                continue
            after_before_width.append(width[idx_after]/ width[idx_before])
            after_before_h.append(height[idx_after]/ height[idx_before])
            tfb_befaft.append(tfb)
        
        ax1.plot(tfb_befaft, after_before_width, c = colors_checks[j], label = names_checks[j])
        ax2.plot(tfb_befaft, after_before_h, c = colors_checks[j])
    
        img_compr = ax1_compr.scatter(tfbs, height_compr/Rstar, c = theta_compr, marker = markers_checks[j], label = names_checks[j], vmin = vmin_ang, vmax = vmax_ang, cmap = 'rainbow', s = 50)
    
    cbar = plt.colorbar(img_compr)
    cbar.set_label(r'$\alpha$ [rad]')
    # Define the exact ticks you want:
    ticks = np.array([vmin_ang, 0, np.pi/8,  vmax_ang])
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_pi_frac))
    ax1_compr.legend(fontsize = 18)
    # ax1_compr.set_ylim(0,1)
    ax1.legend(fontsize = 18)
    ax2.set_xlabel(r't [t$_{\rm fb}$]')
    original_ticks = ax1.get_xticks()
    mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
    new_ticks = np.concatenate((original_ticks, mid_ticks))
    labels = [f'{tick:.1f}' if tick in original_ticks else '' for tick in new_ticks]
    for ax in [ax1, ax2, ax1_compr]:
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(labels)
        ax.tick_params(which = 'major', length = 10)
        ax.tick_params(which = 'minor', length = 5)
        ax.set_xlim(0.01, 1.1)
        ax.grid()
        if ax in [ax1, ax2]:
            ax.axhline(y=1, color='grey', linestyle = '--')
            ax.set_yscale('log')
            ax.set_ylim(1e-1, 5)
        