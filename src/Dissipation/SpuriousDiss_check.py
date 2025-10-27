""" Checks for the initial high dissipation. 
if alice:
    - computes the total internal, orbital+, orbital-, diss+ energy, all cut in density and, if where = '', at R>0.2*apo.
else:
    - plots the energies computed in alice for the 3 res
    - plots ELad's Utable and Diss-T scatter plot
    - plots the widths of the stream for 2 res
""" 
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    prepath = '/data1/martirep/shocks/shock_capturing'
    compute = True
    from Utilities.selectors_for_snap import select_snap
else:
    prepath = abspath
    compute = False
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.colors as colors
    from scipy.integrate import cumulative_trapezoid

import numpy as np
from Utilities.sections import make_slices, transverse_plane, rotate_coordinates
import Utilities.prelude as prel
from src import orbits as orb
from Utilities.operators import make_tree

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5 # 'n1.5'
compton = 'Compton'
check = 'HiResNewAMR' # '' or 'HiRes' or 'LowRes'
where = '_everywhere'
params = [Mbh, Rstar, mstar, beta]
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'


Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
Rp =  Rt / beta
Rs = 2*prel.G*Mbh/prel.csol_cgs**2
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
    # snaps, tfb = snaps[tfb<0.3], tfb[tfb<0.30] # select only the first 30% of the fall
    
    with open(f'{prepath}/data/{folder}/Diss/spuriousDiss_{check}_days.txt', 'w') as file:
            file.write(f'# In spuriousDiss_{check}, you will find internal, orbital+, orbital-, diss+ energy, all cut in density and R>0.2*apo.')
            file.write(f'# {folder} \n' + ' '.join(map(str, snaps)) + '\n')
            file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
            file.close()

    ie_sum = np.zeros(len(snaps))
    orb_en_pos_sum = np.zeros(len(snaps))
    orb_en_neg_sum = np.zeros(len(snaps))
    diss_pos_sum = np.zeros(len(snaps))
    diss_neg_sum = np.zeros(len(snaps))
    for i, snap in enumerate(snaps):
        print(snap, flush = True)

        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        data = make_tree(path, snap, energy = True)
        X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, diss_den = \
            data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Mass, data.Vol, data.Den, data.IE, data.Diss
        diss = diss_den * vol
        ie = ie_den * vol
        # cut in density NOT in radiation
        Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
        vel = np.sqrt(np.power(VX, 2) + np.power(VY, 2) + np.power(VZ, 2))
        orb_en = orb.orbital_energy(Rsph, vel, mass, prel.G, prel.csol_cgs, Mbh, R0)
        if where == '':
            cut = np.logical_and(den > 1e-19, Rsph > 0.2*apo) # cut in density and in radius to be far away from the BH
        if where == '_everywhere':
            cut = den > 1e-19
        orb_en_cut, ie_cut, diss_cut = \
            make_slices([orb_en, ie, diss], cut)

        # total energies with only the cut in density (not in radiation)
        ie_sum[i] = np.sum(ie_cut)
        orb_en_pos_sum[i] = np.sum(orb_en_cut[orb_en_cut > 0])
        orb_en_neg_sum[i] = np.sum(orb_en_cut[orb_en_cut < 0])
        diss_pos_sum[i] = np.sum(diss_cut[diss_cut > 0])
        diss_neg_sum[i] = np.sum(diss_cut[diss_cut < 0])

    np.save(f'{prepath}/data/{folder}/Diss/spuriousDiss_{check}{where}.npy', [ie_sum, orb_en_pos_sum, orb_en_neg_sum, diss_pos_sum, diss_neg_sum])

else:
    how_to_check = 'ionization' # 'energies' or 'widths' or 'ionization'
    t_fall_days = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)
    tfall_cgs = t_fall_days * 24 * 3600 
    recomb_en = 13.6*prel.ev_to_erg * mstar*prel.Msol_cgs/prel.m_p_cgs

    if how_to_check == 'energies':
        checks = ['', 'NewAMR']
        check_label = ['Old', 'NewAMR']

        fig, ax = plt.subplots(1,2, figsize=(16, 6))
        for i, check in enumerate(checks):
            if check in ['NewAMR', 'HiResNewAMR', 'LowResNewAMR']:
                folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
            else:
                folder = f'opacity_tests/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

            snaps, tfbs = np.loadtxt(f'{abspath}/data/{folder}/Diss/spuriousDiss_{check}_days.txt')
            ie_sum, orb_en_pos_sum, orb_en_neg_sum, diss_pos_sum, diss_neg_sum = \
                np.load(f'{abspath}/data/{folder}/Diss/spuriousDiss_{check}{where}.npy', allow_pickle=True)
            ie_sum *= prel.en_converter # convert to erg
            orb_en_neg_sum *= prel.en_converter
            diss_pos_sum *= prel.en_converter / prel.tsol_cgs # convert to erg/s
            diss_neg_sum *= prel.en_converter / prel.tsol_cgs # convert to erg/s
            # integrate diss_pos_sum over tfbs
            tfbs_cgs = tfbs * tfall_cgs # convert to cgs
            diss_pos_int = cumulative_trapezoid(diss_pos_sum, tfbs_cgs, initial = 0)
            diss_neg_int = cumulative_trapezoid(np.abs(diss_neg_sum), tfbs_cgs, initial = 0)
            
            ax[i].plot(tfbs, ie_sum, c = 'forestgreen', label = r'$E_{\rm ie}$') 
            ax[i].plot(tfbs, np.abs(orb_en_neg_sum), c = 'deepskyblue', label = r'$|E_{\rm orb}|$ bound material')
            ax[i].plot(tfbs, diss_pos_int, c = 'orangered', label = r'$E_{\rm diss, +}$')
            ax[i].plot(tfbs, diss_neg_int, c = 'darkviolet', label = r'$|E_{\rm diss, -}|$')
            ax[i].set_yscale('log')
            ax[i].set_xlabel(r'$t [t_{\rm fb}]$')
            ax[i].axhline(recomb_en, c = 'k', ls = '--', label = r'$E_{\rm rec}$')
            ax[i].set_title(f'{check_label[i]}', fontsize = 20)

        ax[0].legend(fontsize = 16)
        ax[0].set_ylabel(r'Energy [erg]')
        if where == '_everywhere':
            plt.suptitle(r'All material with $\rho > 1e-19$', fontsize = 20)
        else:
            plt.suptitle(r'R > 0.2 R$_a$, $\rho > 1e-19$', fontsize = 20)

    if how_to_check == 'ionization':
        snap = 22
        check = 'HiResNewAMR'
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

        path = f'{abspath}/TDE/{folder}/{snap}'
        tfb = np.loadtxt(f'{path}/tfb_{snap}.txt')
        ln_T = np.loadtxt(f'Tfile.txt') # 298980 elements, (first) 495 unique
        ln_T = ln_T[:495]
        ln_Rho = np.loadtxt(f'density.txt') # 604 elements
        ln_U = np.loadtxt(f'Ufile.txt')
        ln_U = ln_U.reshape(495,604) # 495 rows (y for colormesh), 604 columns (x for colormesh)
        # substract vectorially ln_rho to ln_u 
        ln_U_rho = ln_U - ln_Rho[None:,]
        log10_T = np.log10(np.exp(ln_T))
        log10_Rho = np.log10(np.exp(ln_Rho))
        log10_U_rho = np.log10(np.exp(ln_U_rho))

        plt.figure(figsize=(15, 12))
        img = plt.pcolormesh(log10_T, log10_Rho, log10_U_rho.T, cmap = 'jet', alpha = 0.7) 
        cbar = plt.colorbar(img, label=r'$\log_{10}$ specific i.e. [erg/g]')
        plt.xlabel(r'$\log_{10} (T)$ [K]')
        plt.ylabel(r'$\log_{10} (\rho)$ [g/cm$^3$]')

        data = make_tree(path, snap, energy = True)
        cut = data.Den > 1e-19
        X, Y, Z, Vol, Temp, Den, Diss_den, IE_den = \
            make_slices([data.X, data.Y, data.Z, data.Vol, data.Temp, data.Den, data.Diss, data.IE], cut)
        IE_den_cgs = IE_den * prel.en_den_converter
        Diss = Diss_den * Vol
        Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
        if np.sum(Diss[Diss>0]) > np.abs(np.sum(Diss[Diss<0])):
            cut_diss = Diss > 0
            print('Dissipation has positive sign')
        else:
            cut_diss = Diss < 0
            print('Dissipation has negative sign')
        mask_noinfall = np.logical_and(cut_diss, Rsph > 0.2*apo)
        R_noinfall, T_noinfall, Den_infall, Diss_noinfall = make_slices([Rsph, Temp, Den, Diss], mask_noinfall)
        Den_infall_cgs = Den_infall * prel.den_converter

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), dpi=150)
        img = ax1.scatter(T_noinfall*1e-4, Den_infall_cgs,  c = np.abs(Diss_noinfall*prel.en_converter)/prel.tsol_cgs, s = 1, cmap='rainbow', norm = LogNorm(vmin=1e38, vmax=1e43), rasterized=True)
        ax1.set_ylabel(r'$\rho$ [g/cm$^3$]')
        ax1.set_yscale('log')
        ax1.set_ylim(1e-9, 1)
        img = ax2.scatter(T_noinfall*1e-4, R_noinfall/apo, c = np.abs(Diss_noinfall*prel.en_converter)/prel.tsol_cgs, s = 1, cmap='rainbow', norm = LogNorm(vmin=1e38, vmax=1e43), rasterized=True)
        cbar = plt.colorbar(img, label=r'$|$Diss rate$|$ [erg/s]')
        ax2.set_ylabel(r'R/R$_{\rm a}$')
        ax2.set_ylim(0.2, 1)

        for ax in [ax1, ax2]:
            ax.loglog()
            ax.tick_params(axis='both', which='major', width=1.2, length=8, color = 'k')
            ax.tick_params(axis='both', which='minor', width=1, length=4, color = 'k')
            ax.set_xlim(1, 7)
        ax2.set_xlabel(r'T [$10^{4}$K]')
        plt.suptitle(f'run: {check}, points with $R>0.2R_a$, t = {np.round(tfb, 2)}' + r' $t_{\rm fb}$', fontsize=20)
        plt.tight_layout()

    if how_to_check == 'widths':
        snap = 106
        checks = ['NewAMR', '']
        color_checks = ['k', 'r']
        checks_label = ['New AMR', 'Old']
        fig, ax = plt.subplots(2,2, figsize=(20, 12))
        fig_w, ax_w = plt.subplots(1,1, figsize=(8, 8))
        for i, check in enumerate(checks):
            if check == '':
                idx_width = np.arange(190, 211) 
                folder = f'opacity_tests/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
            else:
                idx_width = np.arange(240, 261)
                folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

            x_denproj = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Denxarray.npy')
            y_denproj = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Denyarray.npy')
            flat_den = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Denproj{snap}.npy')
            flat_den_cgs = flat_den * prel.den_converter * prel.Rsol_cgs
            flat_diss = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Dissproj{snap}.npy')
            flat_diss_cgs = flat_diss * prel.en_den_converter * prel.Rsol_cgs / prel.tsol_cgs# [erg/s/cm2]
            # make =1 the nan values so they disappera with logcolor
            flat_diss_cgs_plot = flat_diss_cgs
            flat_diss_cgs_plot[np.isnan(flat_diss_cgs_plot)] = 1
            flat_diss_cgs_plot[flat_diss_cgs_plot == 0] = 1
            img = ax[i][0].pcolormesh(x_denproj/apo, y_denproj/apo, flat_den_cgs.T, cmap = 'plasma', \
                                norm = colors.LogNorm(vmin = 1e2, vmax = 5e7))
            
            cbar = plt.colorbar(img)
            cbar.set_label(r'Column density [g cm$^{-2}$]')
            
            img = ax[i][1].pcolormesh(x_denproj/apo, y_denproj/apo, flat_diss_cgs.T, \
                                cmap = 'viridis', norm = colors.LogNorm(vmin = 1e14, vmax = 1e19))
            
            cbar = plt.colorbar(img)
            cbar.set_label(r'E_{\rm diss} column density [erg s$^{-1}$cm$^{-2}]$')

            path = f'{abspath}/TDE/{folder}/{snap}' 
            data = make_tree(path, snap, energy = True)
            tfb = np.loadtxt(f'{abspath}/TDE/{folder}/{snap}/tfb_{snap}.txt')
            cut = data.Den > 1e-19
            X, Y, Z, mass, den, Vol, Mass, Press, Temp, vx, vy, vz, IE_den, Diss_den = \
                make_slices([data.X, data.Y, data.Z, data.Mass, data.Den, data.Vol, data.Mass, data.Press, data.Temp, data.VX, data.VY, data.VZ, data.IE, data.Diss], cut)
            dim_cell = Vol**(1/3) 
            Diss = Diss_den * Vol
            IE_spec = IE_den / den

            theta_arr, x_stream, y_stream, z_stream, thresh_cm = np.load(f'{abspath}/data/{folder}/WH/stream_{check}{snap}.npy', allow_pickle=True)
            # img = plt.scatter(x_stream/apo, y_stream/apo, c = np.arange(len(x_stream)), label = 'stream', cmap = 'jet')
            # cbar = plt.colorbar(img)
            stream = [theta_arr, x_stream, y_stream, z_stream, thresh_cm]
            indeces_boundary = np.load(f'{abspath}/data/{folder}/WH/indeces_boundary_{check}{snap}.npy')
            indeces_boundary_lowX, indeces_boundary_upX, indeces_boundary_lowZ, indeces_boundary_upZ = \
                indeces_boundary[:,0], indeces_boundary[:,1], indeces_boundary[:,2], indeces_boundary[:,3]
            indeces_all = np.arange(len(X))
            x_low_width, y_low_width = X[indeces_boundary_lowX], Y[indeces_boundary_lowX]
            x_up_width, y_up_width = X[indeces_boundary_upX], Y[indeces_boundary_upX]

            wh = np.loadtxt(f'{abspath}/data/{folder}/WH/wh_{check}{snap}.txt')
            theta_wh, width, N_width, height, N_height = wh[0], wh[1], wh[2], wh[3], wh[4]
                    
            ax_w.plot(theta_wh[idx_width], width[idx_width], c = color_checks[i], label = f'{checks_label[i]}')
            for k in range(2):   
                ax[i][k].plot(x_low_width[idx_width]/apo, y_low_width[idx_width]/apo, c = color_checks[i], label = f'{checks_label[i]}')
                ax[i][k].plot(x_up_width[idx_width]/apo, y_up_width[idx_width]/apo, c = color_checks[i])

            for i in range(2):
                ax[i][0].set_ylabel(r'Y [$R_{\rm a}$]')
                for j in range(2):
                    ax[j][i].set_xlim(-1, 0)
                    ax[j][i].set_ylim(-.5, .1)
                ax[1][i].set_xlabel(r'X [$R_{\rm a}$]')

        fig.suptitle(f't = {np.round(tfb, 2)}' +  r' t$_{\rm fb}$', fontsize = 22)
        fig.tight_layout()
        fig_w.suptitle(f't = {np.round(tfb, 2)}' +  r' t$_{\rm fb}$', fontsize = 22)
        ax_w.set_xlabel(r'$\theta$ [rad]', fontsize = 16)
        ax_w.set_ylabel(r'Width [$R_\odot$]', fontsize = 16)
        ax_w.set_ylim(1, 5)
        fig_w.tight_layout()
        
    if how_to_check == 'pancake':
        snap = 96
        path = f'{abspath}/TDE/{folder}/{snap}'
        tfb = np.loadtxt(f'{abspath}/TDE/{folder}/{snap}/tfb_{snap}.txt')
        data = make_tree(path, snap, energy = True)
        cut = data.Den > 1e-19 
        X, Y, Z, Vol, Mass, Den, VX, VY, VZ, Diss_den = \
            make_slices([data.X, data.Y, data.Z, data.Vol, data.Mass, data.Den, data.VX, data.VY, data.VZ, data.Diss], cut)
        dim_cell = Vol**(1/3)
        Diss = Diss_den * Vol
        # theta_arr = np.arange(-.5, .5, .1)   
        # from src.WH import find_transverse_com
        # x_stream, y_stream, z_stream, thresh_cm = find_transverse_com(X, Y, Z, dim_cell, Den, Mass, theta_arr, params, Rstar)
        theta_arr, x_stream, y_stream, z_stream, thresh_cm = np.load(f'{abspath}/data/{folder}/WH/stream/stream_{check}{snap}.npy', allow_pickle=True)
        idx_chosen = 244 #np.argmin(np.abs(theta_arr))

        condition_T, x_T = transverse_plane(X, Y, Z, dim_cell, x_stream, y_stream, z_stream, idx_chosen, Rstar, just_plane = False)
        x_plane, x_T_plane, y_plane, z_plane, dim_plane, mass_plane, VX_plane, VY_plane, VZ_plane, den_plane = \
            make_slices([X, x_T, Y, Z, dim_cell, Mass, VX, VY, VZ, Den], condition_T)
        # cut at the threshold
        below_thres = np.logical_and(np.abs(x_T_plane) < thresh_cm[idx_chosen], np.abs(z_plane) < thresh_cm[idx_chosen])
        x_plane, x_T_plane, y_plane, z_plane, dim_plane, mass_plane, VX_plane, VY_plane, VZ_plane, den_plane = \
            make_slices([x_plane, x_T_plane, y_plane, z_plane, dim_plane, mass_plane, VX_plane, VY_plane, VZ_plane, den_plane], below_thres)

        
        mid = np.abs(Z) < dim_cell
        X_mid, Y_mid, Diss_mid = make_slices([X, Y, Diss], mid)
        mid_plane = np.abs(z_plane) < dim_plane
        t_dot, _ = rotate_coordinates(VX_plane, VY_plane, x_stream, y_stream, idx_chosen, z_datas = None)
        E_kin_t = 0.5 * np.sum(mass_plane * t_dot**2) * prel.en_converter 
        E_kin_z = 0.5 * np.sum(mass_plane * VZ_plane**2) * prel.en_converter

        # load data energy
        snaps, tfbs = np.loadtxt(f'{abspath}/data/{folder}/Diss/spuriousDiss_{check}_days.txt')
        tfbs_cgs = tfbs * tfall_cgs # convert to cgs
        ie_sum, orb_en_pos_sum, orb_en_neg_sum, diss_pos_sum, diss_neg_sum = \
            np.load(f'{abspath}/data/{folder}/Diss/spuriousDiss_{check}{where}.npy', allow_pickle=True)
        time_en = np.argmin(np.abs(snaps - snap))
        ie_snap = ie_sum[time_en] * prel.en_converter # convert to erg
        orb_en_neg_snap = orb_en_neg_sum[time_en] * prel.en_converter
        diss_pos_sum *= prel.en_converter / prel.tsol_cgs # convert to erg/s
        diss_neg_sum *= prel.en_converter / prel.tsol_cgs # convert to erg/s
        diss_pos_int = cumulative_trapezoid(diss_pos_sum, tfbs_cgs, initial = 0)
        diss_neg_int = cumulative_trapezoid(np.abs(diss_neg_sum), tfbs_cgs, initial = 0)
        diss_pos_snap = diss_pos_int[time_en]
        diss_neg_snap = diss_neg_int[time_en]
        print(f'ie = {ie_snap}, \norb_en_neg = {orb_en_neg_snap}, \ndiss = {max(diss_pos_snap, diss_neg_snap)}, \nE_kin_t = {E_kin_t}')
        print(f'sum new kinetic energy = {E_kin_t + E_kin_z}')

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(27, 8), width_ratios=(1.5, 1))
        img = ax0.scatter(X_mid/apo, Y_mid/apo, c = Diss_mid * prel.en_converter/prel.tsol_cgs, s = 1, cmap = 'viridis', norm = colors.LogNorm(vmin = 1e28, vmax = 1e35))
        plt.colorbar(img,  label=r'Dissipation rate [erg/s]')
        ax0.scatter(x_stream/apo, y_stream/apo, c = 'k', s = 4)
        ax0.scatter(x_plane[mid_plane]/apo, y_plane[mid_plane]/apo, c = 'b', s = 10)
        ax0.scatter(x_stream[idx_chosen]/apo, y_stream[idx_chosen]/apo, c = 'r', s = 20)
        ax0.set_xlabel(r'X [$R_{\rm a}$]')
        ax0.set_ylabel(r'Y [$R_{\rm a}$]')
        ax0.set_xlim(-.8, 0.1)
        ax0.set_ylim(-.5, .2)
        img = ax1.scatter(x_T_plane, z_plane, c = den_plane, s = 40, cmap = 'rainbow', norm = colors.LogNorm(vmin = np.percentile(den_plane, 40), vmax = np.max(den_plane)))
        cbar = plt.colorbar(img)
        ax1.quiver(x_T_plane, z_plane, t_dot, VZ_plane, color = 'k', angles='xy', scale_units='xy', scale=40, width = 0.002)
        cbar.set_label(r'Den [$M_\odot/R_\odot^3$]')
        ax1.set_xlabel(r'T [$R_\odot$]')
        ax1.set_ylabel(r'Z [$R_\odot$]') 
        plt.suptitle(f'TZ plane, t = {np.round(tfb, 2)}' +  r' t$_{\rm fb}, \theta$' + f' = {np.round(theta_arr[idx_chosen], 2)} [rad]', fontsize = 22)
        plt.tight_layout()

        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(27, 8), width_ratios=(1.5, 1))
        # ax0.plot(x_stream/apo, y_stream/apo, c = 'k')
        # ax0.scatter(x_plane[mid_plane]/apo, y_plane[mid_plane]/apo, c = 'b', s = 10)
        # ax0.scatter(x_stream[idx_chosen]/apo, y_stream[idx_chosen]/apo, c = 'r', s = 20)
        # ax0.set_xlabel(r'X [$R_{\rm a}$]')
        # ax0.set_ylabel(r'Y [$R_{\rm a}$]')
        # ax0.set_xlim(-1.2, 0.1)
        # ax0.set_ylim(-.2, .2)
        # img = ax1.scatter(x_plane, z_plane, c = den_plane, s = 20, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-10, vmax = 1e-7))
        # cbar = plt.colorbar(img)
        # ax1.quiver(x_plane, z_plane, VX_plane, VZ_plane, color = 'k', angles='xy', scale_units='xy', scale=80, width = 0.002)
        # cbar.set_label(r'Den [$M_\odot/R_\odot^3$]')
        # ax1.set_xlabel(r'X [$R_\odot$]')
        # ax1.set_ylabel(r'Z [$R_\odot$]') 
        # ax1.set_xlim(-5+x_stream[idx_chosen],5+x_stream[idx_chosen])
        # ax1.set_ylim(-2+z_stream[idx_chosen],2+z_stream[idx_chosen])
        # plt.suptitle(f'XZ plane, t = {np.round(tfb, 2)}' +  r' t$_{\rm fb}, \theta$' + f' = {np.round(theta_arr[idx_chosen], 2)} [rad]', fontsize = 22)
        # plt.tight_layout()


# %%
