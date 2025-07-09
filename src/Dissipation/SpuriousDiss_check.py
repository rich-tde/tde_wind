""" Checks for the initial high dissipation"""
# %%
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

import numpy as np
from Utilities.sections import make_slices
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
check = 'NewAMR' # '' or 'HiRes' or 'LowRes'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
Rp =  Rt / beta
Rs = 2*prel.G*Mbh/prel.csol_cgs**2
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
    snaps, tfb = snaps[tfb<0.3], tfb[tfb<0.30] # select only the first 30% of the fall
    
    with open(f'{abspath}/data/{folder}/Diss/spuriousDiss_{check}_days.txt', 'w') as file:
            file.write(f'# In spuriousDiss_{check}, you will find internal, orbital+, orbital-, diss+ energy, all cut in density and R>0.2*apo.')
            file.write(f'# {folder} \n' + ' '.join(map(str, snaps)) + '\n')
            file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
            file.close()

    ie = np.zeros(len(snaps))
    orb_en_pos = np.zeros(len(snaps))
    orb_en_neg = np.zeros(len(snaps))
    diss_pos = np.zeros(len(snaps))
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
        cut = np.logical_and(den > 1e-19, Rsph > 0.2*apo) # cut in density and in radius to be far away from the BH
        orb_en_cut, ie_cut, diss_cut = \
            make_slices([orb_en, ie, diss], cut)

        # total energies with only the cut in density (not in radiation)
        ie[i] = np.sum(ie_cut)
        orb_en_pos[i] = np.sum(orb_en_cut[orb_en_cut > 0])
        orb_en_neg[i] = np.sum(orb_en_cut[orb_en_cut < 0])
        diss_pos[i] = np.sum(diss_cut[diss_cut > 0])

    np.save(f'{abspath}/data/{folder}/Diss/spuriousDiss_{check}.npy', [ie, orb_en_pos, orb_en_neg, diss_pos])

else:
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

    #%%
    # ln_T = np.loadtxt(f'Tfile.txt') 
    # ln_Rho = np.loadtxt(f'density.txt') 
    # ln_U = np.loadtxt(f'Ufile.txt')
    # print(np.shape(ln_T), np.shape(ln_Rho), np.shape(ln_U))

    # plt.figure(figsize=(12, 12))
    # img = plt.pcolormesh(ln_T, ln_Rho, ln_U.T, cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    # cbar = plt.colorbar(img, label=r'$\ln U$')



