import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Utilities.operators import make_tree
import Utilities.sections as sec
import src.orbits as orb
from Utilities.time_extractor import days_since_distruption
matplotlib.rcParams['figure.dpi'] = 150

#
## PARAMETERS
#

#%%
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
check = 'Low' # 'Low' or 'HiRes' or 'Res20'
snap = '164'
is_tde = True

#
## CONSTANTS
#

G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rp
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
saving_fig = f'Figs/{folder}/{check}'
print(f'We are in: {path}, \nWe save in: {saving_fig}')

#
## MAIN
#

do_dMdE = False
compare_resol = False
compare_times = False

do_dMds = True

plot = True
save = True


#%%
if do_dMdE:
    data = make_tree(path, snap, is_tde, energy = False)
    dim_cell = data.Vol**(1/3) # according to Elad
    tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    Rcyl = np.sqrt(data.X**2 + data.Y**2)
    Vcyl = np.sqrt(data.VX**2 + data.VY**2)
    orbital_enegy = orb.orbital_energy(Rcyl, Vcyl, G, Mbh)

    # Normalisation (what on the x axis you call \Delta E)
    norm = Mbh/Rt * (Mbh/Rstar)**(-1/3)
    print('Normalization for energy:', norm)

    # Energy bins 
    OE = orbital_enegy / norm #or use orbital_energy_mid
    bins = np.arange(-2, 2, .1)
    # compute mass per bin
    hist, bins = np.histogram(OE, bins = bins)
    mass_sum = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        idx = np.where((OE > bins[i]) & (OE < bins[i+1]))
        mass_sum[i] = np.sum(data.Mass[idx])
    # dM/dE
    dm_dE = mass_sum / (np.diff(bins) * norm) # multiply by norm because we normalised the energy

    #%%
    if save:
        try:
            file = open(f'data/{folder}/dMdE_time{np.round(tfb,1)}.txt', 'r')
            # Perform operations on the file
            file.close()
        except FileNotFoundError:
            with open(f'data/{folder}/dMdE_time{np.round(tfb,1)}.txt','a') as fstart:
                # if file doesn'exist
                fstart.write(f'# Energy bins normalised (by DeltaE = {norm}) \n')
                fstart.write((' '.join(map(str, bins[:-1])) + '\n'))

        with open(f'data/{folder}/dMdE_time{np.round(tfb,1)}.txt','a') as file:
            file.write(f'# Check {check}, snap {snap} \n')
            file.write((' '.join(map(str, dm_dE)) + '\n'))
            file.close()

    #%%
    if plot:
        # Section at the midplane
        midplane = np.abs(data.Z) < dim_cell
        X_mid, Y_mid, VX_mid, VY_mid, Rcyl_mid, Mass_mid, Den_mid, orbital_enegy_mid = \
            sec.make_slices([data.X, data.Y, data.VX, data.VY, Rcyl, data.Mass, data.Den, orbital_enegy], midplane)

        # plot the mass distribution with respect to the energy
        fig, ax = plt.subplots(1,2, figsize = (8,4))
        ax[0].scatter(bins[:-1], dm_dE, c = 'k', s = 20)
        ax[0].set_xlabel(r'$\log_{10}E/\Delta E$', fontsize = 16)
        ax[0].set_ylabel('dM/dE', fontsize = 16)
        ax[0].set_xlim(-3,3)
        ax[0].set_ylim(1e-8, 2e-2)
        ax[0].set_yscale('log')

        img = ax[1].scatter(X_mid, Y_mid, c = np.log10(Den_mid), s = .1, cmap = 'jet', vmin = -11, vmax = -6)
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\log_{10}$ Density', fontsize = 16)
        ax[1].set_xlim(-500,100)
        ax[1].set_ylim(-200,50)
        ax[1].set_xlabel(r'X [$R_\odot$]', fontsize = 18)
        ax[1].set_ylabel(r'Y [$R_\odot$]', fontsize = 18)

        plt.suptitle(f'check: {check}, ' + r't/t$_{fb}$ = ' + str(np.round(tfb,3)), fontsize = 16)
        plt.tight_layout()
        if save:
            plt.savefig(f'{saving_fig}/EnM_{snap}.png')
        plt.show()
    #%%
    if compare_times:
        data = np.loadtxt(f'data/{folder}/dMdE.txt')
        bin_plot = data[0]
        data2 = data[1]
        data3 = data[4]
        data4 = data[5]
        
        plt.figure()
        plt.scatter(bin_plot, data2, c = 'b', s = 40, label = '0.2')
        plt.scatter(bin_plot, data3, c = 'k', s = 30, label = '0.3')
        plt.scatter(bin_plot, data4, c = 'r', s = 10, label = '0.7')
        plt.xlabel(r'$\log_{10}E/\Delta E$', fontsize = 16)
        plt.ylabel('dM/dE', fontsize = 16)
        plt.yscale('log')
        plt.legend()
        if save:
            plt.savefig(f'{saving_fig}/dMdE_times.png')
        plt.show()

    #%%
    if compare_resol:
        time_chosen = 0.5 #np.round(tfb,1)
        data = np.loadtxt(f'data/{folder}/dMdE_time{time_chosen}.txt')
        bin_plot = data[0]
        dataL = data[1]
        dataMiddle = data[2]
        dataHigh = data[3]
        
        plt.figure()
        plt.scatter(bin_plot, dataL, c = 'k', s = 35, label = 'Low')
        plt.scatter(bin_plot, dataMiddle, c = 'r', s = 20, label = 'Middle')
        plt.scatter(bin_plot, dataHigh, c = 'b', s = 10, label = 'High')
        plt.xlabel(r'$\log_{10}E/\Delta E$', fontsize = 16)
        plt.ylabel('dM/dE', fontsize = 16)
        plt.yscale('log')
        plt.legend()
        plt.title(r't/t$_{fb}$ = ' + str(time_chosen), fontsize = 16)
        if save:
            plt.savefig(f'Figs/{folder}/dMdE_time{time_chosen}.png')
        plt.show()
    
#%%###########
if do_dMds:
    from src.orbits import find_arclenght
    from Utilities.sections import transverse_plane, make_slices

    step = 0.02
    data = make_tree(path, snap, is_tde, energy = False)
    dim_cell = data.Vol**(1/3) # according to Elad
    stream = np.load(f'data/{folder}/stream_Low{snap}_{step}.npy')
    theta_arr, indeces_orbit = stream[0], stream[1].astype(int)
    x_orbit, y_orbit, z_orbit, den_orbit = data.X[indeces_orbit], data.Y[indeces_orbit], data.Z[indeces_orbit], data.Den[indeces_orbit]
    r_orbit = np.sqrt(x_orbit**2 + y_orbit**2)
    s_array, idx = find_arclenght(theta_arr, [x_orbit, y_orbit], params = None, choose = 'maxima')
    
    threshold = np.zeros(len(s_array)-1)
    dm = np.zeros(len(s_array)-1)
    for i in range(len(dm)):
        condition_T, x_Tplane, x0 = transverse_plane(data.X, data.Y, dim_cell, x_orbit, y_orbit, i, coord = True)
        z_plane, mass_plane= make_slices([data.Z, data.Mass], condition_T)
        # Restrict to not keep points too far away. Important for theta=0 or you take the stream at apocenter
        thresh = 3 * Rstar * (r_orbit[i]/Rp)**(1/3)
        condition_x = np.abs(x_Tplane) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        mass_plane = mass_plane[condition]
        dm[i] =  np.sum(mass_plane) 
        threshold[i] = thresh
    dm_ds = dm / np.diff(s_array)

    check1 = 'HiRes'
    path1 = f'/Users/paolamartire/shocks/TDE/{folder}{check1}/{snap}'
    data1 = make_tree(path1, snap, is_tde, energy = False)
    dim_cell1 = data1.Vol**(1/3) # according to Elad
    tfb = days_since_distruption(f'{path1}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    stream1 = np.load(f'data/{folder}/stream_{check1}{snap}_{step}.npy')
    theta_arr1, indeces_orbit1 = stream1[0], stream1[1].astype(int)
    x_orbit1, y_orbit1, z_orbit1, den_orbit1 = data1.X[indeces_orbit1], data1.Y[indeces_orbit1], data1.Z[indeces_orbit1], data1.Den[indeces_orbit1]
    r_orbit1 = np.sqrt(x_orbit1**2 + y_orbit1**2)
    s_array1, idx1 = find_arclenght(theta_arr1, [x_orbit1, y_orbit1], params = None, choose = 'maxima')
    dm1 = np.zeros(len(s_array1)-1)
    threshold1 = np.zeros(len(s_array1)-1)
    for i in range(len(dm1)):
        condition_T1, x_Tplane1, x01 = transverse_plane(data1.X, data1.Y, dim_cell1, x_orbit1, y_orbit1, i, coord = True)
        z_plane1, mass_plane1= make_slices([data1.Z, data1.Mass], condition_T1)
        thresh1 = 3 * Rstar * (r_orbit1[i]/Rp)**(1/3)
        condition_x1 = np.abs(x_Tplane1) < thresh1
        condition_z1 = np.abs(z_plane1) < thresh1
        condition1 = condition_x1 & condition_z1
        mass_plane1 = mass_plane1[condition1]
        dm1[i] =  np.sum(mass_plane1) 
        threshold1[i] = thresh1
    dm_ds1 = dm1 / np.diff(s_array1)

    check2 = 'Res20'
    snap2 = '169'
    path2 = f'/Users/paolamartire/shocks/TDE/{folder}{check2}/{snap2}'
    data2 = make_tree(path2, snap2, is_tde, energy = False)
    dim_cell2 = data2.Vol**(1/3) # according to Elad
    stream2 = np.load(f'data/{folder}/stream_{check2}{snap2}_{step}.npy')
    theta_arr2, indeces_orbit2 = stream2[0], stream2[1].astype(int)
    x_orbit2, y_orbit2, z_orbit2, den_orbit2 = data2.X[indeces_orbit2], data2.Y[indeces_orbit2], data2.Z[indeces_orbit2], data2.Den[indeces_orbit2]
    r_orbit2 = np.sqrt(x_orbit2**2 + y_orbit2**2)
    s_array2, _ = find_arclenght(theta_arr2, [x_orbit2, y_orbit2], params = None, choose = 'maxima')
    dm2 = np.zeros(len(s_array2)-1)
    threshold2 = np.zeros(len(s_array2)-1)
    for i in range(len(dm2)):
        condition_T2, x_Tplane2, _ = transverse_plane(data2.X, data2.Y, dim_cell2, x_orbit2, y_orbit2, i, coord = True)
        z_plane2, mass_plane2= make_slices([data2.Z, data2.Mass], condition_T2)
        thresh2 = 3 * Rstar * (r_orbit2[i]/Rp)**(1/3)
        condition_x2 = np.abs(x_Tplane2) < thresh2
        condition_z2 = np.abs(z_plane2) < thresh2
        condition2 = condition_x2 & condition_z2
        mass_plane2 = mass_plane2[condition2]
        dm2[i] =  np.sum(mass_plane2) 
        threshold2[i] = thresh2
    dm_ds2 = dm2/ np.diff(s_array2)
    
    #%%
    if plot:
        ratioLM = 1 - dm_ds1[:230]/dm_ds[:230]
        ratioMH = 1- dm_ds2[60:230]/dm_ds1[60:230]
        fig, (ax1,ax2) = plt.subplots(2,1)
        ax1.plot(theta_arr[:230], dm_ds[:230], c = 'k', label = 'Low')
        ax1.plot(theta_arr1[:230], dm_ds1[:230], c = 'r', label = 'Middle')
        ax1.plot(theta_arr2[60:230], dm_ds2[60:230], c = 'b', label = 'High')
        ax1.legend()
        ax1.grid()
        ax1.set_ylabel('dM/ds', fontsize = 16)
        ax1.set_yscale('log')

        ax2.plot(theta_arr[:230], ratioLM, c = 'r', label = '1 - Middle/Low')
        ax2.plot(theta_arr[60:230], ratioMH, c = 'b', label = '1- High/Middle')
        ax2.grid()
        ax2.legend()
        ax2.set_xlabel(r'$\theta$', fontsize = 16)
        if save:
            plt.savefig(f'Figs/{folder}/dMds_time{np.round(tfb,1)}.png')
        plt.show()

        fig, (ax1,ax2) = plt.subplots(2,1)
        ax1.plot(theta_arr[:230], dm[:230], c = 'k', label = 'Low')
        ax1.plot(theta_arr1[:230], dm1[:230], c = 'r', label = 'Middle')
        ax1.plot(theta_arr2[60:230], dm2[60:230], c = 'b', label = 'High')
        ax1.set_yscale('log')
        ax1.set_ylabel('dM', fontsize = 16)
        ax1.grid()
        ax2.plot(theta_arr[:230-1], np.diff(s_array[:230]), c = 'k', label = 'Low')
        ax2.plot(theta_arr1[:230-1], np.diff(s_array1[:230]), c = 'r', label = 'Middle')
        ax2.plot(theta_arr2[60:230-1], np.diff(s_array2[60:230]), c = 'b', label = 'High')
        ax2.grid()
        ax2.set_ylabel('ds', fontsize = 16)
        ax2.set_xlabel(r'$\theta$', fontsize = 16)
        plt.suptitle('dM and ds')
        if save:
            plt.savefig(f'Figs/{folder}/dMANDds_time{np.round(tfb,1)}.png')
        plt.show()

        plt.plot(theta_arr[:230], threshold[:230], c = 'k', label = 'Low')
        plt.plot(theta_arr1[:230], threshold1[:230], c = 'r', label = 'Middle')
        plt.plot(theta_arr2[60:230], threshold2[60:230], c = 'b', label = 'High')
        plt.ylabel('Threshold for mass integration in the TZ plane')
        plt.xlabel(r'$\theta$', fontsize = 16)
        plt.title(r'Threshold : $3 (R_{max}/R_p)^{1/3}$')
        plt.grid()
        plt.legend()
        if save:
            plt.savefig(f'Figs/{folder}/threshold_time{np.round(tfb,1)}.png')
        plt.show()
# %%
