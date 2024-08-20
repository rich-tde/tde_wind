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
## CONSTANTS
#

G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

#%%
## PARAMETERS STAR AND BH
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rp
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

#
## MAIN
#

# Choose what to do
cutoffRes20 = True
print('Cutoff for the highest res (Res20):', cutoffRes20)
do_dMdE = True
compare_resol = False
compare_times = False
do_Ehist = False
E_in_time = False
do_dMds = False
plot = True
save = False

#%%
if do_dMdE:
    check = 'Low' # 'Low' or 'HiRes' or 'Res20'
    snap = '164'
    path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
    data = make_tree(path, snap, energy = False)
    dim_cell = data.Vol**(1/3) # according to Elad
    tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    mass = data.Mass
    R = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
    V = np.sqrt(data.VX**2 + data.VY**2 + data.VZ**2)
    orbital_enegy = orb.orbital_energy(R, V, mass, G, c, Mbh)
    specific_orbital_energy = orbital_enegy / mass

    # Cutoff for the highest res (Res20)
    if check == 'Res20' and cutoffRes20:
        cutoff = data.X > -175
        mass = mass[cutoff]
        specific_orbital_energy = specific_orbital_energy[cutoff]

    # Normalisation (what on the x axis you call \Delta E)
    norm = Mbh/Rt * (Mbh/Rstar)**(-1/3)
    print('Normalization for energy:', norm)

    # (Specific) energy bins 
    specOE_norm = specific_orbital_energy/norm #or use orbital_energy_mid
    bins = np.arange(-2, 2, .1)
    mass_binned, bins_edges = np.histogram(specOE_norm, bins = bins, weights=mass) # sum the mass in each bin (bins done on specOE_norm)
    dm_dE = mass_binned / (np.diff(bins_edges)*norm)

    saving_fig = f'Figs/{folder}/{check}'
    if save:
        (f'Saving data in: data/{folder}')
        try:
            file = open(f'data/{folder}/dMdE_time{np.round(tfb,1)}.txt', 'r')
            # Perform operations on the file
            file.close()
        except FileNotFoundError:
            with open(f'data/{folder}/dMdE_time{np.round(tfb,1)}.txt','a') as fstart:
                # if file doesn'exist
                fstart.write(f'# Energy bins normalised (by DeltaE = {norm}) \n')
                fstart.write((' '.join(map(str, bins_edges[:-1])) + '\n'))
        with open(f'data/{folder}/dMdE_time{np.round(tfb,1)}.txt','a') as file:
            if check == 'Res20' and cutoffRes20:
                file.write(f'# Check {check}, snap {snap} using only data with data.X>-175 \n')
            else:
                file.write(f'# Check {check}, snap {snap} \n')
            file.write((' '.join(map(str, dm_dE)) + '\n'))
            file.close()
    
    if plot:
        # Section at the midplane
        midplane = np.abs(data.Z) < dim_cell
        X_mid, Y_mid, Den_mid = \
            sec.make_slices([data.X, data.Y, data.Den], midplane)
        # plot the mass distribution with respect to the energy
        fig, ax = plt.subplots(1,2, figsize = (8,4))
        # ax[0].scatter(bins_edges[:-1], mass_binned, c = 'r', s = 10, alpha = 0.4, label = 'dM')
        # ax[0].hist(specOE_norm, bins = bins, weights = mass, color = 'r', alpha = 0.2)
        ax[0].scatter(bins_edges[:-1], dm_dE, c = 'k', s = 30)
        ax[0].set_xlabel(r'$E_{spec}/\Delta E$', fontsize = 16)
        ax[0].set_ylabel(r'$\log_{10}$(dM/d$E_{spec}$)', fontsize = 16)
        ax[0].set_xlim(-3,3)
        ax[0].set_ylim(1e-8, 4e-2)
        ax[0].set_yscale('log')
        ax[0].legend()

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
    data = np.loadtxt(f'data/{folder}/dMdE_{check}.txt')
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
    time_chosen = 0.1 #np.round(tfb,1)
    data = np.loadtxt(f'data/{folder}/dMdE_time{time_chosen}.txt')
    bin_plot = data[0]
    dataL = data[1]
    dataMiddle = data[2]
    if cutoffRes20:
        dataHigh = data[4]
    else:
        dataHigh = data[3]
    
    plt.scatter(bin_plot, dataL, c = 'k', s = 35, label = 'Low')
    plt.scatter(bin_plot, dataMiddle, c = 'r', s = 15, label = 'Middle')
    plt.scatter(bin_plot, dataHigh, c = 'b', s = 7, label = 'High')
    plt.xlabel(r'$E/\Delta E$', fontsize = 16)
    plt.ylabel(r'$(\log_{10}$dM/dE)', fontsize = 16)
    plt.yscale('log')
    plt.legend()
    if cutoffRes20:
        plt.title(r'Only points with $X>-175$ for the highest res, t/t$_{fb}$ = ' + str(time_chosen), fontsize = 16)
        if save:
            plt.savefig(f'Figs/{folder}/multiple/dMdE_time{time_chosen}_cutRes20.png')
    else:
        plt.title(r't/t$_{fb}$ = ' + str(time_chosen), fontsize = 16)
        if save:
            plt.savefig(f'Figs/{folder}/multiple/dMdE_time{time_chosen}.png')
    plt.grid()
    plt.show()

#%%###########
if do_Ehist:
    y_value = 'energy'
    checks = ['Low', 'HiRes']#, 'Res20']
    colors = ['k', 'r']#, 'b']
    alphas = [0.6, 0.8]#, 1]
    snap = 164
    fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2)
    for i in range(-1,-len(checks)-1,-1):
        check = checks[i]
        # snap = snaps[i]
        path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
        tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
        data = make_tree(path, snap, energy = True)
        Rsph = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
        vel = np.sqrt(data.VX**2 + data.VY**2 + data.VZ**2)
        mass, ie_den, Erad_den = data.Mass, data.IE, data.Erad
        ie_onmass = ie_den / data.Den
        ie = data.IE * data.Vol 
        Erad = data.Erad * data.Vol 
        orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
        orb_en_onmass = orb_en / mass

        # select only bound elements and not too far away
        bound_elements = np.logical_and(orb_en < 0, data.X > -1.5* np.abs(apo))
        mass_bound, ie_onmass_bound, orb_en_onmass_bound = \
            sec.make_slices([mass, ie_onmass, orb_en_onmass], bound_elements)
        ie_den_bound, Erad_den_bound, ie_bound, Erad_bound, orb_en_bound = \
            sec.make_slices([ie_den, Erad_den, ie, Erad, orb_en], bound_elements)
        
        energies = [np.sum(ie_bound), np.sum(Erad_bound), np.sum(orb_en_bound), np.sum(ie_bound + Erad_bound + orb_en_bound)]

        # with open(f'data/{folder}/boundE_{check}_EradasDen.txt','a') as file:
        #     file.write(f'# Check {check}, snap {snap} \n')
        #     file.write((' '.join(map(str, energies)) + '\n'))
        #     file.close()

        if y_value == 'energy':
            # bins
            if snap == 100:
                bins_intE = np.linspace(0, 2e-2, 50) 
                bins_radE = np.linspace(0, 1e-16, 50) 
                bins_neg_orbE = np.linspace(0,50,50)

            if snap == 115:
                bins_intE = np.linspace(0, 1.2e-2, 50) 
                bins_radE = np.linspace(0, 1e-15, 50) 
                bins_neg_orbE = np.linspace(0,50,50)

            if snap == 164:
                bins_intE = np.linspace(0, 0.018, 50) 
                bins_radE = np.linspace(0, 5e-15, 50) 
                bins_neg_orbE = np.linspace(0,50,50)

            if snap == 199 or snap == 216:
                bins_intE = np.linspace(0, 0.05, 50) 
                bins_radE = np.linspace(0, 2e-15, 50) 
                bins_neg_orbE = np.linspace(0,50,50)

            # weights
            weight_ie = ie_bound
            weight_rad = Erad
            weight_orb = np.abs(orb_en_bound)

        elif y_value == 'mass':
            # bins
            if snap == 100:
                bins_intE = np.linspace(0, 2e-2, 50) 
                bins_radE = np.linspace(0, 1e-10, 50) 
                bins_neg_orbE = np.linspace(0,50,50)

            if snap == 115:
                bins_intE = np.linspace(0, 1.2e-2, 50) 
                bins_radE = np.linspace(0, 1e-11, 50) 
                bins_neg_orbE = np.linspace(0,50,50)

            if snap == 164:
                bins_intE = np.linspace(0, 0.018, 50) 
                bins_radE = np.linspace(0, 3e-13, 50) 
                bins_neg_orbE = np.linspace(0,50,50)

            if snap == 199 or snap == 216:
                bins_intE = np.linspace(0, 0.05, 50) 
                bins_radE = np.linspace(0, 1e-13, 50) 
                bins_neg_orbE = np.linspace(0,50,50)

            # weights
            weight_ie = mass_bound
            weight_rad = mass
            weight_orb = mass_bound

        ax1.hist(ie_onmass_bound, bins = bins_intE, weights = weight_ie, color = colors[i], alpha = alphas[i], label = f'check = {check}')
        ax2.hist(Erad_den, bins = bins_radE, weights = weight_rad, color = colors[i], alpha = alphas[i], label = f'check = {check}')
        ax3.hist(np.abs(orb_en_onmass_bound), bins = bins_neg_orbE, weights = weight_orb, color = colors[i], alpha = alphas[i], label = f'check = {check}')
        ax4.set_axis_off()

    ax1.legend(fontsize = 10)

    if y_value == 'energy':
        ax1.set_ylabel(r'Energy', fontsize = 16)
        ax2.set_ylabel(r'Energy', fontsize = 16)
        ax3.set_ylabel(r'$|$Energy$|$', fontsize = 16)
    elif y_value == 'mass':
        ax1.set_ylabel(r'Mass', fontsize = 16)
        ax2.set_ylabel(r'Mass', fontsize = 16)
        ax3.set_ylabel(r'Mass', fontsize = 16)

    ax1.set_title('Internal Energy')
    ax2.set_title('Radiation Energy')
    ax3.set_title('Orbital Energy')
    ax1.set_xlabel(r'Energy/Mass', fontsize = 16)
    ax2.set_xlabel(r'Energy/Vol', fontsize = 16)
    ax3.set_xlabel(r'$|$Energy$|$/Mass', fontsize = 16)
    
    plt.suptitle(r'Bound cells with $X>-1.5|R_a|$, t/t$_{fb}$ = ' + f'{np.round(tfb,1)}', fontsize = 16)
    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/{y_value}_hist_time{np.round(tfb,1)}.png')
    plt.show()

#%%
if E_in_time:
    dataLow = np.loadtxt(f'data/{folder}/boundE_Low_EradasDen.txt')
    dataLow = dataLow.T
    dataMiddle = np.loadtxt(f'data/{folder}/boundE_HiRes_EradasDen.txt')
    dataMiddle = dataMiddle.T

    time = np.array([0.1, 0.2, 0.5, 0.8, 0.9]) #100, 115, 164, 199, 216
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax2.plot(time, dataLow[0], c = 'darkcyan', label = 'IE Low')
    ax2.plot(time, dataLow[1], '--', c = 'deepskyblue', label = 'RadE Low')
    ax1.plot(time, dataLow[2], c = 'royalblue', label = 'OE Low')
    ax1.plot(time, dataLow[3], '--', c = 'navy', label = 'TotE Low')

    ax2.plot(time, dataMiddle[0], c =    'orange', label = 'IE Middle')
    ax2.plot(time, dataMiddle[1], '--', c = 'darkorange', label = 'RadE Middle')
    ax1.plot(time, dataMiddle[2], c = 'coral', label = 'OE Middle')
    ax1.plot(time, dataMiddle[3], '--', c = 'maroon', label = 'TotE Middle')
    ax1.legend()
    ax2.legend()
    ax1.set_ylabel('Energy', fontsize = 18)
    ax2.set_ylabel('Energy', fontsize = 18)
    ax2.set_xlabel('t/t$_{fb}$', fontsize = 18)
    ax1.grid()
    ax2.grid()
    # plt.savefig(f'Figs/{folder}/multiple/Energy_time.pdf')
    plt.show()



#%%###########
# if do_dMds:
#     from src.orbits import find_arclenght
#     from Utilities.sections import transverse_plane, make_slices

#     step = 0.02
#     data = make_tree(path, snap, energy = False)
#     dim_cell = data.Vol**(1/3) # according to Elad
#     stream = np.load(f'data/{folder}/stream_Low{snap}_{step}.npy')
#     theta_arr, indeces_orbit = stream[0], stream[1].astype(int)
#     x_orbit, y_orbit, z_orbit, den_orbit = data.X[indeces_orbit], data.Y[indeces_orbit], data.Z[indeces_orbit], data.Den[indeces_orbit]
#     r_orbit = np.sqrt(x_orbit**2 + y_orbit**2)
#     s_array, idx = find_arclenght(theta_arr, [x_orbit, y_orbit], params = None, choose = 'maxima')
    
#     threshold = np.zeros(len(s_array)-1)
#     dm = np.zeros(len(s_array)-1)
#     for i in range(len(dm)):
#         condition_T, x_Tplane, x0 = transverse_plane(data.X, data.Y, dim_cell, x_orbit, y_orbit, i, coord = True)
#         z_plane, mass_plane= make_slices([data.Z, data.Mass], condition_T)
#         # Restrict to not keep points too far away. Important for theta=0 or you take the stream at apocenter
#         thresh = 3 * Rstar * (r_orbit[i]/Rp)**(1/3)
#         condition_x = np.abs(x_Tplane) < thresh
#         condition_z = np.abs(z_plane) < thresh
#         condition = condition_x & condition_z
#         mass_plane = mass_plane[condition]
#         dm[i] =  np.sum(mass_plane) 
#         threshold[i] = thresh
#     dm_ds = dm / np.diff(s_array)

#     check1 = 'HiRes'
#     path1 = f'/Users/paolamartire/shocks/TDE/{folder}{check1}/{snap}'
#     data1 = make_tree(path1, snap, energy = False)
#     dim_cell1 = data1.Vol**(1/3) # according to Elad
#     tfb = days_since_distruption(f'{path1}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
#     stream1 = np.load(f'data/{folder}/stream_{check1}{snap}_{step}.npy')
#     theta_arr1, indeces_orbit1 = stream1[0], stream1[1].astype(int)
#     x_orbit1, y_orbit1, z_orbit1, den_orbit1 = data1.X[indeces_orbit1], data1.Y[indeces_orbit1], data1.Z[indeces_orbit1], data1.Den[indeces_orbit1]
#     r_orbit1 = np.sqrt(x_orbit1**2 + y_orbit1**2)
#     s_array1, idx1 = find_arclenght(theta_arr1, [x_orbit1, y_orbit1], params = None, choose = 'maxima')
#     dm1 = np.zeros(len(s_array1)-1)
#     threshold1 = np.zeros(len(s_array1)-1)
#     for i in range(len(dm1)):
#         condition_T1, x_Tplane1, x01 = transverse_plane(data1.X, data1.Y, dim_cell1, x_orbit1, y_orbit1, i, coord = True)
#         z_plane1, mass_plane1= make_slices([data1.Z, data1.Mass], condition_T1)
#         thresh1 = 3 * Rstar * (r_orbit1[i]/Rp)**(1/3)
#         condition_x1 = np.abs(x_Tplane1) < thresh1
#         condition_z1 = np.abs(z_plane1) < thresh1
#         condition1 = condition_x1 & condition_z1
#         mass_plane1 = mass_plane1[condition1]
#         dm1[i] =  np.sum(mass_plane1) 
#         threshold1[i] = thresh1
#     dm_ds1 = dm1 / np.diff(s_array1)

#     check2 = 'Res20'
#     snap2 = '169'
#     path2 = f'/Users/paolamartire/shocks/TDE/{folder}{check2}/{snap2}'
#     data2 = make_tree(path2, snap2, energy = False)
#     dim_cell2 = data2.Vol**(1/3) # according to Elad
#     stream2 = np.load(f'data/{folder}/stream_{check2}{snap2}_{step}.npy')
#     theta_arr2, indeces_orbit2 = stream2[0], stream2[1].astype(int)
#     x_orbit2, y_orbit2, z_orbit2, den_orbit2 = data2.X[indeces_orbit2], data2.Y[indeces_orbit2], data2.Z[indeces_orbit2], data2.Den[indeces_orbit2]
#     r_orbit2 = np.sqrt(x_orbit2**2 + y_orbit2**2)
#     s_array2, _ = find_arclenght(theta_arr2, [x_orbit2, y_orbit2], params = None, choose = 'maxima')
#     dm2 = np.zeros(len(s_array2)-1)
#     threshold2 = np.zeros(len(s_array2)-1)
#     for i in range(len(dm2)):
#         condition_T2, x_Tplane2, _ = transverse_plane(data2.X, data2.Y, dim_cell2, x_orbit2, y_orbit2, i, coord = True)
#         z_plane2, mass_plane2= make_slices([data2.Z, data2.Mass], condition_T2)
#         thresh2 = 3 * Rstar * (r_orbit2[i]/Rp)**(1/3)
#         condition_x2 = np.abs(x_Tplane2) < thresh2
#         condition_z2 = np.abs(z_plane2) < thresh2
#         condition2 = condition_x2 & condition_z2
#         mass_plane2 = mass_plane2[condition2]
#         dm2[i] =  np.sum(mass_plane2) 
#         threshold2[i] = thresh2
#     dm_ds2 = dm2/ np.diff(s_array2)
    
#     #%%
#     if plot:
#         ratioLM = 1 - dm_ds1[:230]/dm_ds[:230]
#         ratioMH = 1- dm_ds2[60:230]/dm_ds1[60:230]
#         fig, (ax1,ax2) = plt.subplots(2,1)
#         ax1.plot(theta_arr[:230], dm_ds[:230], c = 'k', label = 'Low')
#         ax1.plot(theta_arr1[:230], dm_ds1[:230], c = 'r', label = 'Middle')
#         ax1.plot(theta_arr2[60:230], dm_ds2[60:230], c = 'b', label = 'High')
#         ax1.legend()
#         ax1.grid()
#         ax1.set_ylabel('dM/ds', fontsize = 16)
#         ax1.set_yscale('log')

#         ax2.plot(theta_arr[:230], ratioLM, c = 'r', label = '1 - Middle/Low')
#         ax2.plot(theta_arr[60:230], ratioMH, c = 'b', label = '1- High/Middle')
#         ax2.grid()
#         ax2.legend()
#         ax2.set_xlabel(r'$\theta$', fontsize = 16)
#         if save:
#             plt.savefig(f'Figs/{folder}/dMds_time{np.round(tfb,1)}.png')
#         plt.show()

#         fig, (ax1,ax2) = plt.subplots(2,1)
#         ax1.plot(theta_arr[:230], dm[:230], c = 'k', label = 'Low')
#         ax1.plot(theta_arr1[:230], dm1[:230], c = 'r', label = 'Middle')
#         ax1.plot(theta_arr2[60:230], dm2[60:230], c = 'b', label = 'High')
#         ax1.set_yscale('log')
#         ax1.set_ylabel('dM', fontsize = 16)
#         ax1.grid()
#         ax2.plot(theta_arr[:230-1], np.diff(s_array[:230]), c = 'k', label = 'Low')
#         ax2.plot(theta_arr1[:230-1], np.diff(s_array1[:230]), c = 'r', label = 'Middle')
#         ax2.plot(theta_arr2[60:230-1], np.diff(s_array2[60:230]), c = 'b', label = 'High')
#         ax2.grid()
#         ax2.set_ylabel('ds', fontsize = 16)
#         ax2.set_xlabel(r'$\theta$', fontsize = 16)
#         plt.suptitle('dM and ds')
#         if save:
#             plt.savefig(f'Figs/{folder}/dMANDds_time{np.round(tfb,1)}.png')
#         plt.show()

#         plt.plot(theta_arr[:230], threshold[:230], c = 'k', label = 'Low')
#         plt.plot(theta_arr1[:230], threshold1[:230], c = 'r', label = 'Middle')
#         plt.plot(theta_arr2[60:230], threshold2[60:230], c = 'b', label = 'High')
#         plt.ylabel('Threshold for mass integration in the TZ plane')
#         plt.xlabel(r'$\theta$', fontsize = 16)
#         plt.title(r'Threshold : $3 (R_{max}/R_p)^{1/3}$')
#         plt.grid()
#         plt.legend()
#         if save:
#             plt.savefig(f'Figs/{folder}/threshold_time{np.round(tfb,1)}.png')
#         plt.show()
# # %%
