""" Mass distribution from orbital energy.
If alice: compute the dM/dE for all snaps.
If local: plot dM/dE to compare different resolutions (at different times) or to make a movie of a single res.
There's some old code to compute dM/ds where ds = line element of the stream."""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    abspath = '/Users/paolamartire/shocks/'
    save = False

import numpy as np
import matplotlib.pyplot as plt
from Utilities.operators import make_tree
import Utilities.sections as sec
import src.orbits as orb
from Utilities.selectors_for_snap import select_snap
import Utilities.prelude as prel

#
## PARAMETERS STAR AND BH
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
norm = things['E_mb']
Ecirc = -prel.G*Mbh/(4*Rp)
t_fb_days = things['t_fb_days'] 
t_fb_days_cgs = t_fb_days * 24 * 3600 # in seconds

#
## MAIN
#

# Choose what to do
test_bins = False
compare_res = True
movie = False 
dMdecc = False

if alice:
    check = 'HiResNewAMR'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    print(f'Check: {check}', flush=True)

    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    # bins1 = np.arange(-6.5, -1.5, .1) 
    # bins2 = np.arange(-1.5, 0, .01)
    # bins3 = np.arange(0, 2, .1)
    bins = np.arange(-6.5, 2, .01) #np.concatenate((bins1, bins2, bins3))
    # save snaps, tfb and energy bins
    with open(f'{abspath}/data/{folder}/wind/dMdE_{check}_days.txt','w') as filedays:
        filedays.write(f'# {folder}_{check} \n# Snaps \n' + ' '.join(map(str, snaps)) + '\n')
        filedays.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        filedays.close()
    with open(f'{abspath}/data/{folder}/wind/dMdE_{check}_bins.txt','w') as file:
        file.write(f'# Energy bins normalised (by DeltaE = {norm}) \n')
        file.write((' '.join(map(str, bins)) + '\n'))
        file.close()
    for snap in snaps:
        print(f'Snap: {snap}', flush=True)
        # Load data
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        data = make_tree(path, snap, energy = False)
        # Compute the orbital energy
        X, Y, Z, VX, VY, VZ, Den, mass, Vol = \
            data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Den, data.Mass, data.Vol
        cut = Den > 1e-19
        X, Y, Z, VX, VY, VZ, mass = \
            sec.make_slices([X, Y, Z, VX, VY, VZ, mass], cut)
        R = np.sqrt(X**2 + Y**2 + Z**2)
        V = np.sqrt(VX**2 + VY**2 + VZ**2)
        orbital_enegy = orb.orbital_energy(R, V, mass, params, prel.G)
        specific_orbital_energy = orbital_enegy / mass

        # (Specific) energy bins 
        specOE_norm = specific_orbital_energy/norm 
        mass_binned, bins_edges = np.histogram(specOE_norm, bins = bins, weights=mass) # sum the mass in each bin (bins done on specOE_norm)
        dm_dE = mass_binned / (np.diff(bins_edges)*norm)

        with open(f'{abspath}/data/{folder}/wind/dMdE_{check}.txt','a') as file:
            file.write(f'# dM/dE [code units] snap {snap} \n')
            file.write((' '.join(map(str, dm_dE)) + '\n'))
            file.close()

if not alice:
    if test_bins:
        snap = 348
        check = ''
        spacebins = 0.01
        bins = np.arange(-6.5, 2, spacebins) #np.linspace(-5,5,1000)
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        path = f'{abspath}/TDE/{folder}/{snap}'
        data = make_tree(path, snap, energy = False)
        # Compute the orbital energy
        dim_cell = 0.5*data.Vol**(1/3) 
        mass = data.Mass
        R = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
        V = np.sqrt(data.VX**2 + data.VY**2 + data.VZ**2)
        orbital_enegy = orb.orbital_energy(R, V, mass, params, prel.G)
        specific_orbital_energy = orbital_enegy / mass
        specOE_norm = specific_orbital_energy/norm
        # Cutoff for low density
        cut = data.Den > 1e-19
        mass, specOE_norm = mass[cut], specOE_norm[cut]
        # count how many particles in each bins
        counts = np.zeros(len(bins)-1)
        for i, bin in enumerate(bins[:-1]):
            counts[i] = len(mass[np.logical_and(specOE_norm>bin, specOE_norm<bins[i+1])])
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
        ax1.scatter(bins[:-1], counts, c = 'k', s = 2, label = 'Counts')
        ax2.plot(bins[:-1], counts, c = 'k', label = 'Counts')
        for ax in (ax1, ax2):
            ax.set_yscale('log')
            ax.set_xlabel(r'$E/\Delta E$', fontsize = 16)
            ax.legend(fontsize = 16)
        ax1.set_yscale('log')
        plt.suptitle(f'Spacing between bins: {spacebins}', fontsize = 16)
        plt.savefig(f'{abspath}Figs/Test/spacing{spacebins}dMdE.png')

    if compare_res:
        from Utilities.operators import find_ratio
        commonfolder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
        datadays = np.loadtxt(f'{abspath}data/{commonfolder}NewAMR/wind/dMdE_NewAMR_days.txt')
        snaps, tfb= datadays[0], datadays[1]
        bins = np.loadtxt(f'{abspath}data/{commonfolder}NewAMR/wind/dMdE_NewAMR_bins.txt')
        mid_points = (bins[:-1]+bins[1:])/2
        data = np.loadtxt(f'{abspath}data/{commonfolder}NewAMR/wind/dMdE_NewAMR.txt')
        
        datadaysH = np.loadtxt(f'{abspath}data/{commonfolder}HiResNewAMR/wind/dMdE_HiResNewAMR_days.txt')
        snapsH, tfbH = datadaysH[0], datadaysH[1]
        dataH = np.loadtxt(f'{abspath}data/{commonfolder}HiResNewAMR/wind/dMdE_HiResNewAMR.txt')
        
        datadaysL = np.loadtxt(f'{abspath}data/{commonfolder}LowResNewAMR/wind/dMdE_LowResNewAMR_days.txt')
        snapsL, tfbL = datadaysL[0], datadaysL[1]
        dataL = np.loadtxt(f'{abspath}data/{commonfolder}LowResNewAMR/wind/dMdE_LowResNewAMR.txt')

        final_time = 1 #np.max(tfb)
        final_time_cgs = final_time * t_fb_days_cgs #converted to seconds
    
        tsol = final_time_cgs / prel.tsol_cgs # convert to code units
        # Find the energy of the element at time t
        energy = orb.keplerian_energy(Mbh, prel.G, tsol)
        print(f'Energy at time {final_time} tfb: {np.round(energy/norm, 2)} deltaE')
        idx_snap = np.argmin(np.abs(tfb - final_time))
        idx_snapH = np.argmin(np.abs(tfbH - final_time))
        idx_snapL = np.argmin(np.abs(tfbL - final_time))
        time, timeH, timeL = \
            tfb[idx_snap], tfbH[idx_snapH], tfbL[idx_snapL]
        print(time, timeH, timeL)
        data_fin, dataH_fin, dataL_fin = \
            data[idx_snap], dataH[idx_snapH], dataL[idx_snapL]
        ratio_L = np.zeros_like(data_fin) 
        ratio_H = np.zeros_like(data_fin)
        for j in range(len(data_fin)):
            ratio_L[j] = find_ratio(data_fin[j], dataL_fin[j])
            ratio_H[j] = find_ratio(data_fin[j], dataH_fin[j])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.plot(mid_points, data[0], c = 'k', alpha = 0.5)#, label = r't = 0')
        ax1.plot(mid_points, dataL_fin, c = 'C1', label = f'Low')            
        ax1.plot(mid_points, data_fin, c = 'yellowgreen', label = f'Middle')
        ax1.plot(mid_points, dataH_fin, c = 'darkviolet', label = f'High')

        ax2.plot(mid_points, ratio_L, c = 'C1')
        ax2.plot(mid_points, ratio_L, c = 'yellowgreen', linestyle = (0, (5, 10)))
        ax2.plot(mid_points, ratio_H, c = 'darkviolet')
        ax2.plot(mid_points, ratio_H, c = 'yellowgreen', linestyle = (0, (5, 10)))

        ax1.legend(fontsize = 16)
        ax1.set_yscale('log')
        ax1.set_ylabel(r'dM/d$\varepsilon$')
        ax1.set_ylim(2e-6, 2e-2)
        ax2.set_xlabel(r'$\varepsilon/\Delta\varepsilon$')
        ax2.set_ylabel(r'$\kappa$')
        # original_ticksy = ax2.get_yticks()
        # midpointsy = (original_ticksy[:-1] + original_ticksy[1:]) / 2
        # new_ticksy = np.sort(np.concatenate((original_ticksy, midpointsy)))
        # ax2.set_yticks(new_ticksy)
        ax2.set_ylim(0.9, 3)
        for ax in (ax1, ax2):
            ax.set_xlim(-2.5,2)
            ax.tick_params(axis='both', which='major', width = 1.2, length = 9, color = 'k')
            ax.tick_params(axis='y', which='minor', width = 1, length = 5, color = 'k')
        # put the legend outside the plo
        # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 14)
        # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 14)
        # plt.suptitle(f't/tfb = {final_time}', fontsize = 16)
        plt.tight_layout()
        plt.savefig(f'{abspath}Figs/paper/dMdE.pdf', bbox_inches='tight')
        plt.show()

    if movie:
        import subprocess
        check = 'NewAMR'
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        datadays = np.loadtxt(f'{abspath}data/{folder}/wind/dMdE_{check}_days.txt')
        snaps, tfb= datadays[0], datadays[1]
        bins = np.loadtxt(f'{abspath}data/{folder}/wind/dMdE_{check}_bins.txt')
        data = np.loadtxt(f'{abspath}data/{folder}/wind/dMdE_{check}.txt')
        mid_points = (bins[:-1]+bins[1:])/2
        plt.figure()
        for i in range(len((data))):
            if i!=0 and i != np.argmin(np.abs(tfb - 0.2)) and i != np.argmin(np.abs(tfb - 0.3)):
                continue
            snap = snaps[i]
            plt.plot(mid_points, data[i], alpha = 0.8, label = f't/tfb = {np.round(tfb[i],2)}')
            plt.xlabel(r'$\log_{10}E/\Delta E$', fontsize = 16)
            plt.ylabel('dM/dE', fontsize = 16)
            plt.yscale('log')
            plt.xlim(-2.5,2.5)
            plt.ylim(9e-7, 1.5e-2)
        plt.legend(fontsize = 14)
        plt.tight_layout()
        plt.show()
            # plt.text(-1.5, 1e-2, f't/tfb = {np.round(tfb[i],2)}', fontsize = 14)

            # if save:
            #     plt.savefig(f'{abspath}Figs/{folder}/dM/snap{int(snap)}.png')
            # plt.close()
        # Make the movie
        # path = f'{abspath}Figs/{folder}/dM/snap'
        # output_path = f'{abspath}Figs/{folder}/dM/moviedMdE_.mp4'

        # start = 100
        # slow_down_factor = 2  # Increase this value to make the video slower

        # ffmpeg_command = (
        #     f'ffmpeg -y -start_number {start} -i {path}%d.png -vf "setpts={slow_down_factor}*PTS" '
        #     f'-c:v libx264 -pix_fmt yuv420p {output_path}'
        #     )

        # subprocess.run(ffmpeg_command, shell=True)
            
    if dMdecc:
        checks = ['']#['LowRes','', 'HiRes']
        colors_check = ['yellowgreen']#['C1', 'yellowgreen', 'darkviolet']
        labels = ['Low', 'Fid', 'High']
        snap = 318
        bins = np.arange(-6.5, 2, .1) 
        bins_ecc = np.arange(0, 1, .01) #np.linspace(-5,5,1000) 
        mid_points = (bins[:-1]+bins[1:])/2
        mid_points_ecc = (bins_ecc[:-1]+bins_ecc[1:])/2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

        for i, check in enumerate(checks):
            folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
            # save snaps, tfb and energy bins
            with open(f'{abspath}/data/{folder}/dMdEdecc_{check}.txt','w') as file:
                file.write(f'# Energy bins normalised (by DeltaE = {norm}) \n')
                file.write((' '.join(map(str, bins)) + '\n'))
                file.close()
            
            # Load data
            path = f'{abspath}/TDE/{folder}/{snap}'
            data = make_tree(path, snap, energy = False)
            # Compute the orbital energy
            cut = data.Den > 1e-19
            X, Y, Z, VX, VY, VZ, mass = \
                sec.make_slices([data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Mass], cut)
            R_vec = np.transpose(np.array([X, Y, Z]))
            vel_vec = np.transpose(np.array([VX, VY, VZ]))
            R = np.linalg.norm(R_vec, axis = 1)
            V = np.linalg.norm(vel_vec, axis=1)
            orbital_enegy = orb.orbital_energy(R, V, mass, params, prel.G)
            specific_orbital_energy = orbital_enegy / mass
            bound = specific_orbital_energy<0
            R_vec_bound, vel_vec_bound, specOE_bound, mass_bound  = \
                R_vec[bound], vel_vec[bound], specific_orbital_energy[bound], mass[bound]
            ecc2 = eccentricity_squared(R_vec_bound, vel_vec_bound, specOE_bound, Mbh, prel.G)
            ecc_bound = np.sqrt(ecc2)
            nans = np.isnan(ecc_bound)
            mass_bound, ecc_bound = mass_bound[~nans], ecc_bound[~nans]

            # (Specific) energy bins 
            specOE_norm = specific_orbital_energy/norm 
            mass_binned, bins_edges = np.histogram(specOE_norm, bins = bins, weights=mass) # sum the mass in each bin (bins done on specOE_norm)
            dm_dE = mass_binned / (np.diff(bins_edges)*norm)

            # Eccentricity bins
            mass_binned_ecc, bins_edges_ecc = np.histogram(ecc_bound, bins = bins_ecc, weights=mass_bound) # sum the mass in each bin (bins done on specOE_norm)
            dm_dE_ecc = mass_binned_ecc / (np.diff(bins_edges_ecc))

            with open(f'{abspath}/data/{folder}/dMdEdecc_{check}.txt','a') as file:
                file.write(f'# dM/dE snap {snap} \n')
                file.write((' '.join(map(str, dm_dE)) + '\n'))
                file.write(f'# dM/de bound material snap {snap} \n')
                file.write((' '.join(map(str, dm_dE_ecc)) + '\n'))
                file.close()
            
            ax1.plot(mid_points, dm_dE, c = colors_check[i], label = labels[i], linewidth = 3-i)
            ax2.plot(mid_points_ecc, dm_dE_ecc, c = colors_check[i], label = labels[i], linewidth = 3-i)
            ax2.axvline(np.mean(ecc_bound), c = 'k', linestyle = '--')
            ax2.text(np.mean(ecc_bound)+0.05, 2e-4, r'$<e>$ ' + f'= {np.round(np.mean(ecc_bound),2)}', fontsize = 14, rotation = 90)
            
        ax1.set_ylabel('dM/dE', fontsize = 16)
        ax1.set_xlabel(r'$E/\Delta E$', fontsize = 16)
        ax1.set_ylim(2e-6, 2e-2)
        ax1.set_xlim(-2,2)
        ax2.set_ylabel('dM/de bound material', fontsize = 16)
        ax2.set_xlabel(r'$e$', fontsize = 16)
        ax2.set_ylim(1e-4, 10)
        for ax in (ax1, ax2):
            ax.set_yscale('log')
        plt.suptitle(f'dM/dE and dM/de for at snap {snap}', fontsize = 16)
        plt.tight_layout()
        # ax2.legend(loc='upper left', fontsize = 14)
        plt.savefig(f'{abspath}Figs/multiple/dEdMdecc{snap}.png')

    #%%###########
    # if do_dMds:
    #     from src.orbits import find_arclenght
    #     from Utilities.sections import transverse_plane, make_slices

    #     step = 0.02
    #     data = make_tree(path, snap, energy = False)
    #     dim_cell = data.Vol**(1/3) # according to Elad
    #     stream = np.load(f'data/{folder}/stream_Low{snap}_.npy')
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
    #     stream1 = np.load(f'data/{folder}/stream_{check1}{snap}_.npy')
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
    #     stream2 = np.load(f'data/{folder}/stream_{check2}{snap2}_.npy')
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
 