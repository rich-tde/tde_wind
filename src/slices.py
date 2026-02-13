""" Make/plot slices at a fixed coordinate (choose x,y,z).
If alice: make the section (with density cut 1e-19) and save it: X, Y, Z, vol, den, mass, Temp, ie_den, Rad_den, VX, VY, VZ, Diss_den, Press
If not alice: load the section and plot the slice with color = chosen quantity.
"""
from fileinput import filename
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

import numpy as np
from Utilities.selectors_for_snap import select_snap, select_prefix
import src.orbits as orb
import Utilities.prelude as prel
import Utilities.sections as sec
from Utilities.operators import make_tree, to_spherical_components

#
##
# CHOICES
##
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' # 'LowRes' or 'HiRes'
coord_to_cut = 'y' # 'x', 'y', 'z'
cut_chosen = 0
single_plot = False
print(f'cut at {coord_to_cut} = {cut_chosen}', flush=True)

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
apo = things['apo']

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
if cut_chosen == Rp:
    cut_name = 'Rp'
else:
    cut_name =  cut_chosen

if alice:
    # get ready to slice and save
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
    with open(f'{abspath}/data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}_time.txt','w') as file:
        file.write('#Snap \n') 
        file.write(' '.join(map(str, snaps)) + '\n')
        file.write('# Time \n')
        file.write(' '.join(map(str, tfb)) + '\n')
        file.close()

    for idx, snap in enumerate(snaps):
        if snap > 50:
                continue
        print(snap, flush=True)
        path = select_prefix(m, check, mstar, Rstar, beta, n, compton)
        path = f'{path}/snap_{snap}'

        data = make_tree(path, snap, energy = True)
        X, Y, Z, vol, den, mass, Temp, ie_den, Rad_den, VX, VY, VZ, Diss_den, Press = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Temp, data.IE, data.Rad, data.VX, data.VY, data.VZ, data.Diss, data.Press
        Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
        dim_cell = vol**(1/3)
        if coord_to_cut == 'x':
            cutcoord = X
        elif coord_to_cut == 'y':
            cutcoord = Y
        elif coord_to_cut == 'z':
            cutcoord = Z

        # make slices 
        density_cut = den > 1e-19
        coordinate_cut = np.abs(cutcoord-cut_chosen) < dim_cell
        cut = np.logical_and(density_cut, coordinate_cut)
        x_cut, y_cut, z_cut, dim_cut, den_cut, mass_cut, temp_cut, ie_den_cut, Rad_den_cut, VX_cut, VY_cut, VZ_cut, Diss_den_cut, Press_cut = \
            sec.make_slices([X, Y, Z, dim_cell, den, mass, Temp, ie_den, Rad_den, VX, VY, VZ, Diss_den, Press], cut)

        np.savez(f'{abspath}/data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}slice_{snap}.npz',
                    x=x_cut,
                    y=y_cut,
                    z=z_cut,
                    dim=dim_cut,
                    density=den_cut,
                    mass=mass_cut,
                    temperature=temp_cut,
                    ie_density=ie_den_cut,
                    rad_density=Rad_den_cut,
                    vx=VX_cut,
                    vy=VY_cut,
                    vz=VZ_cut,
                    diss_density=Diss_den_cut,
                    pressure=Press_cut,
                )

else:
    time = np.loadtxt(f'{abspath}data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}_time.txt')
    snaps = time[0]
    snaps = np.array([int(snap) for snap in snaps])
    tfb = time[1]

    if single_plot:
        for idx, snap in enumerate(snaps):
            if snap > 50:
                    continue
            fig, ax = plt.subplots(1,1, figsize = (14,8))
            # load the data
            data = np.load(f'{abspath}data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}slice_{snap}.npz')
            x = data["x"]
            y = data["y"]
            z = data["z"]
            dim = data["dim"]
            den = data["density"]
            mass = data["mass"]
            temp = data["temperature"]
            ie_den = data["ie_density"]
            Rad_den = data["rad_density"]
            VX = data["vx"]
            VY = data["vy"]
            VZ = data["vz"]
            Diss_den = data["diss_density"]
            Press = data["pressure"]

            img = ax.scatter(x/Rt, y/Rt, c = den*prel.den_converter, cmap = 'plasma', s= .1, \
                        norm = colors.LogNorm(vmin = 5e-9, vmax = 1e-4))
            cb = plt.colorbar(img)
            cb.set_label(r'Density [g/cm$^3$]')
            cb.ax.tick_params(which='major', length=7, width=1.2)
            cb.ax.tick_params(which='minor', length=4, width=1)
            ax.set_ylabel(r'$ y [r_{\rm t}]$')
            ax.set_xlabel(r'$x [r_{\rm t}]$')
            ax.set_xlim(-50, 50)#(-340,25)
            ax.set_ylim(-50, 50)#(-70,70)

            plt.suptitle(f't = {np.round(tfb[idx], 2)}' + r'$t_{\rm fb}$', fontsize = 20)
            plt.tight_layout()
            plt.savefig(f'{abspath}Figs/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}slice_{snap}.png')
            plt.close()
        
    else:
        from matplotlib import gridspec
        idx_wanted = [np.argmin(np.abs(snaps - 76)), 
                      np.argmin(np.abs(snaps - 109)), 
                      np.argmin(np.abs(snaps - 151))]
        
        fig = plt.figure(figsize=(12*len(idx_wanted), 12))
        gs = gridspec.GridSpec(2, len(idx_wanted), width_ratios=[1]*len(idx_wanted), hspace=0.3,  height_ratios=[1, 0.05], wspace = 0.2)
        
        for i, idx in enumerate(idx_wanted):
            snap = snaps[idx]
            time = tfb[idx]
            # load the data
            data = np.load(f'{abspath}data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}slice_{snap}.npz')
            x = data["x"]
            y = data["y"]
            z = data["z"]
            dim = data["dim"]
            den = data["density"]
            mass = data["mass"]
            ie_den = data["ie_density"]
            Rad_den = data["rad_density"]
            VX = data["vx"]
            VY = data["vy"]
            VZ = data["vz"]
            Press = data["pressure"]

            cut, bern, V_r = orb.pick_wind(x, y, z, VX, VY, VZ, den, mass, Press, ie_den, Rad_den, params)
            x, y, z, V_r, den, mass, Press, ie_den, Rad_den, dim, bern = \
                sec.make_slices([x, y, z, V_r, den, mass, Press, ie_den, Rad_den, dim, bern], cut)      
            wind_cond = np.logical_and(V_r >= 0, bern > 0)

            if coord_to_cut == 'y':
                xlabel = r'$x (r_{\rm t})$'
                y_to_plot = z
                ylabel = r'$z (r_{\rm t})$'
            elif coord_to_cut == 'z':
                xlabel = r'$x (r_{\rm t})$'
                y_to_plot = y
                ylabel = r'$y (r_{\rm t})$'

            ax = fig.add_subplot(gs[0, i])
            img = ax.scatter(x[wind_cond]/Rt, y_to_plot[wind_cond]/Rt, c = den[wind_cond]*prel.den_converter, cmap = 'rainbow', s= .1,
                             norm = colors.LogNorm(vmin = 1e-15, vmax = 1e-8))
            ax.set_xlabel(xlabel, fontsize = 35)
            if i == 0:
                ax.set_ylabel(ylabel, fontsize = 35)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_title(f't = {np.round(time, 2)}' + r'$t_{\rm fb}$', fontsize = 25)
            ax.tick_params(axis='both', which='major', width=1.2, length=9)

        cbar_ax = fig.add_subplot(gs[1, 0:len(idx_wanted)])  # Colorbar subplot below the first two
        cb = fig.colorbar(img, orientation='horizontal', cax=cbar_ax)
        cb.set_label(r'$\rho$ (g/cm$^3)$', fontsize = 40)
        cb.ax.tick_params(labelsize=30)
        cb.ax.tick_params(which='major', length=9, width=1.4)
        cb.ax.tick_params(which='minor', length=6, width=1.2)

        plt.tight_layout()
        plt.savefig(f'{abspath}Figs/{folder}/Wind{coord_to_cut}{cut_name}slices_wind.png')