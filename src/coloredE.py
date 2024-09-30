abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Utilities.operators import make_tree, single_branch, to_cylindric
import Utilities.sections as sec
import src.orbits as orb
from Utilities.selectors_for_snap import select_snap
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
step = 'DoubleRad'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{step}'

check = 'Low'
xaxis = 'radii' # radii or angles
save = True

snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) #[100,115,164,199,216]
Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

col_ie = []
col_orb_en = []
col_Rad = []
col_Rad_samecut = []

if xaxis == 'angles':
    from Utilities.operators import Ryan_sampler
    radii = np.arange(-np.pi, np.pi, 0.02)
    radii = Ryan_sampler(radii)
elif xaxis == 'radii':
    radii = np.logspace(np.log10(R0), np.log10(1.5*apo),
                    num=200)  # simulator units
    
for i,snap in enumerate(snaps):
    if alice:
        if check == 'Low':
            check = ''
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}/snap_{snap}'
    else:
        path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
    print(path, snap)
    data = make_tree(path, snap, energy = True)
    Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
    vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
    mass, vol, ie_den, Rad_den = data.Mass, data.Vol, data.IE, data.Rad
    theta, _ = to_cylindric(data.X, data.Y)
    orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
    Rad = Rad_den * vol
    ie = ie_den * vol
    dim_cell = (3/(4*np.pi) * vol)**(1/3)
    ie_onmass = ie_den / data.Den
    orb_en_onmass = orb_en / mass

    # graph = {}
    # graph[0] = {"fluff": neighbours_list}

    # throw fluff
    cut = data.Den > 1e-9 
    cutsmall = data.Den > 1e-19
    Rsph_cut, theta_cut, mass_cut, ie_cut, ie_onmass_cut, orb_en_cut, orb_en_onmass_cut, Rad_cut, Rad_den_cut, vol_cut = \
            sec.make_slices([Rsph, theta, mass, ie, ie_onmass, orb_en, orb_en_onmass, Rad, Rad_den, vol], cut)
    Rsph_cutsmall, theta_cutsmall, Rad_cutsmall, Rad_den_cutsmall, vol_cutsmall = \
        sec.make_slices([Rsph, theta, Rad, Rad_den, vol], cutsmall)

    if xaxis == 'angles':
        tocast_cut = theta_cut
        tocast_cutsmall = theta_cutsmall
    elif xaxis == 'radii':
        tocast_cut = Rsph_cut
        tocast_cutsmall = Rsph_cutsmall
        
    # Cast down 
    ie_cast = single_branch(radii, xaxis, tocast_cut, ie_onmass_cut, weights = mass_cut)
    orb_en_cast = single_branch(radii, xaxis, tocast_cut, orb_en_onmass_cut, weights = mass_cut)
    Rad_castsmall = single_branch(radii, xaxis, tocast_cutsmall, Rad_den_cutsmall, weights = vol_cutsmall)
    Rad_cast = single_branch(radii, xaxis, tocast_cut, Rad_den_cut, weights = vol_cut)
    # ie_cast = single_branch(radii, xaxis, tocast_cut, ie_onmass_cut, weights = ie_cut)
    # orb_en_cast = single_branch(radii, xaxis, tocast_cut, orb_en_onmass_cut, weights = orb_en_cut)
    # Rad_castsmall = single_branch(radii, xaxis, tocast_cutsmall, Rad_den_cutsmall, weights = Rad_cutsmall)
    # Rad_cast = single_branch(radii, xaxis, tocast_cut, Rad_den_cut, weights = Rad_cut)

    col_ie.append(ie_cast)
    col_orb_en.append(orb_en_cast)
    # just cut the very low fluff
    col_Rad.append(Rad_castsmall)
    # same cut as the other energies
    col_Rad_samecut.append(Rad_cast)

#%%
if save:
    if alice:
        if check == '':
            check = 'Low'
        prepath = f'/data1/martirep/shocks/shock_capturing'
    else: 
        prepath = f'/Users/paolamartire/shocks'
    np.save(f'{prepath}/data/{folder}/coloredE_{check}{step}_{xaxis}.npy', [col_ie, col_orb_en, col_Rad, col_Rad_samecut])
    with open(f'{prepath}/data/{folder}/coloredE_{check}_days.txt', 'a') as file:
        file.write(f'# {folder}_{check} \n' + ' '.join(map(str, snaps)) + '\n')
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        file.close()
    np.save(f'{prepath}/data/{folder}/{xaxis}En_{check}.npy', radii)
# %% Plotting
if plot:
    img = plt.pcolormesh(radii, tfb, col_ie,
                        cmap='cet_rainbow4')
    cb = plt.colorbar(img)
    cb.set_label('IE/Mass', fontsize=14)
    plt.axvline(Rt, color='black', linestyle='--')
    plt.xscale('log')
    plt.ylabel(r'Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Specific internal energy of time and radius', fontsize=17)
    plt.show()

    img = plt.pcolormesh(radii, tfb, col_Rad,
                        cmap='cet_rainbow4')
    cb = plt.colorbar(img)
    cb.set_label('Rad/Vol', fontsize=14)
    plt.axvline(Rt, color='black', linestyle='--')
    plt.xscale('log')
    plt.ylabel(r'Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Radiation energy density of time and radius', fontsize=17)
    plt.show()

    img = plt.pcolormesh(radii, tfb, col_orb_en,
                        cmap='cet_rainbow4')
    cb = plt.colorbar(img)
    cb.set_label(r'E$_{orb}$/Vol', fontsize=14)
    plt.axvline(Rt, color='black', linestyle = '--')
    plt.xscale('log')
    plt.ylabel(r'Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Specific orbital energy of time and radius', fontsize=17)
    plt.show()
# %%
