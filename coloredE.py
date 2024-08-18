import sys
sys.path.append('/Users/paolamartire/shocks')

from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Utilities.operators import make_tree, radial_caster
import Utilities.sections as sec
import src.orbits as orb
from Utilities.time_extractor import days_since_distruption
from Utilities.selectors import select_snap
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
check = 'Low'

# 
save = True
snaps = select_snap(m, check, mstar, Rstar, beta, n, time = False) #[100,115,164,199,216]

Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

col_ie = []
col_Erad = []
col_orb_en = []
tfb_array = np.zeros(len(snaps))
for i,snap in enumerate(snaps):
    print(snap)
    if alice:
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}Compton'
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    else:
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
        path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
    data = make_tree(path, snap, energy = True)
    tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
    vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
    mass, ie_den, Erad_den = data.Mass, data.IE, data.Erad
    orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
    ie_onmass = ie_den / data.Den
    orb_en_onmass = orb_en / mass
    
    bound_elements = orb_en < 0
    R_bound, mass_bound, vol_bound, ie_onmass_bound, Erad_den_bound, orb_en_onmass_bound = \
            sec.make_slices([Rsph, mass, data.Vol, ie_onmass, Erad_den, orb_en_onmass], bound_elements)
    
    # Cast down to 100 values
    radii = np.logspace(np.log10(R0), np.log10(apo),
                        num=100)  # simulator units
    ie_cast = radial_caster(radii, R_bound, ie_onmass_bound, weights = mass_bound)
    orb_en_cast = radial_caster(radii, R_bound, orb_en_onmass_bound, weights = mass_bound)

    col_ie.append(ie_cast)
    col_orb_en.append(orb_en_cast)

    if not alice:
        Erad_cast = radial_caster(radii, R_bound, Erad_den_bound, weights = vol_bound)
        col_Erad.append(Erad_cast)
    tfb_array[i] = tfb
# tfb_array = np.array(tfb)
#%%
if save:
    if alice:
        prepath = f'/data1/martirep/shocks/shock_capturing/'
    else: 
        prepath = f'/Users/paolamartire/shocks/'
    with open(f'{prepath}/data/{folder}/coloredE.txt', 'a') as file:
        file.write(f'# {folder}_{check} \n#' + ' '.join(map(str, snaps)) + '\n')
        file.write('# t/tfb \n' + ' '.join(map(str, tfb_array)) + '\n')
        file.write('# IE/mass \n') 
        file.write(' '.join(map(str, col_ie)) + '\n')
        file.write('# Orbital energy/mass \n')
        file.write(' '.join(map(str, col_orb_en))+ '\n')
        if not alice:
            file.write('# Erad/Vol \n')
            file.write(' '.join(map(str, col_Erad)) + '\n')
        file.close() 
# %% Plotting
if plot:
    img = plt.pcolormesh(radii, tfb_array, col_ie,
                        cmap='jet')
    cb = plt.colorbar(img)
    cb.set_label('IE/Mass', fontsize=14)
    plt.axvline(Rt, color='black', linestyle='--')
    plt.xscale('log')
    plt.ylabel(r'Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Specific internal energy of time and radius', fontsize=17)
    plt.show()

    img = plt.pcolormesh(radii, tfb_array, col_Erad,
                        cmap='jet')
    cb = plt.colorbar(img)
    cb.set_label('Erad/Vol', fontsize=14)
    plt.axvline(Rt, color='black', linestyle='--')
    plt.xscale('log')
    plt.ylabel(r'Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Radiation energy density of time and radius', fontsize=17)
    plt.show()

    img = plt.pcolormesh(radii, tfb_array, col_orb_en,
                        cmap='jet')
    cb = plt.colorbar(img)
    cb.set_label(r'E$_{orb}$/Vol', fontsize=14)
    plt.axvline(Rt, color='black', linestyle='--')
    plt.xscale('log')
    plt.ylabel(r'Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Specific orbital energy of time and radius', fontsize=17)
    plt.show()
# %%
