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
check = 'Low'
compton = 'Compton'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
# 
save = True
cutoff = 'cutden' # or '' (no cut) or 'cutden' (cut in density) 'bound' (cut in orb_den) or 'cutdenbound' (both)
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) #[100,115,164,199,216]

Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

col_ie = []
col_Rad = []
col_orb_en = []
radii = np.logspace(np.log10(R0), np.log10(1.5*apo),
                    num=500)  # simulator units
for i,snap in enumerate(snaps):
    print(snap)
    if alice:
        if check == 'Low':
            check = ''
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}/snap_{snap}'
    else:
        path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
    data = make_tree(path, snap, energy = True)
    Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
    vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
    mass, vol, ie_den, Rad_den = data.Mass, data.Vol, data.IE, data.Rad
    orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
    dim_cell = (3/(4*np.pi) * vol)**(1/3)
    ie_onmass = ie_den / data.Den
    orb_en_onmass = orb_en / mass

    # cutoff just for IE and Orbital energy
    if cutoff == 'cutden' or cutoff == 'bound' or cutoff == 'cutdenbound':
        if cutoff == 'cutden':
            print('Cutting off low density elements (just for IE and Orbital energy)')
            finalcut = data.Den > 1e-9 # throw fluff
        elif cutoff == 'bound':
            print('Cutting off unbound elements (just for IE and Orbital energy)')
            finalcut = orb_en < 0 # bound elements
        elif cutoff == 'cutdenbound':
            print('Cutting off low density elements and unbound elements (just for IE and Orbital energy)')
            cutden = data.Den > 1e-9 # throw fluff
            cutbound = orb_en < 0 # bound elements
            finalcut = cutden & cutbound

        Rsph_cut, mass_cut, ie_onmass_cut, orb_en_onmass_cut = \
                sec.make_slices([Rsph, mass, ie_onmass, orb_en_onmass], finalcut)
    
    # Cast down to 100 values
    ie_cast = radial_caster(radii, Rsph_cut, ie_onmass_cut, weights = mass_cut)
    orb_en_cast = radial_caster(radii, Rsph_cut, orb_en_onmass_cut, weights = mass_cut)
    Rad_cast = radial_caster(radii, Rsph, Rad_den, weights = vol)

    col_ie.append(ie_cast)
    col_orb_en.append(orb_en_cast)
    col_Rad.append(Rad_cast)

#%%
if save:
    if alice:
        if check == '':
            check = 'Low'
        prepath = f'/data1/martirep/shocks/shock_capturing/'
    else: 
        prepath = f'/Users/paolamartire/shocks/'
    np.save(f'{prepath}/data/{folder}/{cutoff}coloredE_{check}.npy', [col_ie, col_orb_en, col_Rad])
    with open(f'{prepath}/data/{folder}/{cutoff}coloredE_{check}_days.txt', 'a') as file:
        file.write(f'# {folder}_{check} \n' + ' '.join(map(str, snaps)) + '\n')
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        file.close()
    np.save(f'{prepath}/data/{folder}/{cutoff}coloredE_{check}_radii.npy', radii)
# %% Plotting
if plot:
    img = plt.pcolormesh(radii, tfb_array, col_ie,
                        cmap='cet_rainbow4')
    cb = plt.colorbar(img)
    cb.set_label('IE/Mass', fontsize=14)
    plt.axvline(Rt, color='black', linestyle='--')
    plt.xscale('log')
    plt.ylabel(r'Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Specific internal energy of time and radius', fontsize=17)
    plt.show()

    img = plt.pcolormesh(radii, tfb_array, col_Rad,
                        cmap='cet_rainbow4')
    cb = plt.colorbar(img)
    cb.set_label('Rad/Vol', fontsize=14)
    plt.axvline(Rt, color='black', linestyle='--')
    plt.xscale('log')
    plt.ylabel(r'Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Radiation energy density of time and radius', fontsize=17)
    plt.show()

    img = plt.pcolormesh(radii, tfb_array, col_orb_en,
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
