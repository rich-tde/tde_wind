abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
from Utilities.selectors_for_snap import select_snap
from Utilities.operators import make_tree, single_branch, to_cylindric
import Utilities.sections as sec
import src.orbits as orb

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
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

checks = ['Low','HiRes']
step = ''
xaxis = 'radii' # radii or angles
weight = 'weightDen' # weightDen or weightE or '' if you have weight for vol/mass
save = True

Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

col_ie = []
col_orb_en = []
col_Rad = []
col_Rad_cutsmall = []
col_ie_cutsmall = []
col_orb_en_cutsmall = []

if xaxis == 'angles':
    from Utilities.operators import Ryan_sampler
    radii = np.arange(-np.pi, np.pi, 0.02)
    radii = Ryan_sampler(radii)
elif xaxis == 'radii':
    radii = np.logspace(np.log10(R0), np.log10(1.5*apo),
                    num=200)  # simulator units

for check in checks:    
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, step, time = True) #[100,115,164,199,216]
    for i,snap in enumerate(snaps):
        print(snap)
        if alice:
            if check == 'Low':
                check = ''
            path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}{step}/snap_{snap}'
        else:
            path = f'/Users/paolamartire/shocks/TDE/{folder}{check}{step}/{snap}'
        data = make_tree(path, snap, energy = True)
        Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
        vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
        mass, vol, den, ie_den, Rad_den = data.Mass, data.Vol, data.Den, data.IE, data.Rad
        theta, _ = to_cylindric(data.X, data.Y)
        orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
        Rad = Rad_den * vol
        ie = ie_den * vol
        dim_cell = (3/(4*np.pi) * vol)**(1/3)
        ie_onmass = ie_den / data.Den
        orb_en_onmass = orb_en / mass

        # throw fluff
        cutsmall = den > 1e-19
        cut = den > 1e-9 
        Rsph_cut, theta_cut, mass_cut, den_cut, ie_cut, ie_onmass_cut, orb_en_cut, orb_en_onmass_cut, Rad_cut, Rad_den_cut, vol_cut = \
                sec.make_slices([Rsph, theta, mass, den, ie, ie_onmass, orb_en, orb_en_onmass, Rad, Rad_den, vol], cut)
        Rsph_cutsmall, theta_cutsmall, den_cutsmall, ie_cutsmall, orb_en_cutsmall, Rad_cutsmall, Rad_den_cutsmall, vol_cutsmall = \
            sec.make_slices([Rsph, theta, den, ie, orb_en, Rad, Rad_den, vol], cutsmall)

        if xaxis == 'angles':
            tocast_cut = theta_cut
            tocast_cutsmall = theta_cutsmall
        elif xaxis == 'radii':
            tocast_cut = Rsph_cut
            tocast_cutsmall = Rsph_cutsmall
            
        # Cast down 
        if weight == '':
            ie_cast = single_branch(radii, xaxis, tocast_cut, ie_onmass_cut, weights = mass_cut)
            orb_en_cast = single_branch(radii, xaxis, tocast_cut, orb_en_onmass_cut, weights = mass_cut)
            Rad_cast = single_branch(radii, xaxis, tocast_cut, Rad_den_cut, weights = vol_cut)
            Rad_castsmall = single_branch(radii, xaxis, tocast_cutsmall, Rad_den_cutsmall, weights = vol_cutsmall)
        elif weight == 'weightE':
            ie_cast = single_branch(radii, xaxis, tocast_cut, ie_onmass_cut, weights = ie_cut)
            orb_en_cast = single_branch(radii, xaxis, tocast_cut, orb_en_onmass_cut, weights = orb_en_cut)
            Rad_cast = single_branch(radii, xaxis, tocast_cut, Rad_den_cut, weights = Rad_cut)
            Rad_castsmall = single_branch(radii, xaxis, tocast_cutsmall, Rad_den_cutsmall, weights = Rad_cutsmall)
        elif weight == 'weightDen':
            ie_cast = single_branch(radii, xaxis, tocast_cut, ie_onmass_cut, weights = den_cut)
            orb_en_cast = single_branch(radii, xaxis, tocast_cut, orb_en_onmass_cut, weights = den_cut)
            Rad_cast = single_branch(radii, xaxis, tocast_cut, Rad_den_cut, weights = den_cut)
            Rad_castsmall = single_branch(radii, xaxis, tocast_cutsmall, Rad_den_cutsmall, weights = den_cutsmall)

            # ie_cast = single_branch(radii, xaxis, tocast_cut, ie_cut, weights = 1)
            # orb_en_cast = single_branch(radii, xaxis, tocast_cut, orb_en_cut, weights = 1)
            # Rad_cast = single_branch(radii, xaxis, tocast_cut, Rad_cut, weights = 1)
            # ie_castsmall = single_branch(radii, xaxis, tocast_cutsmall, ie_cutsmall, weights = 1)
            # orb_en_castsmall = single_branch(radii, xaxis, tocast_cutsmall, orb_en_cutsmall, weights = 1)
            # Rad_castsmall = single_branch(radii, xaxis, tocast_cutsmall, Rad_cutsmall, weights = 1)

        col_ie.append(ie_cast)
        col_orb_en.append(orb_en_cast)
        col_Rad.append(Rad_cast)
        # just cut the very low fluff
        col_Rad_cutsmall.append(Rad_castsmall)

    #%%
    if save:
        if alice:
            if check == '':
                check = 'Low'
            prepath = f'/data1/martirep/shocks/shock_capturing'
        else: 
            prepath = f'/Users/paolamartire/shocks'
        np.save(f'{prepath}/data/{folder}/coloredE_{check}{step}_{xaxis}{weight}.npy', [col_ie, col_orb_en, col_Rad_cutsmall, col_Rad])
        # np.save(f'{prepath}/data/{folder}/coloredEenergy_{check}{step}_{xaxis}.npy', [col_ie, col_orb_en, col_Rad])
        # np.save(f'{prepath}/data/{folder}/coloredEenergy_{check}{step}_{xaxis}NOCUT.npy', [col_ie_cutsmall, col_orb_en_cutsmall, col_Rad_cutsmall])
        with open(f'{prepath}/data/{folder}/coloredE_{check}{step}_days.txt', 'w') as file:
            file.write(f'# {folder}_{check}{step} \n' + ' '.join(map(str, snaps)) + '\n')
            file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
            file.close()
        np.save(f'{prepath}/data/{folder}/{xaxis}En_{check}{step}.npy', radii)