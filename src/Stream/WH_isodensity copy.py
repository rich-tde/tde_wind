""" Follow the evolution of one section (based on energy). """
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    from Utilities.selectors_for_snap import select_snap
    compute = True
else:
    abspath = '/Users/paolamartire/shocks'
    import matplotlib.pyplot as plt
    from Utilities.basic_units import radians
    import matplotlib.colors as colors
    compute = False

import numpy as np
from scipy.optimize import brentq
import gc
import Utilities.prelude as prel
import Utilities.sections as sec
import src.orbits as orb
from Utilities.operators import make_tree, Ryan_sampler, sort_list
from Utilities.selectors_for_snap import select_prefix, select_snap


#%%
## Parameters
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
check = 'HiResNewAMR'
compton = 'Compton'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']


dataLum = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfbs = dataLum[:, 0], dataLum[:, 1]
snaps, tfbs = sort_list([snaps, tfbs], tfbs)
snaps = [int(snap) for snap in snaps]

for i, snap in enumerate(snaps):
    theta_stream, x_stream, y_stream, z_stream, _ = \
    r_stream = np.sqrt(x_stream**2 + y_stream**2 + z_stream**2)

    theta_wh, width, N_width, height, N_height = \
        np.loadtxt(f'{abspath}/data/{folder}/WH/wh_{check}{snap}.txt')

    # print('indices for theta = -pi/2, 0, pi/2:', np.argmin(np.abs(theta_wh + np.pi/2)), np.argmin(np.abs(theta_wh)), np.argmin(np.abs(theta_wh - np.pi/2)))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        ax1.scatter(x_ax, width, c = theta_wh*radians, cmap = 'jet')
        img = ax3.scatter(x_ax, N_width, c = theta_wh*radians, cmap = 'jet')
        cbar = plt.colorbar(img, orientation = 'horizontal')
        cbar.set_label(r'$\theta$ [rad]')
        ax2.scatter(x_ax, height, c = theta_wh*radians, cmap = 'jet')
        ax2.plot(x_ax, 8*np.sqrt(x_ax), c = 'k', ls = '--')
        img = ax4.scatter(x_ax, N_height, c = theta_wh*radians, cmap = 'jet')
        cbar = plt.colorbar(img, orientation = 'horizontal')
        cbar.set_label(r'$\theta$ [rad]')
    else:
        ax1.plot(x_ax, width, c = 'darkviolet')
        ax3.plot(x_ax, N_width, c = 'darkviolet')
        ax2.plot(x_ax, height, c = 'darkviolet')
        ax4.plot(x_ax, N_height, c = 'darkviolet')
    ax1.set_ylabel(r'Width [$R_\odot$]')
    ax3.set_ylabel(r'N cells')
    ax2.set_ylabel(r'Height [$R_\odot$]')
    ax1.set_ylim(1, 15)
    ax2.set_ylim(.2, 10)
    ax3.set_ylim(10, 40)    
    ax4.set_ylim(0.9, 35)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel(r'$r [r_{\rm t}]$')
        ax.set_xlim(.5, 1e2)
        ax.loglog()
        ax.tick_params(axis='both', which='major', length = 10, width = 1.2)
        ax.tick_params(axis='both', which='minor', length = 8, width = 1)
        ax.grid()
    plt.suptitle(f't = {np.round(tfbs[i],2)} ' + r't$_{\rm fb}$', fontsize= 25)
    plt.tight_layout()
    # plt.savefig(f'{abspath}/Figs/{folder}/stream/WH_theta{snap}{x_axis}.png')
    # plt.close()


