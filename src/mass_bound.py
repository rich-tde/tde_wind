""" If alice: Compute and save the unbound mass.
If local: plots"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    path = '/home/martirep/data_pi-rossiem/TDE_data'
else:
    abspath = '/Users/paolamartire/shocks'
    path = f'{abspath}/TDE'
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices
import src.orbits as orb

#
# PARAMETERS
## 
m = 4
Mbh = 10**m
Mbh_cgs = Mbh * prel.Msol_cgs
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'NewAMR'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

#%%
# MAIN
##
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
# apo = things['apo']
# amin = things['a_mb'] # semimajor axis of the bound orbit

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    # compute the unbound mass for all the snapshots
    for i, snap in enumerate(snaps):
        print(snap, flush = True)
        pathfold = f'{path}/{folder}/snap_{snap}'
        data = make_tree(pathfold, snap, energy = True)
        X, Y, Z, mass, den, Press, vx, vy, vz, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Mass, data.Den, data.Press, data.VX, data.VY, data.VZ, data.IE, data.Rad
        IE_spec = IE_den/den
        cut = np.logical_and(den > 1e-19, np.sqrt(X**2 + Y**2 + Z**2)>Rs)
        X, Y, Z, mass, den, Press, vx, vy, vz, IE_spec, Rad_den = \
            make_slices([X, Y, Z, mass, den, Press, vx, vy, vz, IE_spec, Rad_den], cut)
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        vel = np.sqrt(vx**2 + vy**2 + vz**2)
        orb_en = orb.orbital_energy(Rsph, vel, mass, params, prel.G)
        bern = orb.bern_coeff(Rsph, vel, den, mass, Press, IE_den, Rad_den, params)
        Mass_dynunboundOE = np.sum(mass[orb_en > 0]) #- Mass_dynunboundOE
        # Mass_dynunboundOE_frombound = mstar - np.sum(mass[orb_en < 0]) #- Mass_dynunboundOE_frombound
        Mass_unbound = np.sum(mass[bern > 0]) #- Mass_dynunboundbern
        # Mass_unbound_frombound = mstar - np.sum(mass[bern < 0]) #- Mass_dynunboundbern_frombound

        with open(f'{abspath}/data/{folder}/wind/Mass_unbound{check}.csv','a', newline='') as file:
            writer = csv.writer(file)
            if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                writer.writerow(['snap', ' tfb', ' Mass unbound (bern > 0)', ' Mass unbound (OE > 0)'])
            writer.writerow([snap, tfb[i], Mass_unbound, Mass_dynunboundOE])
            file.close()

#%%
if plot:
    # among resolutions
    commonfold = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    _, tfbL, M_bernL, M_bernL, M_OEL, = np.loadtxt(f'{abspath}/data/{commonfold}LowRes/wind/Mass_unboundLowResNewAMR.csv', delimiter = ',', skiprows = 1)
    _, tfb, M_bern, M_bern, M_OE  = np.loadtxt(f'{abspath}/data/{commonfold}/wind/Mass_unboundNewAMR.csv', delimiter = ',', skiprows = 1)
    _, tfbH, M_bernH, M_bernH, M_OEH = np.loadtxt(f'{abspath}/data/{commonfold}HiRes/wind/Mass_unboundHiResNewAMR.csv', delimiter = ',', skiprows = 1)
    plt.plot(tfb, (M_bernL-M_OEL[0])/mstar, c = 'C1', label = r'$OE>0$')
    plt.plot(tfb, (M_bern-M_OE[0])/mstar,  c = 'yellowgreen', label = r'$B>0$')
    plt.plot(tfb, (M_bernH-M_OEH[0])/mstar, c = 'darkviolet', ls = '--', label = r'$M_\star - (B<0)$')
    plt.xlabel(r'$t [t_{\rm fb}]$')
    plt.ylabel(r'Mass unbound [$M_\star$]')
    plt.ylim(0.0001, 0.006)
    plt.grid()
    plt.legend(fontsize = 15)
    # plt.savefig(f'{abspath}/Figs/multiple/Mass_unbound.png', dpi = 300, bbox_inches='tight')
    plt.show()


    
# %%
