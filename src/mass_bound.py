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
check = ''
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

#%%
# MAIN
##
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
Rs = 2*prel.G*Mbh/prel.csol_cgs**2
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
Rcheck = np.array([R0, Rt, a_mb, apo])
tfallback = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    # find the unboud mass in the first snapshot which is due to the disruption
    data = make_tree(f'{path}/{folder}/snap_{snaps[0]}', snaps[0], energy = True)
    X, Y, Z, mass, den, Press, vx, vy, vz, IE_den = \
        data.X, data.Y, data.Z, data.Mass, data.Den, data.Press, data.VX, data.VY, data.VZ, data.IE
    IE_spec = IE_den/den
    cut = np.logical_and(den > 1e-19, np.sqrt(X**2 + Y**2 + Z**2)>Rs)
    X, Y, Z, mass, den, Press, vx, vy, vz, IE_spec = \
        make_slices([X, Y, Z, mass, den, Press, vx, vy, vz, IE_spec], cut)
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)
    vel = np.sqrt(vx**2 + vy**2 + vz**2)
    orb_en = orb.orbital_energy(Rsph, vel, mass, prel.G, prel.csol_cgs, Mbh) 
    Mass_dynunboundOE = np.sum(mass[orb_en > 0]) 
    Mass_dynunboundOE_frombound = mstar - np.sum(mass[orb_en < 0]) 
    bern = orb_en/mass + IE_spec + Press/den
    Mass_dynunboundbern = np.sum(mass[bern > 0]) 
    Mass_dynunboundbern_frombound = mstar - np.sum(mass[bern < 0]) 
    # compute the unbound mass for all the snapshots
    Mass_dynunboundOE = np.zeros(len(snaps))
    Mass_dynunboundOE_frombound = np.zeros(len(snaps))
    Mass_unbound = np.zeros(len(snaps))
    Mass_unbound_frombound = np.zeros(len(snaps))
    for i,snap in enumerate(snaps):
        print(snap)
        pathfold = f'{path}/{folder}/snap_{snap}'
        data = make_tree(pathfold, snap, energy = True)
        X, Y, Z, mass, den, Press, vx, vy, vz, IE_den = \
        data.X, data.Y, data.Z, data.Mass, data.Den, data.Press, data.VX, data.VY, data.VZ, data.IE
        IE_spec = IE_den/den
        cut = np.logical_and(den > 1e-19, np.sqrt(X**2 + Y**2 + Z**2)>Rs)
        X, Y, Z, mass, den, Press, vx, vy, vz, IE_spec = \
            make_slices([X, Y, Z, mass, den, Press, vx, vy, vz, IE_spec], cut)
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        vel = np.sqrt(vx**2 + vy**2 + vz**2)
        orb_en = orb.orbital_energy(Rsph, vel, mass, prel.G, prel.csol_cgs, Mbh) 
        Mass_dynunboundOE[i] = np.sum(mass[orb_en > 0]) #- Mass_dynunboundOE
        Mass_dynunboundOE_frombound[i] = mstar - np.sum(mass[orb_en < 0]) #- Mass_dynunboundOE_frombound
        bern = orb_en/mass + IE_spec + Press/den
        Mass_unbound[i] = np.sum(mass[bern > 0]) #- Mass_dynunboundbern
        Mass_unbound_frombound[i] = mstar - np.sum(mass[bern < 0]) #- Mass_dynunboundbern_frombound

    with open(f'{abspath}/data/{folder}/Mass_unbound{check}.txt','w') as file:
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')  
        file.write('# unbound mass [M_odot] considering bern > 0 \n' + ' '.join(map(str, Mass_unbound)) + '\n')  
        file.write('# unbound mass [M_odot] considering Mstar - [bern < 0] \n' + ' '.join(map(str, Mass_unbound_frombound)) + '\n')
        file.write('# unbound mass [M_odot] considering OE > 0 \n' + ' '.join(map(str, Mass_dynunboundOE)) + '\n')
        file.write('# unbound mass [M_odot] considering Mstar - [OE < 0] \n' + ' '.join(map(str, Mass_dynunboundOE_frombound)) + '\n')
        file.close()

#%%
if plot:
    commonfold = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    tfbL, M_unL, _, M_unnocutL = np.loadtxt(f'{abspath}/data/{commonfold}LowRes/Mass_unboundLowRes.txt')
    tfb, M_bern, M_bern_frombound, M_orben, M_orben_frombound = np.loadtxt(f'{abspath}/data/{commonfold}/Mass_unbound.txt')
    tfbH, M_unH, _, M_unnocutH = np.loadtxt(f'{abspath}/data/{commonfold}HiRes/Mass_unboundHiRes.txt')
    # plt.plot(tfbL, M_unL/mstar, c = 'C1', label = 'Low')
    plt.plot(tfb, M_bern/mstar, c = 'yellowgreen', label = r'$B>0$')
    plt.plot(tfb, M_bern_frombound/mstar, c = 'forestgreen', ls = '--', label = r'$M_\star - (B<0)$')
    plt.plot(tfb, M_orben/mstar, c = 'dodgerblue', label = r'$OE>0$')
    plt.plot(tfb, M_orben_frombound/mstar, c = 'b', ls = '--', label = r'$M_\star - (OE<0)$')
    # plt.plot(tfbH, M_unH/mstar, c = 'darkviolet', label = 'High')
    plt.xlabel(r'$t [t_{\rm fb}]$')
    plt.ylabel(r'Mass unbound [$M_\star$]')
    plt.grid()
    plt.legend(fontsize = 15)
    # plt.savefig(f'{abspath}/Figs/multiple/Mass_unbound.png', dpi = 300, bbox_inches='tight')
    plt.show()

    
# %%
