""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    abspath = '/Users/paolamartire/shocks'
    save = False

import gc
import warnings
warnings.filterwarnings('ignore')
import csv

import numpy as np
from src.Opacity.linextrapolator import nouveau_rich

import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import src.orbits as orb

#%% Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' # 'QuadraticOpacity' # '' or 'HiRes'
Rt = orb.tidal_radius(Rstar, mstar, Mbh)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

## Snapshots stuff
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
print('we are in: ', pre)

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
    opac_path = f'{abspath}/src/Opacity'
    T_cool = np.loadtxt(f'{opac_path}/T.txt')
    min_T = np.exp(T_cool[0]) 
    max_T = np.exp(T_cool[-1])
    Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
    min_rho = np.exp(Rho_cool[0]) /prel.den_converter #from log-cgs to code units
    rossland = np.loadtxt(f'{opac_path}/ross.txt')
    T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5, treat_den = 'linear')
    R_bin = np.logspace(-1, 3.5, 5_000) 
    np.savetxt(f'{abspath}/data/{folder}/testOpac/{check}_TestOpacRbin.txt', R_bin)
    Tvol_all = np.zeros(len(snaps))
    for idx_s, snap in enumerate(snaps):
        print('\n Snapshot: ', snap, '\n', flush=True)
        sys.stdout.flush()
        # Load data -----------------------------------------------------------------
        if alice:
            X = np.load(f'{pre}/snap_{snap}/CMx_{snap}.npy')
            Y = np.load(f'{pre}/snap_{snap}/CMy_{snap}.npy')
            Z = np.load(f'{pre}/snap_{snap}/CMz_{snap}.npy')
            T = np.load(f'{pre}/snap_{snap}/T_{snap}.npy')
            Den = np.load(f'{pre}/snap_{snap}/Den_{snap}.npy')
            Rad_spec = np.load(f'{pre}/snap_{snap}/Rad_{snap}.npy')
            Vol = np.load(f'{pre}/snap_{snap}/Vol_{snap}.npy')
        
        Rad_den = Rad_spec * Den
        R = np.sqrt(X**2 + Y**2 + Z**2)
        denmask = Den > 1e-19
        R, T, Den, Rad_den, Vol = make_slices([R, T, Den, Rad_den, Vol], denmask) 
        f_Rall, bins_edges = np.histogram(R, bins = R_bin)

        lowden = Den < min_rho
        R, T, Rad_den, Vol = make_slices([R, T, Rad_den, Vol], lowden)
        f_R, bins_edges = np.histogram(R, bins = R_bin)
        f_R = f_R/f_Rall

        T_rad = (Rad_den*prel.en_den_converter/ prel.alpha_cgs)**(1/4)
        Tvol = np.zeros(len(T))
        Tvol = np.where(T >= 1.1*T_rad, T**4 * Vol, 0)
        print(T, flush=True)
        sys.stdout.flush()
        Tvol_all[idx_s] = np.sum(Tvol)

        inside_tab = np.logical_and(T > min_T, T < max_T)
        R, Rad_den, Vol = make_slices([R, Rad_den, Vol], inside_tab)
        f_RnoT, bins_edges = np.histogram(R, bins = R_bin)
        f_RnoT = f_RnoT/f_Rall

        with open(f'{abspath}/data/{folder}/testOpac/{check}_TestOpac{snap}.txt', 'w') as file:
            file.write('# f_R (i.e. fraction of cells in each Rbin with Density lower than Table lower limit)\n')
            file.write(f' '.join(map(str, f_R)) + '\n')  
            file.write('# f_RnoT (i.e. fraction of cells in each Rbin with Density lower than Table lower limit but T in table)\n')
            file.write(f' '.join(map(str, f_RnoT)) + '\n')
            file.close()

    with open(f'{abspath}/data/{folder}/testOpac/{check}_Tvol.txt', 'w') as file:
        file.write('# t/tfb \n')
        file.write(f' '.join(map(str, tfb)) + '\n') 
        file.write('# Vol * T_gas^4 for extrapolat4ed cells with t_gas>=1.1. T_rad\n')
        file.write(f' '.join(map(str, Tvol_all)) + '\n')  
        file.close()

if plot:
    import matplotlib.pyplot as plt

    time = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/Ecc_high_{check}_days.txt')
    snaps = [int(snap) for snap in time[0]]
    tfb = time[1]
    R_bin = np.loadtxt(f'{abspath}/data/{folder}/testOpac/{check}_TestOpacRbin.txt')
    mid_Rbin = (R_bin[:-1] + R_bin[1:]) / 2
    Tvol_all = np.loadtxt(f'{abspath}/data/{folder}/testOpac/{check}_Tvol.txt')
    for i, snap in enumerate(snaps):
        data = np.loadtxt(f'{abspath}/data/{folder}/testOpac/{check}_TestOpac{snap}.txt')
        f_R = data[0]
        f_RnoT = data[1]
        where_nan = np.isnan(f_R)
        f_R = f_R[~where_nan] 
        f_RnoT = f_RnoT[~where_nan]
        mid_Rbin_plot = mid_Rbin[~where_nan]

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.scatter(mid_Rbin_plot/Rt, f_R, s = 1, c = 'k')#, label=f'all low density')
        ax.scatter(mid_Rbin_plot/Rt, f_RnoT, s = 1, c = 'dodgerblue')#, label=f'low density inside table')
        ax.axvline(0.6, color = 'k', linestyle = '--', alpha = 0.5)
        ax.axvline(apo/Rt, color = 'k', linestyle = '--', alpha = 0.5)
        ax.set_ylim(1e-2, 1.1)
        ax.set_xlim(np.min(mid_Rbin/Rt), 1.2*np.max(mid_Rbin/Rt))
        ax.set_xlabel(r'$R [R_{\rm t}]$')
        ax.set_ylabel(r'$f_R$')
        ax.loglog()
        ax.text(0.015, 0.02, f't = {tfb[i]:.2f} ' + r'$t_{\rm fb}$', fontsize=18)
        # ax.legend()
        plt.savefig(f'{abspath}/Figs/{folder}/testOpac/{check}_TestOpac{snap}.png', bbox_inches='tight')
        plt.close()
    
    plt.figure(figsize=(7, 5))
    Rad_extr = prel.alpha_cgs * Tvol_all * prel.Rsol_cgs**3
    plt.scatter(tfb, Tvol_all, s = 1, c = 'k')


        



# %%
