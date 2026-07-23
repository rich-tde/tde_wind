""" Find/plot radial profiles as weighted average on spherical sections. 
Find/plot polar profiles for fixed r and phi_array. 
Written to be run locally."""

import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib import lines as mlines
import healpy as hp
from Utilities.basic_units import radians
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix
import src.orbits as orb
import Utilities.operators as op

#
# PARAMS
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
which_obs = 'left_right_z' # 'left_right_z', 'all' or 'in_out_z'
which_part = 'wind' 
snap = 109

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
v_esc_kms = v_esc * conversion_sol_kms
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs


observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
indices_obs, label_obs, colors_obs, _ = op.choose_observers(observers_xyz, which_obs)
fig, (axd, axV, axM, axLkin) = plt.subplots(4, 1, figsize=(8, 22)) 
figM, (axT, axLadv) = plt.subplots(1, 2, figsize=(15, 7))
figr, axratio = plt.subplots(1, 1, figsize=(12, 10))
all_axes = [axd, axV, axT, axM, axLadv, axLkin, axratio]
axd.set_ylim(5e-14, 1e-8)
axV.set_ylim(1.5e3, 1.5e4)
axM.set_ylim(1e1, 1e6)
axLkin.set_ylim(1e-4, 5) 
path = f'{pre}/{snap}'
tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
radii_chosen_name = ['05amin', 'amin', 'apo']
line_styles_r = [ 'dotted', 'dashed', 'solid']

handles_color = []
labels_color = []
for k, r_chosen_name in enumerate(radii_chosen_name):
    # Load profiles
    profiles = np.load(f'{abspath}/data/{folder}/wind/theta_profile/theta{r_chosen_name}_profSec{snap}_{which_obs}_{which_part}.npy', allow_pickle=True).item()
    for i, lab in enumerate(profiles.keys()):
        if i > 1: #label_obs[i] == 'South pole':
            continue 
        r_plot = profiles[lab]['r'] 
        d = profiles[lab]['d_prof']
        v_rad = profiles[lab]['v_rad_prof'] 
        t = profiles[lab]['t_prof']
        Mdot = profiles[lab]['Mdot_prof'] #Mdotmean_prof
        L_adv = profiles[lab]['L_adv_prof'] #L_advmean_prof
        L_kin = profiles[lab]['L_kin_prof'] #L_kinmean_prof
        ratio_un = profiles[lab]['ratio_un']
        colors_sec = profiles[lab]['colors_obs']
        # Mdot = d * r_plot**2 * v_rad
        not_zero = np.where(d != 0) 
        
        line = axd.plot(r_plot*radians, d * prel.den_converter, label = f'{lab}' if k == 2 else '', color = colors_sec, ls = line_styles_r[k], linewidth = 2)[0]
        axV.plot(r_plot*radians, v_rad * conversion_sol_kms, color = colors_sec, ls = line_styles_r[k], linewidth = 2)
        axM.plot(r_plot*radians, Mdot/Medd_sol, color = colors_sec, ls = line_styles_r[k], linewidth = 2)
        axratio.plot(r_plot*radians, ratio_un, label = f'{lab}' if k == 2 else '', color = colors_sec, ls = line_styles_r[k], linewidth = 2)
        axT.plot(r_plot*radians, t, label = f'{lab}' if k == 2 else '', color = colors_sec, ls = line_styles_r[k], linewidth = 2)
        axLadv.plot(r_plot*radians, L_adv/Ledd_sol, color = colors_sec, ls = line_styles_r[k], linewidth = 2)
        axLkin.plot(r_plot*radians, L_kin/Ledd_sol, color = colors_sec, ls = line_styles_r[k], linewidth = 2)

        if k == 2:
            handles_color.append(line)
            labels_color.append(lab)
            
axV.axhline(v_esc_kms, c = 'k', ls = 'dotted')# 
# axV.text(35, 1.1*0.2*v_esc_kms, r'0.2v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
# axd.legend(fontsize = 18, loc = 'upper right')

# Legend 1: colored observer lines (three colors)
legend1 = axd.legend(handles=handles_color,
                labels=labels_color,
                fontsize=16,
                loc='upper right')
axd.add_artist(legend1)

# Legend 2: line-style explanation (solid vs dashed)
proxy_lines = []
proxy_lines = []
for l, line in enumerate(line_styles_r):
    proxy_lines.append(
        mlines.Line2D([0], [0], color='cornflowerblue', ls=line, linewidth=2,
                    label= f'r = {radii_chosen_name[l]}'))

legend2 = axd.legend(handles=proxy_lines, fontsize=16, 
                    loc='lower left')

for ax in all_axes: 
    ax.tick_params(axis='both', which='minor', length = 8, width = 1)
    ax.tick_params(axis='both', which='major', length = 15, width = 1.5)
    ax.set_yscale('log')
    ax.set_xlim(np.pi/3, 2*np.pi/3)
    ax.set_xticks([np.pi/3, 4*np.pi/9, np.pi/2, 5*np.pi/9, 2*np.pi/3])
    ax.set_xticklabels([r'$\pi/3$', r'$4\pi/9$', r'$\pi/2$', r'$5\pi/9$', r'$2\pi/3$'])
    # ax.set_xticks([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])
    # ax.set_xticklabels([r'$0$', r'$\pi/6$', r'$\pi/3$', r'$\pi/2$', r'$2\pi/3$', r'$5\pi/6$', r'$\pi$'])
    ax.grid()
    if ax != axLkin:
        ax.set_xlabel('')
    
axd.set_ylabel(r'$\rho$ (g/cm$^3$)', fontsize = 28)
axV.set_ylabel(r'v$_{\rm r}$ (km/s)', fontsize = 28)
axT.set_ylabel(r'$T_{\rm rad}$ (K)', fontsize = 28)
axM.set_ylabel(r'$\dot{M} (\dot{M}_{\rm Edd})$', fontsize = 28)
axLadv.set_ylabel(r'$L_{\rm adv} (L_{\rm Edd})$', fontsize = 28)
axLkin.set_ylabel(r'$L_{\rm kin} (L_{\rm Edd})$', fontsize = 28)
axratio.set_ylabel(r'f$_{\rm unb}$', fontsize = 28)
axT.legend(fontsize = 18)
axratio.legend(fontsize = 18)
axLkin.set_xlabel(r'$\theta$', fontsize = 28)
axd.set_title(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 20)
fig.tight_layout()
figM.tight_layout()
fig.savefig(f'{abspath}/Figs/{folder}/wind/den_prof_theta_Revol{snap}.png', bbox_inches = 'tight')
