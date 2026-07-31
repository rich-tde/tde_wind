''' Convergence test for the wind properties, comparing the simulations with different resolution.'''

import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib import lines as mlines
from Utilities import prelude as prel
from Utilities import operators as op
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
checks = ['NewAMR', 'HiResNewAMR']
snaps = [362, 151]
which_obs = 'split_stream'
isoent = 'isoent' 
line_styles = ['-.', '-']
label_res = ['Middle res', 'High res']

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rt = things['Rt']
apo = things['apo']
tfallback = things['t_fb_days']
tfallback_code_units = tfallback * 24 * 3600 / prel.tsol_cgs
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs
x_test = np.arange(1., 300)
y_test2 = op.draw_line(x_test, [2e-7, -2], 'powerlaw')
y_test23 = op.draw_line(x_test, [2e2, -2/3], 'powerlaw')

figd, (axd, axV, axM, axLadv) = plt.subplots(4, 1, figsize=(8, 22)) 
figM, axT = plt.subplots(1, 1, figsize=(10, 8))
figr, ((axNcell, axNmass), (axratio, axratioM)) = plt.subplots(2, 2, figsize=(18, 18))
axT.legend(fontsize = 18)
axd.set_ylabel(r'$\rho$ (g/cm$^3$)', fontsize = 28)
# axd.plot(x_test, y_test2, c = 'gray', ls = ':', label = r'$\rho \propto r^{-2}$')
# axd.text(75, 2e-11, r'$\rho \propto r^{-2}$', fontsize = 18, color = 'gray', rotation = -20)
axV.set_ylabel(r'v$_{\rm r}$ (km/s)', fontsize = 28)
axT.set_ylabel(r'$T_{\rm rad}$ (K)', fontsize = 28)
axM.set_ylabel(r'$\dot{M} (\dot{M}_{\rm Edd})$', fontsize = 28)
axLadv.set_ylabel(r'$L_{\rm adv} (L_{\rm Edd})$', fontsize = 28)
# axLadv.plot(x_test, y_test23, c = 'gray', ls = '-.', label = r'$\rho \propto r^{-2/3}$')
# axLadv.text(58, 11, r'$L \propto r^{-2/3}$', fontsize = 18, color = 'gray', rotation = -15)   
axratio.set_ylabel(r'f$_{\rm unb}$', fontsize = 28)
axratioM.set_ylabel(r'M$_{\rm wind}/M_{\rm sec}$', fontsize = 28)
axNcell.set_ylabel(r'N$_{\rm cells}$', fontsize = 28)
axNmass.set_ylabel(r'M$_{\rm cells} [M_\odot]$', fontsize = 28)
axd.set_ylim(2e-13, 1e-5)
axV.set_ylim(1.5e3, 1.5e4)
axT.set_ylim(2e4, 1e6)
# axM.set_ylim(1e2, 1e7)
# axLadv.set_ylim(1e-2, 1e2)
axM.set_ylim(2e3, 2e7)
axLadv.set_ylim(5e-1, 2e3)
axratio.set_ylim(1e-2, 1.1)
axratioM.set_ylim(1e-2, 1.1)


all_axes = [axd, axV, axT, axM, axLadv, axratio, axratioM, axNcell, axNmass]

handles_color = []
labels_color = []
r_arr_forratio = []
d_forratio = []
v_rad_forratio = []
Mdot_forratio = []
colors_forratio = []
for k, check in enumerate(checks):
    snap = snaps[k]
    pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    path = f'{pre}/{snap}'
    tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
    print(f'For {check}, t = {tfb} tfb')

    profiles = np.load(f'{abspath}/data/{folder}/wind/r_profile/r{isoent}_profSec{snap}_{which_obs}_wind.npy', allow_pickle=True).item()
    for i, lab in enumerate(profiles.keys()): 
        colors_sec = profiles[lab]['colors_obs']
        r_arr = profiles[lab]['r'] 
        d = profiles[lab]['d_prof']
        v_rad = profiles[lab]['v_rad_prof']
        t = profiles[lab]['t_prof']
        ratio_un = profiles[lab]['ratio_un']
        Ntot_cells = profiles[lab]['Ntot_cells']
        Nwind_cells = ratio_un * Ntot_cells
        Mass_wind = profiles[lab]['Mass_wind']
        Mass_tot = profiles[lab]['Mass_tot']
        ratio_Mass = Mass_wind/Mass_tot
        if isoent == 'isoent':
            print('Isoentropic Mdot and Lum')
            area = profiles[lab]['area']
            Mdot = (4 * np.pi * r_arr**2) /area * profiles[lab]['Mdot_prof']
            L_kin = (4 * np.pi * r_arr**2) / area * profiles[lab]['L_kin_prof']
            L_adv = (4 * np.pi * r_arr**2) / area * profiles[lab]['L_adv_prof']
        else:
            Mdot = profiles[lab]['Mdot_prof'] 
            L_kin = profiles[lab]['L_kin_prof'] 
            L_adv = profiles[lab]['L_adv_prof'] 

        if lab == 'Stream side': # just to cut the initially unbound material
            idx_stop_d = np.where(np.logical_and(d > d[5], r_arr > apo))[0][0] #np.argmin(np.abs(r_plot - idx_stop_d_unb[k]*Rt)) 
            d[idx_stop_d:] = 1e-20
            Mdot[idx_stop_d:] = 1e-20
            ratio_un[idx_stop_d:] = 1e-20
            ratio_Mass[idx_stop_d:] = 1e-20
            Nwind_cells[idx_stop_d-2:] = 0 # -2 to avoid weird spikes
            Mass_wind[idx_stop_d-2:] = 0
            Ntot_cells[idx_stop_d-2:] = 0
            Mass_tot[idx_stop_d-2:] = 0
        r_arr_forratio.append(r_arr)
        d_forratio.append(d)
        v_rad_forratio.append(v_rad) 
        Mdot_forratio.append(Mdot)
        colors_forratio.append(colors_sec)

        not_zero = np.where(np.logical_and(d != 0, r_arr > 0))
        r_arr, d, v_rad, t, Mdot, L_adv, ratio_un, ratio_Mass, Nwind_cells, Ntot_cells, Mass_wind, Mass_tot = \
                make_slices([r_arr, d, v_rad, t, Mdot, L_adv, ratio_un, ratio_Mass, Nwind_cells, Ntot_cells, Mass_wind, Mass_tot], not_zero)

        if lab == 'South pole' or lab == r'Stream side $\theta\in[4\pi/9,\pi/2]$':
                    continue
        line = axd.plot(r_arr/Rt, d * prel.den_converter, color = colors_sec, ls = line_styles[k], linewidth = 2)[0]
        if check == 'HiResNewAMR': 
            handles_color.append(line)
            labels_color.append(lab)
        axV.plot(r_arr/Rt, v_rad * conversion_sol_kms, color = colors_sec, ls = line_styles[k], linewidth = 2)
        axM.plot(r_arr/Rt, Mdot/Medd_sol, color = colors_sec, ls = line_styles[k], linewidth = 2)
        axT.plot(r_arr/Rt, t, color = colors_sec, ls = line_styles[k], linewidth = 2, label = lab if check == 'HiResNewAMR' else None) 
        axLadv.plot(r_arr/Rt, L_adv/Ledd_sol, color = colors_sec, ls = line_styles[k], linewidth = 2, label = lab if check == 'HiResNewAMR' else None)
        axNcell.plot(r_arr/Rt, Nwind_cells, color = colors_sec, linewidth = 2, ls = line_styles[k])
        axNmass.plot(r_arr/Rt, Mass_wind, color = colors_sec, linewidth = 2, ls = line_styles[k])
        axratio.plot(r_arr/Rt, ratio_un, color = colors_sec, ls = line_styles[k], linewidth = 2)
        axratioM.plot(r_arr/Rt, ratio_Mass, color = colors_sec, ls = line_styles[k], linewidth = 2)
                    
legend1 = axd.legend(handles=handles_color,
                    labels=labels_color,
                    fontsize=16,
                    loc='upper right')

axd.add_artist(legend1)
# figd.legend(handles=handles_color,
#             labels=labels_color,
#             loc='upper center',
#             bbox_to_anchor=(0.525, 1.03),  # centered, near bottom of figure
#             ncol=len(labels_color),
#             fontsize=16)
# Legend 2: line-style explanation (solid vs dashed)
proxy_lines = []
proxy_lines = []
for l, line in enumerate(line_styles):
    proxy_lines.append(
        mlines.Line2D([0], [0], color='k', ls=line, linewidth=2,
                    label=label_res[l])
    )

for ax in all_axes: 
    if ax in [axd, axT, axNcell]:
        ax.legend(handles=proxy_lines, fontsize=18, loc='lower left')
        # ax.set_title(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 22)
    ax.tick_params(axis='both', which='minor', length = 8, width = 1)
    ax.tick_params(axis='both', which='major', length = 15, width = 1.5)
    ax.loglog()
    # ax.axvline(apo/Rt, color = 'gray', ls = '--')
    ax.set_xlim(1.5, 1.4e2)
    ax.grid()

# axd.text(0.8*apo/Rt, 0.2*axd.get_ylim()[1], r'$r_{\rm a}$', fontsize = 20, color = 'gray', rotation = 90)
axLadv.set_xlabel(r'$r /r_{\rm t}$', fontsize = 28)
axT.legend(fontsize = 18)
axNcell.legend(fontsize = 18)
for fig in [figd, figM, figr]:
    fig.tight_layout()
figd.savefig(f'{abspath}/Figs/2.paperWind/conv_test_rprof_{which_obs}.pdf', bbox_inches = 'tight')

#%% check by which factor they differ at the same radius, for the density and the mass outflow rate
figC, (axd_C, axV_C, axM_C) = plt.subplots(3, 1, figsize=(8, 16)) 
axd_C.set_ylabel(r'$\rho_{\rm middle}/\rho_{\rm high}$', fontsize = 28)
axV_C.set_ylabel(r'v$_{\rm r, middle}$/v$_{\rm r, high}$', fontsize = 28)
axM_C.set_ylabel(r'$\dot{M}_{\rm middle}/\dot{M}_{\rm high}$', fontsize = 28)
for i, lab in enumerate(profiles.keys()):
    if lab == 'South pole' or lab == r'Stream side $\theta\in[4\pi/9,\pi/2]$':
        continue 
    ratio = d_forratio[i]/d_forratio[i+len(profiles.keys())]
    where_nan = np.where(np.isnan(ratio))
    # ratio[where_nan] = 0
    axd_C.plot(r_arr_forratio[i]/Rt, ratio, color = colors_forratio[i], linewidth = 2)
    ratio = v_rad_forratio[i]/v_rad_forratio[i+len(profiles.keys())]
    axV_C.plot(r_arr_forratio[i]/Rt, ratio, color = colors_forratio[i], linewidth = 2)
    ratio = Mdot_forratio[i]/Mdot_forratio[i+len(profiles.keys())]
    axM_C.plot(r_arr_forratio[i]/Rt, ratio, color = colors_forratio[i], linewidth = 2)
for ax in [axd_C, axV_C, axM_C]:
    ax.set_xlim(1.5, 1.4e2)
    ax.tick_params(axis='both', which='minor', length = 8, width = 1)
    ax.tick_params(axis='both', which='major', length = 15, width = 1.5)
    ax.grid()
    ax.loglog()
    ax.set_ylim(0.5, 5)

# %%
