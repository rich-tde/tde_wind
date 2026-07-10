import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib import lines as mlines
import healpy as hp
from Utilities.basic_units import radians
from Utilities.sections import make_slices
import Utilities.operators as op
from Utilities.selectors_for_snap import select_prefix
import src.orbits as orb
import Utilities.prelude as prel
from src.Wind.Rtrapp_tdiff import load_and_adjust_rtrap

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
which_obs = 'left_right_z' # 'left_right_z', 'all' or 'in_out_z'
r_chosen_name_theta = 'apo' 
snap = 151

pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
path = f'{pre}/{snap}'
tfb = np.loadtxt(f'{path}/tfb_{snap}.txt')
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']
tfallback = things['t_fb_days']
tfallback_code_units = tfallback * 24 * 3600 / prel.tsol_cgs
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
v_esc_kms = v_esc * conversion_sol_kms
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs
x_test = np.arange(1., 350)
y_test2 = op.draw_line(x_test, [2e-7, -2], 'powerlaw')
# arrange for plotting
observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
indices_obs, label_obs, colors_obs, _ = op.choose_observers(observers_xyz, which_obs)

what_varies_all = ['r','theta'] 
r_chosen_names = ['', r_chosen_name_theta]
norms = [1/Rt, radians]
which_parts = ['outflow', 'wind'] 
labels_parts =  ['(unbound + bound) Outflow', 'Unbound outflow (wind)'] #['Unbound outflow (wind)']#
line_styles_parts = ['--', '-'] 
    
fig, all_axes = plt.subplots(4, 2, figsize=(22, 26)) 
for j in range(2):
    all_axes[0][j].set_ylim(2e-13, 1e-5)
    all_axes[1][j].axhline(v_esc_kms, c = 'k', ls = 'dotted')
    all_axes[1][j].set_ylim(1.5e3, 1.5e4)
    all_axes[2][j].set_ylim(5e1, 5e6)
    all_axes[3][j].set_ylim(1e-3, 7e2)
    for i in range(4):
        all_axes[i][j].grid()
        all_axes[i][j].tick_params(axis='both', which='minor', length = 8, width = 1, labelsize = 35)
        all_axes[i][j].tick_params(axis='both', which='major', length = 15, width = 1.5, labelsize = 35)
        if j == 0:
            all_axes[i][j].loglog()
            all_axes[i][j].axvline(apo*norms[j], color = 'gray', ls = '--')
            all_axes[i][j].set_xlim(1.5, 1.4e2)
        elif j == 1:
            all_axes[i][j].set_yscale('log')
            all_axes[i][j].set_xlim(0, 2*np.pi/3)
            if i != 3:
                all_axes[i][1].set_xlabel('', fontsize = 35)

all_axes[3][0].set_xlabel(r'$r /r_{\rm t}$', fontsize = 35)
all_axes[3][1].set_xlabel(r'$\theta$ (rad)', fontsize = 35)

all_axes[0][0].text(0.8*apo*norms[0], 0.2*all_axes[0][0].get_ylim()[1], r'$r_{\rm a}$', fontsize = 25, color = 'gray', rotation = 90)   
all_axes[0][0].plot(x_test, y_test2, c = 'gray', ls = '-.', label = r'$\rho \propto r^{-2}$')
all_axes[0][0].text(76, 2e-11, r'$\rho \propto r^{-2}$', fontsize = 25, color = 'gray', rotation = -20)            
handles_color = []
labels_color = []
        
if check == 'HiResNewAMR':
    photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
    xph, yph, zph = photo['x'], photo['y'], photo['z']
else:
    photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    xph, yph, zph = photo[0], photo[1], photo[2]
rph_all = np.sqrt(xph**2 + yph**2 + zph**2)
pathtrap = f"{abspath}/data/{folder}/trap"
dataRtr = load_and_adjust_rtrap(pathtrap, check, snap)
x_tr, y_tr, z_tr, den_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr']
r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
rph_medians = []
rph_nonzero_medians = []
rtr_medians = []
rtr_nonzero_medians = []
for i, idx_list in enumerate(indices_obs): 
    rph_medians.append(np.median(rph_all[idx_list]))
    rtr_medians.append(np.median(r_tr_all[idx_list]))
    non_zero = idx_list[r_tr_all[idx_list]> Rt]
    rph_nonzero_medians.append(np.median(rph_all[non_zero]))
    rtr_nonzero_medians.append(np.median(r_tr_all[non_zero]))
for w, what_varies in enumerate(what_varies_all):
    norm = norms[w]
    for k, which_part in enumerate(which_parts):
        # Load profiles
        profiles = np.load(f'{abspath}/data/{folder}/wind/{what_varies}_profile/{what_varies}{r_chosen_names[w]}_profSec{snap}_{which_obs}_{which_part}.npy', allow_pickle=True).item()
        for i, lab in enumerate(profiles.keys()):
            if lab == 'South pole':
                continue 
            r_arr = profiles[lab]['r'] 
            d = profiles[lab]['d_prof']
            v_rad = profiles[lab]['v_rad_prof'] 
            Mdot = profiles[lab]['Mdot_prof'] #Mdotmean_prof
            L_kin = profiles[lab]['L_kin_prof'] #L_kinmean_prof
            colors_sec = colors_obs[i] 
            not_zero = np.where(np.logical_and(d != 0, r_arr > 0))
            if what_varies == 'r':
                r_plot, d, v_rad,  Mdot, L_kin = \
                    make_slices([r_arr, d, v_rad, Mdot, L_kin], not_zero)
                idx_rtr = np.argmin(np.abs(r_plot - rtr_nonzero_medians[i]))
                idx_rph = np.argmin(np.abs(r_plot - rph_nonzero_medians[i]))
                if lab == 'Stream side': # just to cut the initially unbound material
                    idx_stop_d = np.where(np.logical_and(d > d[0], r_plot > apo))[0][0] #np.argmin(np.abs(r_plot - idx_stop_d_unb[k]*Rt)) 
                    d[idx_stop_d:] = 1e-20
                    Mdot[idx_stop_d:] = 1e-20
                    L_kin[idx_stop_d:] = 1e-20
            else:
                r_plot = r_arr 
            
            if np.logical_and(w == 0, which_part == 'wind'): 
                line = all_axes[0][w].plot(r_plot*norm, d * prel.den_converter, label = f'{lab}', color = colors_sec, ls = line_styles_parts[k], linewidth = 2)[0]
                handles_color.append(line)
                labels_color.append(lab)
            else:
                all_axes[0][w].plot(r_plot*norm, d * prel.den_converter, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
            all_axes[1][w].plot(r_plot*norm, v_rad * conversion_sol_kms, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
            all_axes[2][w].plot(r_plot*norm, Mdot/Medd_sol, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
            all_axes[3][w].plot(r_plot*norm, L_kin/Ledd_sol, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)

            if np.logical_and(what_varies =='r' , which_part == 'wind'):
                all_axes[0][w].scatter(r_plot[idx_rph]*norm, d[idx_rph] * prel.den_converter, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k])
                all_axes[1][w].scatter(r_plot[idx_rph]*norm, v_rad[idx_rph] * conversion_sol_kms, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k])
                all_axes[2][w].scatter(r_plot[idx_rph]*norm, Mdot[idx_rph]/Medd_sol, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k])
                all_axes[3][w].scatter(r_plot[idx_rph]*norm, L_kin[idx_rph]/Ledd_sol, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k])

                all_axes[0][w].scatter(r_plot[idx_rtr]*norm, d[idx_rtr] * prel.den_converter, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k])
                all_axes[1][w].scatter(r_plot[idx_rtr]*norm, v_rad[idx_rtr] * conversion_sol_kms, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k])
                all_axes[2][w].scatter(r_plot[idx_rtr]*norm, Mdot[idx_rtr]/Medd_sol, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k])
                all_axes[3][w].scatter(r_plot[idx_rtr]*norm, L_kin[idx_rtr]/Ledd_sol, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k])
                


# Legend 1: colored observer lines (three colors)
legend1 = all_axes[0][0].legend(handles=handles_color,
                    labels=labels_color,
                    fontsize=20,
                    loc='upper right')
all_axes[0][0].add_artist(legend1)

# Legend 2: line-style explanation (solid vs dashed)
proxy_lines = []
proxy_lines = []
for l, line in enumerate(line_styles_parts):
    proxy_lines.append(
        mlines.Line2D([0], [0], color='k', ls=line, linewidth=2,
                    label=labels_parts[l])
    )

all_axes[0][0].legend(handles=proxy_lines, fontsize=20, 
                loc='lower left')

all_axes[0][0].set_ylabel(r'$\rho$ (g/cm$^3$)', fontsize = 35)
all_axes[1][0].set_ylabel(r'v$_{\rm r}$ (km/s)', fontsize = 35)
all_axes[2][0].set_ylabel(r'$\dot{M} (\dot{M}_{\rm Edd})$', fontsize = 35)
all_axes[3][0].set_ylabel(r'$L_{\rm adv} (L_{\rm Edd})$', fontsize = 35)
fig.suptitle(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 28, y = 1, x = 0.53)
fig.tight_layout(w_pad=15.0)
fig.savefig(f'{abspath}/Figs/2.paperWind/den_profRtheta_{snap}.pdf', bbox_inches = 'tight')
