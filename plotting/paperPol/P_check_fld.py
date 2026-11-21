#!/usr/bin/env python3
abspath = '/Users/paolamartire/shocks'
import sys

sys.path.append(abspath)
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from src import orbits as orb
from Utilities.operators import split_cells, sort_list
from plotting.paperPol.P_moll import extract_poli


m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snap = 151
generat = 75
choice = 'split_stream'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
Rt = things['Rt']
Rt_cgs = Rt * prel.Rsol_cgs
tfallback_cgs = tfallback * 24 * 3600

path_file =  f'{abspath}/TDE/{folder}/Polarization/{snap}/tde_{generat}polarization{snap}.vtk'

if check == 'MG':
    time = np.loadtxt(f'{abspath}/data/{folder}/snap_{snap}/Time_snap_{snap}.txt')
    time *= prel.tsol_cgs/tfallback_cgs
    print('time MG: ', time)

folder_fld = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiResNewAMR'
data_fld = np.loadtxt(f'{abspath}/data/{folder_fld}/HiResNewAMR_red.csv', delimiter=',', dtype=float)
snaps_fld, tfb_fld, Lum_fld = data_fld[:, 0], data_fld[:, 1], data_fld[:, 2]
snaps_fld, tfb_fld, Lum_fld = sort_list([snaps_fld, tfb_fld, Lum_fld], tfb_fld, unique=True) 
snaps_fld = snaps_fld.astype(int)
if check == 'MG':
    idx_time_fld = np.argmin(np.abs(tfb_fld - time))
if check == 'HiResNewAMR':
    idx_time_fld = np.argmin(np.abs(snaps_fld - snap))
time = tfb_fld[idx_time_fld]
snap_fld = snaps_fld[idx_time_fld]
print('time fld: ', tfb_fld[idx_time_fld])
photo = np.load(f'{abspath}/data/{folder_fld}/photo/HiResNewAMR_photo{snap_fld}.npz')
x_fld, y_fld, z_fld, Lum_fld = photo['x'], photo['y'], photo['z'], photo['Lum']
r_ph_fld = np.sqrt(x_fld**2 + y_fld**2 + z_fld**2)
indices_sec, labels_sec, color_obs = split_cells(x_fld, y_fld, z_fld, choice=choice)
Lum_fld_med_idx = np.zeros(len(indices_sec))
r_ph_fld_med_idx = np.zeros(len(indices_sec))
for i, indices in enumerate(indices_sec):
    Lum_fld_med_idx[i] = np.median(Lum_fld[indices])
    r_ph_fld_med_idx[i] = np.median(r_ph_fld[indices])

data = extract_poli(
            path_file,
            fields=["forward_luminosity",
                    "photosphere_grey_radius_tau_total",
                    "photosphere_integrated_radius_tau_total",
                    "grey_luminosity",
                    "grey_isotropic_equivalent_luminosity"],)   
x = data["x"]
y = data["y"]
z = data["z"]
theta = data["theta"]
phi = data["phi"]
indices_sec, labels_sec, color_obs = split_cells(x, y, z, choice=choice)
r_ph = data["photosphere_grey_radius_tau_total"]
r_ph_int = data["photosphere_integrated_radius_tau_total"]
Lum = data["forward_luminosity"] * len(r_ph) # to make it isotropic
Lum_gray = data["grey_luminosity"] * len(r_ph) # to make it isotropic
Lum_iso = data["grey_isotropic_equivalent_luminosity"] 
rph_med_idx = np.zeros(len(indices_sec))
rph_int_med_idx = np.zeros(len(indices_sec))
Lum_med_idx = np.zeros(len(indices_sec)) 
Lum_gray_med_idx = np.zeros(len(indices_sec))
Lum_iso_med_idx = np.zeros(len(indices_sec))
for i, indices in enumerate(indices_sec):
    rph_med_idx[i] = np.median(r_ph[indices])
    rph_int_med_idx[i] = np.median(r_ph_int[indices])
    Lum_gray_med_idx[i] = np.median(Lum_gray[indices])
    Lum_med_idx[i] = np.median(Lum[indices])
    Lum_iso_med_idx[i] = np.median(Lum_iso[indices])

figc, (axR, axL) = plt.subplots(1, 2, figsize=(16, 8))
for i in range(len(indices_sec)):
    axR.scatter(i, rph_med_idx[i]/prel.Rsol_cgs/r_ph_fld_med_idx[i], color=color_obs[i], s = 120, marker='o', label=r'photosphere_grey_radius_tau_total' if i == 1 else "")
    axR.scatter(i, rph_int_med_idx[i]/prel.Rsol_cgs/r_ph_fld_med_idx[i], color=color_obs[i], s = 120, marker='*', label=r'photosphere_integrated_radius_tau_total' if i == 1 else "")
    axL.scatter(i, Lum_med_idx[i]/Lum_fld_med_idx[i], color=color_obs[i], s = 120, marker='x', label=r'$N_{\rm obs}\times$ forward_luminosity' if i == 1 else "")
    axL.scatter(i, Lum_gray_med_idx[i]/Lum_fld_med_idx[i], color=color_obs[i], s = 120, marker='^', label=r'$N_{\rm obs}\times$ grey_luminosity' if i == 1 else "")
    axL.scatter(i, Lum_iso_med_idx[i]/Lum_fld_med_idx[i], color=color_obs[i], s = 120, marker='s', label=f'grey_isotropic_equivalent_luminosity' if i == 1 else "")
for ax in [axL, axR]:
    ax.grid()
    ax.axhline(1, color='gray', ls='--', lw=1)
    ax.legend(fontsize=18)
    ax.set_xticks(np.arange(len(indices_sec)))
    ax.set_xticklabels(labels_sec, fontsize=18, rotation=45)
axR.set_ylabel('MC/fld')
plt.suptitle(f'time = {time:.2f} t_fb, generations = {generat}', fontsize=20)
plt.tight_layout()
                                
