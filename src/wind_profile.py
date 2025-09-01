""" Find density and (radial velocity) profiles for different lines of sight. NB: observers have to be noramlized to 1. """
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cm

import sys
sys.path.append(abspath)

import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
# import matlab.engine
import k3match
from sklearn.neighbors import KDTree
import healpy as hp
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import to_spherical_components, sort_list, make_tree, choose_observers

def CouBegel(r, theta, q, gamma=4/3):
    # I'm missing the normalization
    alpha = (1-q*(gamma-1))/(gamma-1)
    rho = r**(-q) * np.sin(theta)**(2*alpha)
    return rho

def Metzger(r, q, Mbh, mstar, Rstar, k = 0.9, Me = 1, G = prel.G):
    # I'm missing the correct noramlization
    Rv = 2 * Rstar/(5*k) * (Mbh/mstar)**(2/3) * Me/mstar
    norm = Me*(3-q)/(4*np.pi*Rv**3 * (7-2*q))
    if norm == 0.0:
        print('norm is 0')
    if r < Rv:
        rho = (r/Rv)**(-q)
    else:
        rho = np.exp(-(r-Rv)/Rv)
    return (norm * rho)


#%%
# MAIN
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'NewAMR' 
which_obs = 'arch' # 'arch', 'quadrants', 'axis'
which_part = 'outflow'
snap = 318

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
t_fb_days_cgs = things['t_fb_days'] * 24 * 3600 # in seconds
t_fb_sol = t_fb_days_cgs/prel.tsol_cgs
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
v_esc_kms = v_esc * conversion_sol_kms
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
Medd = Ledd/(0.1*prel.c_cgs**2)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
tfb = np.loadtxt(f'{abspath}/TDE/{folder}/{snap}/tfb_{snap}.txt')
# print('t dyn/tfb', (apo/(1e9/prel.Rsol_cgs))/t_fb_days_cgs)
#%% MATLAB 
# eng = matlab.engine.start_matlab()
# Observers
observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
r_obs = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
indices_sorted, label_obs, colors_obs = choose_observers(observers_xyz, which_obs)
observers_xyz = np.transpose(observers_xyz) #shape: Nx3

#%% Load data -----------------------------------------------------------------
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
if alice:
    path = f'{pre}/snap_{snap}'
else:
    path = f'{pre}/{snap}'
data = make_tree(path, snap, energy = True)
X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
    data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
cut = Den > 1e-19
X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
    make_slices([X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den], cut)       
R = np.sqrt(X**2 + Y**2 + Z**2)  
vel = np.sqrt(VX**2 + VY**2 + VZ**2)  
bern = orb.bern_coeff(R, vel, Den, Mass, Press, IE_den, Rad_den, params)

xyz = np.array([X, Y, Z]).T
tree = KDTree(xyz, leaf_size = 50) 
N_ray = 5_000
rmax = 10*apo
r = np.logspace(np.log10(Rt), np.log10(rmax), N_ray)
with open(f'{abspath}/data/{folder}/wind/den_prof{snap}{which_obs}{which_part}.txt','w') as file:
    file.write(f'# radii \n')
    file.write(f' '.join(map(str, r)) + '\n')
    file.close()

for j, idx_list in enumerate(indices_sorted):
    print(label_obs[j], flush= True)
    d_all = []
    v_rad_all = []
    t_all = []
    rad_den_all = []    
    for i in idx_list: # i in [0, 192]: pick the line of sight that you'll use for the mean of the chosen direction
        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        del x, y, z

        # ray tracing along observer i 
        _, idx = tree.query(xyz2, k=1) #you can do k=4, comment the line afterwards and then do d = np.mean(d, axis=1) and the same for all the quantity. But you doesn't really change since you are averaging already on line of sights
        idx = [ int(idx[i][0]) for i in range(len(idx))]
        idx = np.array(idx)
        # Quantity corresponding to the ray
        d = Den[idx] * prel.den_converter
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        ray_t = T[idx]
        ray_bern = bern[idx]  
        ray_rad_den = Rad_den[idx]
        
        v_rad, _, _ = to_spherical_components(ray_vx, ray_vy, ray_vz, ray_x, ray_y, ray_z)
        if which_part == 'outflow':
            discard = np.logical_or(ray_bern < 0, v_rad < 0)
        if which_part == 'inflow':
            discard = np.logical_or(ray_bern > 0, v_rad > 0)
        d[discard] = 0
        v_rad[discard] = 0
        ray_t[discard] = 0
        ray_rad_den[discard] = 0
        # store
        d_all.append(d)
        v_rad_all.append(v_rad)
        t_all.append(ray_t)
        rad_den_all.append(ray_rad_den)

    # all the list are of shape (len(idx_list), N_ray)
    d_mean = np.mean(d_all, axis=0) # shape: (N_ray,)
    v_rad_mean = np.mean(v_rad_all, axis=0)
    t_mean = np.mean(t_all, axis=0)
    rad_den_mean = np.mean(rad_den_all, axis=0)

    with open(f'{abspath}/data/{folder}/wind/den_prof{snap}{which_obs}{which_part}.txt','a') as file:
        file.write(f'# Observer {label_obs[j]}. NB: only density is in CGS \n')
        file.write(f' '.join(map(str, d_mean)) + '\n')
        file.write(f' '.join(map(str, v_rad_mean)) + '\n')
        file.write(f' '.join(map(str, t_mean)) + '\n')
        file.write(f' '.join(map(str, rad_den_mean)) + '\n')
        file.close()
#%%
# R_edge = v_esc / prel.tsol_cgs * tfb * t_fb_days_cgs
x_test = np.arange(1e-2, 10)
y_test23 = 7e4*(x_test)**(-2/3)
y_test2 = 4e-13* (x_test)**(-2)
y_testplus2 = 1e2* (x_test)**(2)
y_test3 = 1e-12 * (x_test)**(-3)
which_part = 'outflow'
profiles = np.loadtxt(f'{abspath}/data/{folder}/wind/den_prof{snap}{which_obs}{which_part}.txt')
r, d_mean, v_rad_mean, t_mean, rad_den_mean = \
    profiles[0], profiles[1::4], profiles[2::4], profiles[3::4], profiles[4::4]

ph_data = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
xph, yph, zph = ph_data[0], ph_data[1], ph_data[2]
rph = np.sqrt(xph**2 + yph**2 + zph**2)
rph = rph[indices_sorted]
rph_mean = np.mean(rph, axis=1)
        
# find the average trapping radius for the directions chosen
dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz")
x_tr, y_tr, z_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr']
r_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
r_tr = r_tr[indices_sorted]
r_tr_mean = np.mean(r_tr, axis=1) 

normalize_by = ''
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
figT, axT = plt.subplots(1, 1, figsize=(10, 7))
figL, axL = plt.subplots(1, 1, figsize=(10, 7))
for i in range(len(d_mean)):
    if i not in [0, 2]:
        continue
    r_normalizer = apo if normalize_by == 'apo' else r_tr_mean[i]

    d = d_mean[i]
    v_rad = v_rad_mean[i] 
    t = t_mean[i]
    rad_den = rad_den_mean[i]
    L = 4 *np.pi * r**2 * rad_den * v_rad * prel.en_converter / prel.tsol_cgs
    # L = 4/3 * np.pi * r**2 * v_rad * prel.Rsol_cgs**3/prel.tsol_cgs * t**4 * prel.alpha_cgs 
    Mdot = 4 * np.pi * r**2 * np.abs(v_rad) * prel.Rsol_cgs**3/prel.tsol_cgs * d

    ax1.plot(r/r_normalizer, d, color = colors_obs[i])#, label = f'{label_obs[i]}')# Observer {label_obs[i]} ({indices_sorted[i]})')
    ax2.plot(r/r_normalizer, v_rad*prel.Rsol_cgs*1e-5/prel.tsol_cgs, color = colors_obs[i], label = f'{label_obs[i]}')
    ax3.plot(r/r_normalizer, Mdot/Medd, color = colors_obs[i], label = f'{label_obs[i]}')
    axT.plot(r/r_normalizer, t, color = colors_obs[i], label = f'{label_obs[i]}')
    axL.plot(r/r_normalizer, L/Ledd, color = colors_obs[i], label = f'{label_obs[i]}')

    for ax in [ax1, ax2, ax3, axT, axL]:
        ax.axvline(rph_mean[i]/r_normalizer, c = colors_obs[i], ls = '--', label = r'$R_{\rm ph}$ ' + f'{label_obs[i]}')
# ax1.plot(r/apo, rho_from_dM, c = 'gray', ls = '--', label = r'$\rho \propto R^{-2}$') #'From dM/dt')
ax1.plot(x_test, y_test2, c = 'gray', ls = 'dashed', label = r'$\rho \propto R^{-2}$')
# ax1.text(4, 1e-14, r'$\propto R^{-2}$', fontsize = 20, color = 'k', rotation = -20)
ax2.axhline(v_esc_kms, c = 'gray', ls = 'dotted', label = r'$v_{\rm esc} (R_p)$')
ax1.plot(x_test, y_test3, c = 'gray', ls = 'dotted', label = r'$\rho \propto R^{-3}$')
ax2.plot(x_test, y_testplus2, c = 'gray', ls = 'dotted', label = r'$v_r \propto R^{2}$')
axT.plot(x_test, y_test23, c = 'gray', ls = 'dashed', label = r'$T \propto R^{-2/3}$')
axL.plot(x_test, 8e-6*y_test23, c = 'gray', ls = 'dashed', label = r'$L \propto R^{-2/3}$')

for ax in [ax1, ax2, ax3, axT, axL]:
    #put the legend if which_obs != 'all_rotate'. Lt it be outside
    # boxleg = ax.get_position()
    # ax.set_position([boxleg.x0, boxleg.y0, boxleg.width * 0.8, boxleg.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)
    # ax.axvline(Rp/apo, c = 'k', ls = '--')
    ax.set_xlabel(r'$R [R_{\rm a}]$' if normalize_by == 'apo' else r'$R [R_{\rm tr}]$')
    # ax.axvline(R_edge/apo,  c = 'k', ls = ':')
    ax.set_xlim(4e-2, 10)
    ax.legend(fontsize = 14)
    ax.loglog()
    ax.tick_params(axis='both', which='minor', size=4)
    ax.tick_params(axis='both', which='major', size=6)
    ax.grid()

ax1.set_ylim(1e-15, 1e-9)
ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
ax2.set_ylim(8e2, 2e4)
ax2.set_ylabel(r'$v_r$ [km/s]')
ax3.set_ylim(1e-1, 1e3)
ax3.set_ylabel(r'$4\pi v_r \rho R^2 \,[\dot{M}_{\rm{Edd}}]$')
axT.set_ylim(1e4, 1e6)
axT.set_ylabel(r'$T$ [K]')
axL.set_ylabel(r'$L [L_{\rm Edd}]$')
axL.set_ylim(2e-2, 2e1)
fig.suptitle(f't = {np.round(tfb,2)}' + r't$_{\rm fb}$', fontsize = 20)
fig.tight_layout()
fig.savefig(f'{abspath}/Figs/next_meeting/den_prof{snap}{which_part}X.png', bbox_inches = 'tight')
figT.savefig(f'{abspath}/Figs/next_meeting/T{snap}{which_part}X.png', bbox_inches = 'tight')
figL.savefig(f'{abspath}/Figs/next_meeting/L{snap}{which_part}X.png', bbox_inches = 'tight')
plt.show()

#%% find eta = mfall(t_fb-t_dyn)/Mwind(tfb)
_, tfb_fall, mfall, \
mwind_pos_Rt, mwind_pos_half_amb, mwind_pos_amb, mwind_pos_50Rt, \
Vwind_pos_Rt, Vwind_pos_half_amb, Vwind_pos_amb, Vwind_pos_50Rt, \
mwind_neg_Rt, mwind_neg_half_amb, mwind_neg_amb, mwind_neg_50Rt, \
Vwind_neg_Rt, Vwind_neg_half_amb, Vwind_neg_amb, Vwind_neg_50Rt = \
    np.loadtxt(f'{abspath}/data/{folder}/wind/Mdot_{check}.csv', 
                delimiter = ',', 
                skiprows=1,  
                unpack=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
for i, r_tr in enumerate(r_tr_mean):
    d = d_mean[i]
    v_rad = v_rad_mean[i]
    d_tr = d[np.argmin(np.abs(r-r_tr))]
    v_rad_tr = v_rad[np.argmin(np.abs(r-r_tr))]
    Mdot = 4 * np.pi * r_tr**2 * d_tr/prel.den_converter * np.abs(v_rad_tr) 
    t_dyn = (r_tr/v_rad_tr)*prel.tsol_cgs/t_fb_days_cgs # you want it in t_fb
    print('t_dyn/t_fb', t_dyn)
    tfb_adjusted = tfb - t_dyn
    find_time = np.argmin(np.abs(tfb_fall-tfb_adjusted))
    eta = np.abs(Mdot/mfall[find_time])
    ax.scatter(r_tr/apo, eta, color = colors_obs[i], s = 80, label = f'{label_obs[i]}')
ax.set_xlabel(r'$R_{\rm tr} [R_{\rm a}]$')
ax.tick_params(axis='both', which='minor', length=5, width=1)
ax.tick_params(axis='both', which='major', length=8, width=1.2)
ax.set_ylabel(r'$\eta = |\dot{M}_{\rm w}/\dot{M}_{\rm fb}|$')
ax.legend(fontsize = 14)
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e-2)
ax.grid()
fig.tight_layout()
plt.show()


# %%
