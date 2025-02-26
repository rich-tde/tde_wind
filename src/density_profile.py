""" Finddensity profile"""
#%%
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
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
import healpy as hp
from scipy.ndimage import uniform_filter1d
import Utilities.prelude as prel
import scipy.integrate as spi
import scipy.optimize as spo
from src.Opacity.linextrapolator import nouveau_rich
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb

##
#
##
def generate_elliptical_observers(num_observers, amb=1, emb=0.5):
    """
    Generates `num_observers` points evenly spaced along the arc length of an ellipse.

    Parameters:
    - num_observers: Number of observer points to generate.
    - amb: Semi-major axis length (a).
    - emb: Eccentricity of the ellipse (e).

    Returns:
    - observers_xyz: (num_observers, 3) array of observer positions in 3D space (x, y, z).
    """
    a = amb  # Semi-major axis
    b = a * np.sqrt(1 - emb**2)  # Compute semi-minor axis

    # Arc length integrand
    def arc_length_integrand(theta):
        return np.sqrt((a * np.sin(theta))**2 + (b * np.cos(theta))**2)

    # Compute total perimeter (arc length from 0 to 2π)
    total_arc_length, _ = spi.quad(arc_length_integrand, 0, 2 * np.pi)

    # Generate evenly spaced arc-length values
    s_values = np.linspace(0, total_arc_length, num_observers, endpoint=False)

    # Function to find theta for a given arc length
    def arc_length_to_theta(target_s):
        return spo.root_scalar(lambda theta: spi.quad(arc_length_integrand, 0, theta)[0] - target_s, 
                               bracket=[0, 2 * np.pi]).root

    # Compute theta values for evenly spaced arc lengths
    theta_values = np.array([arc_length_to_theta(s) for s in s_values])

    # Convert to Cartesian coordinates
    x_values = a * np.cos(theta_values)
    y_values = b * np.sin(theta_values)
    z_values = np.zeros_like(x_values)  # Observers in the xy-plane

    # Stack into (N, 3) format
    observers_xyz = np.vstack((x_values, y_values, z_values)).T
    return observers_xyz

def CouBegel(r, theta, q, gamma=4/3):
    # I'm missing the noramlization
    alpha = (1-q*(gamma-1))/(gamma-1)
    rho = r**(-q) * np.sin(theta)**(2*alpha)
    return rho

def Metzger(r, Rv, q, Me = 1):
    # I'm missing the correct noramlization
    norm = Me*(3-q)/(4*np.pi*Rv**3 * (7-2*q))
    if norm == 0.0:
        print('norm is 0')
    if r < Rv:
        rho = (r/Rv)**(-q)
    else:
        rho = np.exp(-(r-Rv)/Rv)
    return (norm * rho)

theta_test = np.linspace(-np.pi, np.pi, 100)
d_test = CouBegel(1, theta_test, 0.5)
d_test2 = CouBegel(1, theta_test, 1)
d_test3 = CouBegel(1, theta_test, 1.5)
plt.figure()
plt.plot(theta_test, d_test, label = r'R=1, q = 0.5')
plt.plot(theta_test, d_test2, label = r'R=1, q = 1')
plt.plot(theta_test, d_test3, label = r'R=1, q = 1.5')
plt.legend()
plt.title(r'$\rho \propto R^{-q} \sin^{2\alpha}(\theta)$, $\alpha = \frac{1-q(\gamma-1)}{\gamma-1}, \gamma = 4/3$')
#%%
##
# MAIN
##
save = True

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' # '' or 'HiRes'
which_obs = 'elliptical' # 'healpix' or 'elliptical'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snap = 348
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
e_mb = orb.eccentricity(Rstar, mstar, Mbh, beta)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = Rt * beta
x_test = np.arange(1e-1, 20)
y_test3 = 2e-13 * x_test**(-3)
y_test4 = 2e-13 * x_test**(-4)
y_test7 = 1e-10 * x_test**(-7)
r_const = np.array([Rt, 10*Rt, apo])
r_const_label = [r'$R_t$', r'$10R_t$', r'$R_a$']
r_const_colors = ['dodgerblue', 'darkviolet', 'forestgreen']

# Opacity Input (they are ln)
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')

T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5)

#%% MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()

pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
box = np.zeros(6)
#%% Load data -----------------------------------------------------------------
if alice:
    X = np.load(f'{pre}/snap_{snap}/CMx_{snap}.npy')
    Y = np.load(f'{pre}/snap_{snap}/CMy_{snap}.npy')
    Z = np.load(f'{pre}/snap_{snap}/CMz_{snap}.npy')
    T = np.load(f'{pre}/snap_{snap}/T_{snap}.npy')
    Den = np.load(f'{pre}/snap_{snap}/Den_{snap}.npy')
    Rad = np.load(f'{pre}/snap_{snap}/Rad_{snap}.npy')
    Vol = np.load(f'{pre}/snap_{snap}/Vol_{snap}.npy')
    box = np.load(f'{pre}/snap_{snap}/box_{snap}.npy')
else:
    X = np.load(f'{pre}/{snap}/CMx_{snap}.npy')
    Y = np.load(f'{pre}/{snap}/CMy_{snap}.npy')
    Z = np.load(f'{pre}/{snap}/CMz_{snap}.npy')
    T = np.load(f'{pre}/{snap}/T_{snap}.npy')
    Den = np.load(f'{pre}/{snap}/Den_{snap}.npy')
    Rad = np.load(f'{pre}/{snap}/Rad_{snap}.npy')
    Vol = np.load(f'{pre}/{snap}/Vol_{snap}.npy')
    box = np.load(f'{pre}/{snap}/box_{snap}.npy')

denmask = Den > 1e-19
X, Y, Z, T, Den, Rad, Vol = make_slices([X, Y, Z, T, Den, Rad, Vol], denmask)
Rad_den = np.multiply(Rad,Den) # now you have energy density
del Rad   
R = np.sqrt(X**2 + Y**2 + Z**2)    

#%% Observers -----------------------------------------------------------------
if which_obs == 'elliptical':
    observers_xyz = generate_elliptical_observers(num_observers = 50, amb = a_mb, emb = e_mb) # shape: (200, 3)
    observers_xyz = np.array(observers_xyz)
    x_obs, y_obs, z_obs = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
if which_obs == 'healpix':
    observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
    # select only the observers in the orbital plane (will give you a N bool array--> apply to columns)
    # mid = np.abs(observers_xyz[2]) == 0 # you can do that beacuse healpix gives you the observers also in the orbital plane (Z==0)
    # observers_xyz, obs_indices = observers_xyz[:,mid], obs_indices[mid]
    observers_xyz = observers_xyz[:,np.arange(0,192,22)]
    x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
    observers_xyz = np.transpose(observers_xyz) #shape: Nx3

r_obs = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
theta_obs = np.arctan2(y_obs, x_obs)          # Azimuthal angle in radians
phi_obs = np.arccos(z_obs / r_obs)

#%%
xyz = np.array([X, Y, Z]).T
N_ray = 5_000
x_ph = np.zeros(len(observers_xyz))
y_ph = np.zeros(len(observers_xyz))
z_ph = np.zeros(len(observers_xyz))
ph_idx = np.zeros(len(observers_xyz))
r_initial = np.zeros(len(observers_xyz))
idx_obs = np.zeros(len(observers_xyz))
idx_b = np.zeros(len(observers_xyz))
d_r = []
img1, (ax4, ax5, ax6) = plt.subplots(3,1,figsize = (8,15)) # this is to check all observers from Healpix on the orbital plane
with open(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}.txt','w') as file:
    file.write(f'# radii, density profile. Look at den_prof_indices{snap} to know when, starting from inside, you stop \n')
    file.close()
for i in range(len(observers_xyz)):
    # Progress 
    print(f'Obs: {i}', flush=False)
    sys.stdout.flush()

    # Dynamic Box -----------------------------------------------------------------
    mu_x = observers_xyz[i][0] 
    mu_y = observers_xyz[i][1] 
    mu_z = observers_xyz[i][2]

    # Box is for dynamic ray making
    # box gives -x, -y, -z, +x, +y, +z
    if mu_x < 0:
        rmax = box[0] / mu_x
    else:
        rmax = box[3] / mu_x
    if mu_y < 0:
        rmax = min(rmax, box[1] / mu_y)
    else:
        rmax = min(rmax, box[4] / mu_y)
    if mu_z < 0:
        rmax = min(rmax, box[2] / mu_z)
    else:
        rmax = min(rmax, box[5] / mu_z)

    r = np.logspace( -0.25, np.log10(rmax), N_ray)
    x = r*mu_x
    y = r*mu_y
    z = r*mu_z
    r = r*np.sqrt(mu_x**2 + mu_y**2 + mu_z**2) #if you choose your elliptical, r CHANGE !!
    rmax = np.max(r)


    r_initial[i] = rmax #np.max(r)
    xyz2 = np.array([x, y, z]).T
    del x, y, z

    tree = KDTree(xyz, leaf_size=50)
    _, idx = tree.query(xyz2, k=1)
    idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
    d = Den[idx] * prel.den_converter
    t = T[idx]
    ray_x = X[idx]
    ray_y = Y[idx]
    ray_z = Z[idx]
    rad_den = Rad_den[idx]
    volume = Vol[idx]
    # find density at a given R for each observer. Since observers are noramlized, you can directly use r to query.
    r_fortree = r.reshape(-1, 1)
    tree_r = KDTree(r_fortree, leaf_size=50)
    _, idx_r = tree_r.query(r_const.reshape(-1,1), k=1)
    idx_r = [ int(idx_r[i][0]) for i in range(len(idx_r))] # no -1 because we start from 0
    d_r.append(d[idx_r] * prel.den_converter)

    # Interpolate ----------------------------------------------------------
    # sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T,np.log(t),np.log(d),'linear',0)
    # sigma_rossland = [sigma_rossland[0][i] for i in range(N_ray)]
    sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T # needs T for the new RICH extrapol
                                     ,np.log(t), np.log(d),'linear',0)
    sigma_rossland = np.array(sigma_rossland)[0]
    underflow_mask = sigma_rossland != 0.0
    d, t, r, sigma_rossland, ray_x, ray_y, ray_z, rad_den, volume = make_slices([d, t, r, 
                                                               sigma_rossland, 
                                                               ray_x, ray_y, ray_z,
                                                               rad_den, volume], underflow_mask)
    sigma_rossland_eval = np.exp(sigma_rossland) 
    del sigma_rossland
    gc.collect()

    # Optical Depth ---------------------------------------------------------------
    r_fuT = np.flipud(r)#.T)
    kappa_rossland = np.flipud(sigma_rossland_eval) 
    los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r

    # Red -----------------------------------------------------------------------
    # Get 20 unique, nearest neighbors
    xyz3 = np.array([ray_x, ray_y, ray_z]).T
    _, idxnew = tree.query(xyz3, k=20)
    idxnew = np.unique(idxnew).T
    dx = 0.5 * volume**(1/3) # Cell radius #the constant should be 0.62

    # Get the Grads
    f_inter_input = np.array([X[idxnew], Y[idxnew], Z[idxnew]]).T

    gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ ray_x+dx, ray_y, ray_z]).T )
    gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ ray_x-dx, ray_y, ray_z]).T )
    gradx = (gradx_p - gradx_m)/ (2*dx)
    gradx = np.nan_to_num(gradx, nan =  0)
    del gradx_p, gradx_m

    grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ ray_x, ray_y+dx, ray_z]).T )
    grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ ray_x, ray_y-dx, ray_z]).T )
    grady = (grady_p - grady_m)/ (2*dx)
    grady = np.nan_to_num(grady, nan =  0)
    del grady_p, grady_m

    gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ ray_x, ray_y, ray_z+dx]).T )
    gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ ray_x, ray_y, ray_z-dx]).T )
    gradz = (gradz_p - gradz_m)/ (2*dx)
    gradz = np.nan_to_num(gradz, nan =  0)
    del gradz_p, gradz_m

    grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
    gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
    del gradx, grady, gradz
    gc.collect()

    R_lamda = grad / ( prel.Rsol_cgs * sigma_rossland_eval* rad_den)
    R_lamda[R_lamda < 1e-10] = 1e-10
    fld_factor = (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
    smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) # i have remov
    photosphere = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0]
    
    ph_idx[i] = idx[photosphere]
    x_ph[i], y_ph[i], z_ph[i] = ray_x[photosphere], ray_y[photosphere], ray_z[photosphere]

    del smoothed_flux, R_lamda, fld_factor, rad_den
    gc.collect()

    # plot from inside till photosphere
    ax4.plot(r[:photosphere+1]/apo, d[:photosphere+1], c = cm.jet(i / len(observers_xyz)), label = f'{i}')
    ax5.plot(r[:photosphere+1]/apo, t[:photosphere+1], c = cm.jet(i / len(observers_xyz)))
    ax6.plot(r[:photosphere+1]/apo, los[:photosphere+1], c = cm.jet(i / len(observers_xyz)), label = f'{i}')
    idx_obs[i] = i
    idx_b[i] = photosphere
    with open(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}.txt','a') as file:
        file.write(f' '.join(map(str, r)) + '\n')
        file.write(f' '.join(map(str, d)) + '\n')
        file.close()

ax4.axhspan(np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool)), alpha=0.2, color='gray')
ax5.axhspan(np.min(np.exp(T_cool)), np.max(np.exp(T_cool)), alpha=0.2, color='gray')
ax6.axhline(2/3, c = 'k', ls = '--', label = r'$\tau = 2/3$')
ax4.set_ylabel(r'$\rho$ [g/cm$^3]$')
ax5.set_ylabel(r'T [K]')
ax6.set_ylabel(r'$\tau$')
ax6.set_xlabel(r'R [R$_a]$')
# ax5.set_ylim(3e-1, 20)
ax4.set_ylim(2e-17, 1e-5)#2e-9)

# rph_h = np.sqrt((x_ph[0])**2 + (y_ph[0])**2 + (z_ph[0])**2)
for ax in [ax4, ax5, ax6]:
    ax.loglog()
    # ax.set_xlim(3e-1, 20)
    ax.axvline(Rt/apo, ls = 'dotted', c = 'k', label = r'R$_t$')
    ax.grid()
ax4.legend(loc = 'upper right')
img1.tight_layout()
np.save(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof_indices{snap}.npy', [idx_obs, idx_b])
# data_to_save = [r_initial, x_ph, y_ph, z_ph, ph_idx] # 5x16x2 where 16=len(obsevers OrbPl)

#%%
d_r_trans = np.transpose(d_r)
img, ax1 = plt.subplots(1,1,figsize = (7, 7)) # this is to check all observers from Healpix on the orbital plane
for i, r in enumerate(r_const):
    ax1.scatter(theta_obs, d_r_trans[i], c= r_const_colors[i], label = f'R = {r_const_label[i]}')
    for j, theta in enumerate(theta_obs):
        ax1.scatter(theta, CouBegel(r*prel.Rsol_cgs, theta, .8, 4/3), c= r_const_colors[i], marker = 'x')
ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
ax1.set_xlabel(r'$\theta$ [rad]')
ax1.set_yscale('log')
ax1.legend(loc='upper right')
ax1.set_ylim(2e-15, 2e-5)#2e-9)
#%% 
eng.exit()

#%% Check density profile from Rin == Rp till Rout= Rph (but should be the colorsphere at tau=1)
idx_load = np.load(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof_indices{snap}.npy')
i_all, b_all = idx_load[0].astype(int), idx_load[1].astype(int)
data_loaded = np.loadtxt(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}.txt')
r_all, d_all = data_loaded[0::2], data_loaded[1::2]
img, (ax1, ax2) = plt.subplots(1,2,figsize = (15, 5)) # this is to check all observers from Healpix on the orbital plane
for j, i in enumerate(i_all):
    b = b_all[j]
    r = r_all[j]
    d = d_all[j]
    idx_Rp = np.argmin(np.abs(r - Rp))
    den_Metzger3 = np.zeros(b+1-idx_Rp)
    den_Metzger7 = np.zeros(b+1-idx_Rp)
    for k in range(idx_Rp, b+1):
        den_Metzger3[k] = Metzger(r[k]*prel.Rsol_cgs, Rv=apo*prel.Rsol_cgs, q = 2.8, Me = 0.2*mstar*prel.Msol_cgs)
        den_Metzger7[k] = Metzger(r[k]*prel.Rsol_cgs, Rv=apo*prel.Rsol_cgs, q = 7, Me = 0.2*mstar*prel.Msol_cgs)
    if y_obs[j] < 0:
        ax1.plot(r[idx_Rp:b+1]/apo, d[idx_Rp:b+1], c = cm.jet(j / len(i_all)))#, label = f'{j}')
        ax1.plot(r[idx_Rp:b+1]/apo, den_Metzger7, label = r'Metzger $\xi = 3, R_v=R_{\rm circ}$' if j == np.argmin(y_obs-20) else None) # dumb way to have the label once
    else:
        ax2.plot(r[idx_Rp:b+1]/apo, d[idx_Rp:b+1], c = cm.jet(j / len(i_all)))#, label = f'{j}')
        ax2.plot(r[idx_Rp:b+1]/apo, den_Metzger3, label = r'Metzger $\xi = 3, R_v=R_{\rm circ}$' if j == np.argmin(np.abs(y_obs-20)) else None) # dumb way to have the label once
ax1.set_title(r'Obs $y<0$')
ax2.set_title(r'Obs $y>0$')
ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
ax2.plot(x_test, y_test3, c = 'gray', ls = '--', label = r'$\rho \propto R^{-3}$')
ax1.plot(x_test, y_test4, c = 'gray', ls = '-.', label = r'$\rho \propto R^{-4}$')
ax1.plot(x_test, y_test7, c = 'gray', ls = '--', label = r'$\rho \propto R^{-7}$')

for ax in [ax1, ax2]:
    ax.axhspan(np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool)), alpha=0.2, color='gray')
    ax.set_xlabel(r'R [R$_a]$')
    ax.set_ylim(2e-14, 2e-4)#2e-9)
    ax.loglog()
    ax.grid()
    ax.legend(loc='upper right')
img.tight_layout()
img.savefig(f'{abspath}/Figs/Test/photosphere/{snap}/OrbPl/{snap}_raysHzoom.png', bbox_inches='tight')


# %%
plt.figure()
plt.scatter(x_obs, y_obs, c= np.arange(len(x_obs)), cmap = 'jet', edgecolors='k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Observers')
# %%
