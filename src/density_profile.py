""" Find photosphere following FLD curve Elad's script. 
Compare Healpix and my sample as Rinitial and Rph at the orbital plane.
Check optical depth and density along rays for Healpix sample at the orbital plane"""
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
# MAIN
##
save = False

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' # '' or 'HiRes'

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
observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
# select only the observers in the orbital plane (will give you a N bool array--> apply to columns)
mid = np.abs(observers_xyz[2]) == 0 # you can do that beacuse healpix gives you the observers also in the orbital plane (Z==0)
observers_xyz = observers_xyz[:,mid]
x_heal, y_heal, z_heal = observers_xyz[0], observers_xyz[1], observers_xyz[2]
r_heal = np.sqrt(x_heal**2 + y_heal**2 + z_heal**2)   
observers_xyz = np.transpose(observers_xyz) #shape: Nx3
cross_dot = np.matmul(observers_xyz,  observers_xyz.T)
cross_dot[cross_dot<0] = 0
cross_dot *= 4/len(observers_xyz)

#%% find the corresponding mine observer for each healpix observer
theta_heal = np.arctan2(y_heal, x_heal)          # Azimuthal angle in radians
phi_heal = np.arccos(z_heal / r_heal)


#%% Tree ----------------------------------------------------------------------
xyz = np.array([X, Y, Z]).T
N_ray = 5_000

# Dynamic Box -----------------------------------------------------------------
x_ph = np.zeros(len(observers_xyz))
y_ph = np.zeros(len(observers_xyz))
z_ph = np.zeros(len(observers_xyz))
ph_idx = np.zeros(len(observers_xyz))
r_initial = np.zeros(len(observers_xyz))
idx_obs = np.zeros(len(observers_xyz))
idx_b = np.zeros(len(observers_xyz))
img1, (ax4, ax5, ax6) = plt.subplots(3,1,figsize = (8,15)) # this is to check all observers from Healpix on the orbital plane
with open(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}.txt','w') as file:
    file.write(f'# radii, density profile. Look at den_prof_indices{snap} to know when, starting from inside, you stop \n')
    file.close()
for i in range(len(observers_xyz)):
    # Progress 
    print(f'Obs: {i}', flush=False)
    sys.stdout.flush()

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
    fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
    smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) # i have remov
    photosphere = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0]
    
    Lphoto2 = 4*np.pi*prel.c_cgs*smoothed_flux[photosphere] * prel.Msol_cgs / (prel.tsol_cgs**2)
    if Lphoto2 < 0:
        Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives
    max_length = 4*np.pi*prel.c_cgs*rad_den[photosphere]*r[photosphere]**2 * prel.Msol_cgs * prel.Rsol_cgs / (prel.tsol_cgs**2) #the conversion is for Erad: energy*r^2/lenght^3 [in SI would be kg m^2/s^2 * m^2 * 1/m^3]
    Lphoto = np.min( [Lphoto2, max_length])
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
    if int(i)<=10:
        ax1.plot(r[idx_Rp:b+1]/apo, d[idx_Rp:b+1], c = cm.jet(j / len(i_all)), label = f'{j}')
    else:
        ax2.plot(r[idx_Rp:b+1]/apo, d[idx_Rp:b+1], c = cm.jet(j / len(i_all)), label = f'{j}')
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
