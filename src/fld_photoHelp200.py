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

def generate_orbital_observers(num_observers, radius=1):
    """
    Generates `num_observers` points in a circular distribution in the orbital plane (xy-plane). 
    They are evenly spaced in angle.
    """
    angles = np.linspace(0, 2 * np.pi, num_observers, endpoint=False)  # Evenly spaced angles
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x)  # All points in the xy-plane

    observers_xyz = np.vstack((x, y, z)).T  # Stack into (N, 3) format
    return observers_xyz

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
x_test = np.arange(1e-1, 20)
y_test3 = 1e-13 * x_test**(-3)
y_test4 = 1e-13 * x_test**(-4)
y_test7 = 1e-10 * x_test**(-7)

#%% observers 
observers_xyz_mine = generate_elliptical_observers(num_observers = 200, amb = a_mb, emb = e_mb) # shape: (200, 3)
observers_xyz_mine = np.array(observers_xyz_mine)
x_mine, y_mine, z_mine = observers_xyz_mine[:, 0], observers_xyz_mine[:, 1], observers_xyz_mine[:, 2]
r_mine = np.sqrt(x_mine**2 + y_mine**2 + z_mine**2)

# Opacity Input (they are ln)
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')

T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland)

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
theta_mine = np.arctan2(y_mine, x_mine)          # Azimuthal angle in radians
phi_mine = np.arccos(z_mine / r_mine)
points_toquery = np.transpose([theta_heal, phi_heal])
tree = KDTree(np.transpose([theta_mine, phi_mine]))
_, idx = tree.query(points_toquery, k=1)
x_mine, y_mine, z_mine = np.concatenate(x_mine[idx]), np.concatenate(y_mine[idx]), np.concatenate(z_mine[idx])
r_mine = np.sqrt(x_mine**2 + y_mine**2 + z_mine**2)

#%% Plot selected observers
# Healpix
longitude_moll_h = theta_heal             
latitude_moll_h = np.pi / 2 - phi_heal
# Mine
longitude_moll_m = theta_mine            
latitude_moll_m = np.pi / 2 - phi_mine
longitude_moll_m, latitude_moll_m = longitude_moll_m[idx], latitude_moll_m[idx]
# Plot in 2D using a Mollweide projection
print(len(longitude_moll_m))
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
img = ax.scatter(longitude_moll_h, latitude_moll_h, s=80, c= np.arange(len(longitude_moll_h)), cmap='jet', edgecolors='k')
img1 = ax.scatter(longitude_moll_m, latitude_moll_m, s=40, marker = '*', c= np.arange(len(longitude_moll_m)), edgecolors='k', cmap='jet')
plt.colorbar(img, ax=ax, label='Observer Number')
ax.grid(True)
ax.set_xticks(np.radians(np.linspace(-180, 180, 9)))
ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/Test/photosphere/{snap}/observersOrbPlMoll.png')
plt.show()

#%% Tree ----------------------------------------------------------------------
# observers_xyz = observers_xyz_mine 

xyz = np.array([X, Y, Z]).T
N_ray = 5_000

# Dynamic Box -----------------------------------------------------------------
x_ph = []
y_ph = []
z_ph = []
ph_idx = []
r_initial = []
img1, (ax4, ax5, ax6) = plt.subplots(3,1,figsize = (8,15)) # this is to check all observers from Healpix on the orbital plane
for i in range(len(observers_xyz)):
    # if you want to check a specific observer
    if int(i) >=10:
        continue
    # Progress 
    print(f'Obs: {i}', flush=False)
    sys.stdout.flush()

    mu_x = observers_xyz[i][0] # mu_x = x_heal[i]
    mu_y = observers_xyz[i][1] # mu_y = y_heal[i]
    mu_z = observers_xyz[i][2] # mu_z = z_heal[i]
    mu_x_mine = x_mine[i]
    mu_y_mine = y_mine[i]
    mu_z_mine = z_mine[i]

    # Box is for dynamic ray making
    # box gives -x, -y, -z, +x, +y, +z
    if mu_x < 0:
        rmax = box[0] / mu_x
        rmax_mine = box[0] / mu_x_mine
    else:
        rmax = box[3] / mu_x
        rmax_mine = box[3] / mu_x_mine
    if mu_y < 0:
        rmax = min(rmax, box[1] / mu_y)
        rmax_mine = min(rmax_mine, box[1] / mu_y_mine)
    else:
        rmax = min(rmax, box[4] / mu_y)
        rmax_mine = min(rmax_mine, box[4] / mu_y_mine)
    if mu_z < 0:
        rmax = min(rmax, box[2] / mu_z)
        rmax_mine = min(rmax_mine, box[2] / mu_z_mine)
    else:
        rmax = min(rmax, box[5] / mu_z)
        rmax_mine = min(rmax_mine, box[5] / mu_z_mine)

    # we want rmax = rmax_mine*Robsmax_mine where Robs = sqrt(mu_ x_mine**2 + mu_ y_mine**2 + mu_ z_mine**2)
    # rmax_new = rmax_mine * np.sqrt(mu_x_mine**2 + mu_y_mine**2 + mu_z_mine**2)
    rs_max = [rmax]#, rmax_mine]
    label_rs = ['R_Healp', 'R_mine']
    linestyle_rs = ['--', 'solid']
    marker_rs = ['o', '*']

    img, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10)) # this is to check Healpix with my sample
    x_ph_r = np.zeros(len(rs_max))
    y_ph_r = np.zeros(len(rs_max))
    z_ph_r = np.zeros(len(rs_max))
    ph_idx_r = np.zeros(len(rs_max))
    r_initial_r = np.zeros(len(rs_max))
    for j, rmax_chosen in enumerate(rs_max):
        r = np.logspace( -0.25, np.log10(rmax_chosen), N_ray)
        if j == 0:
            x = r*mu_x
            y = r*mu_y
            z = r*mu_z
        else:
            x = r*mu_x_mine
            y = r*mu_y_mine
            z = r*mu_z_mine
            r = r*np.sqrt(mu_x_mine**2 + mu_y_mine**2 + mu_z_mine**2) #CHANGE r!!!
        
        r_initial_r[j] = np.max(r)
        xyz2 = np.array([x, y, z]).T
        del x, y, z

        tree = KDTree(xyz, leaf_size=50)
        _, idx = tree.query(xyz2, k=1)
        idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
        d = Den[idx] * prel.den_converter
        t = T[idx]

        # Interpolate ----------------------------------------------------------
        sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T,np.log(t),np.log(d),'linear',0)
        sigma_rossland = [sigma_rossland[0][i] for i in range(N_ray)]
        sigma_rossland_eval = np.exp(sigma_rossland) 
        del sigma_rossland
        gc.collect()

        # Optical Depth ---------------------------------------------------------------
        r_fuT = np.flipud(r.T)
        kappa_rossland = np.flipud(sigma_rossland_eval) 
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r

        # Red -----------------------------------------------------------------------
        # Get 20 unique, nearest neighbors
        xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
        _, idxnew = tree.query(xyz3, k=20)
        idxnew = np.unique(idxnew)
        dx = 0.5 * Vol[idx]**(1/3) # Cell radius #the constant should be 0.62

        # Get the Grads
        f_inter_input = np.array([ X[idxnew], Y[idxnew], Z[idxnew] ]).T

        gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx]+dx, Y[idx], Z[idx]]).T )
        gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx]-dx, Y[idx], Z[idx]]).T )
        gradx = (gradx_p - gradx_m)/ (2*dx)
        del gradx_p, gradx_m

        gradx = np.nan_to_num(gradx, nan =  0)
        grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx]+dx, Z[idx]]).T )
        grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx]-dx, Z[idx]]).T )
        grady = (grady_p - grady_m)/ (2*dx)
        del grady_p, grady_m

        grady = np.nan_to_num(grady, nan =  0)

        gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx], Z[idx]+dx]).T )
        gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx], Z[idx]-dx]).T )
        # some nans here
        gradz_m = np.nan_to_num(gradz_m, nan =  0)
        gradz = (gradz_p - gradz_m)/ (2*dx)
        del gradz_p, gradz_m

        grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
        gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
        del gradx, grady, gradz
        gc.collect()

        R_lamda = grad / ( prel.Rsol_cgs * sigma_rossland_eval* Rad_den[idx])
        R_lamda[R_lamda < 1e-10] = 1e-10
        fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) # i have remov

        try:
            b = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0] 
        except IndexError:
            print('No b found, observer ', i)
            b = 3117 # elad_b = 3117
        Lphoto2 = 4*np.pi*prel.c_cgs*smoothed_flux[b] * prel.Msol_cgs / (prel.tsol_cgs**2)
        EEr = Rad_den[idx]
        if Lphoto2 < 0:
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives
        max_length = 4*np.pi*prel.c_cgs*EEr[b]*r[b]**2 * prel.Msol_cgs * prel.Rsol_cgs / (prel.tsol_cgs**2) #the conversion is for Erad: energy*r^2/lenght^3 [in SI would be kg m^2/s^2 * m^2 * 1/m^3]
        Lphoto = np.min( [Lphoto2, max_length])
        ph_idx_r[j] = idx[b]
        x_ph_r[j], y_ph_r[j], z_ph_r[j] = X[idx][b], Y[idx][b], Z[idx][b]

        del smoothed_flux, R_lamda, fld_factor, EEr
        gc.collect()

        ax1.plot(r/apo,d, c = 'k', linestyle = linestyle_rs[j])
        ax1.axhspan(np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool)), alpha=0.2, color='gray')
        ax2.plot(r/apo,t, c ='k',  linestyle = linestyle_rs[j])
        ax2.axhspan(np.min(np.exp(T_cool)), np.max(np.exp(T_cool)), alpha=0.2, color='gray')
        ax3.plot(r/apo,los, c ='k', linestyle = linestyle_rs[j])
        for ax in [ax1, ax2, ax3]:
            ax.axvline(r[b]/apo, c = 'b', linestyle = linestyle_rs[j], label = f'Photosphere {label_rs[j]}')
            ax.axvline(r_initial_r[j]/apo, c = 'r', linestyle = linestyle_rs[j], label = f'R_initial {label_rs[j]}')
        
        if j == 0:
            ax4.plot(r/apo, d, c = cm.jet(i / len(observers_xyz)), label = f'{i}')
            ax5.plot(r/apo, t, c = cm.jet(i / len(observers_xyz)))
            ax6.plot(r/apo, los, c = cm.jet(i / len(observers_xyz)), label = f'{i}')

    ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
    ax2.set_ylabel(r'T [K]')
    ax3.set_ylabel(r'$\tau$')
    ax3.set_xlabel(r'R [R$_a]$')
    ax3.axhline(2/3, c = 'k', ls = '--')
    ax3.legend()
    for ax in [ax1, ax2, ax3]:
        ax.loglog()
        ax.set_xlim(1e-1, 1e2)
    img.suptitle(f'Observer: {i}', fontsize=16)
    img.tight_layout()
    # img.savefig(f'{abspath}/Figs/Test/photosphere/{snap}/OrbPl/{snap}_ray{i}.png')
    # save data for photosphere
    r_initial.append(r_initial_r)
    x_ph.append(x_ph_r)
    y_ph.append(y_ph_r)
    z_ph.append(z_ph_r)
    ph_idx.append(ph_idx_r)

ax4.axhspan(np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool)), alpha=0.2, color='gray')
ax5.axhspan(np.min(np.exp(T_cool)), np.max(np.exp(T_cool)), alpha=0.2, color='gray')
ax6.axhline(2/3, c = 'k', ls = '--', label = r'$\tau = 2/3$')
ax4.set_ylabel(r'$\rho$ [g/cm$^3]$')
ax5.set_ylabel(r'T [K]')
ax6.set_ylabel(r'$\tau$')
ax6.set_xlabel(r'R [R$_a]$')
# ax5.set_ylim(3e-1, 20)
ax4.set_ylim(2e-17, 1e-5)#2e-9)
# ax4.plot(x_test, y_test3, c = 'gray', ls = '--', label = r'$\rho \propto R^{-3}$')
ax4.plot(x_test, y_test4, c = 'gray', ls = '-.', label = r'$\rho \propto R^{-4}$')
ax4.plot(x_test, y_test7, c = 'gray', ls = '--', label = r'$\rho \propto R^{-7}$')

# rph_h = np.sqrt((x_ph[0])**2 + (y_ph[0])**2 + (z_ph[0])**2)
for ax in [ax4, ax5, ax6]:
    ax.loglog()
    ax.set_xlim(3e-1, 20)#(1e-3, 1e2)
    # ax.axvline(Rt/apo, ls = 'dotted', c = 'k', label = r'R$_t$')
    ax.grid()
    # ax.axvline(np.median(rph_h)/apo, ls = 'dashed', c = 'b', label = r'median $R_{ph}$')
ax4.legend(loc = 'upper right')
img1.suptitle('Healpix Observers', fontsize=16)
img1.tight_layout()
img1.savefig(f'{abspath}/Figs/Test/photosphere/{snap}/OrbPl/{snap}_raysHzoom_up10.png', bbox_inches='tight')
# data_to_save = [r_initial, x_ph, y_ph, z_ph, ph_idx] # 5x16x2 where 16=len(obsevers OrbPl)
#%% Save red of the single snap
if save:
    pre_saving = f'{abspath}/data/{folder}/testRph'
    # save observers position
    np.save(f'{pre_saving}/{check}{snap}_compareHeal200.txt', [r_initial, x_ph, y_ph, z_ph, ph_idx])
eng.exit()

#%%
pre_saving = f'{abspath}/data/{folder}/testRph'
data_loaded = np.load(f'{pre_saving}/{check}{snap}_compareHeal200.txt.npy')#, allow_pickle=True)
r_initial, x_ph, y_ph, z_ph, ph_idx = data_loaded[0], data_loaded[1], data_loaded[2], data_loaded[3], data_loaded[4]
r_initial_h, x_ph_h, y_ph_h, z_ph_h = r_initial[:, 0], x_ph[:, 0], y_ph[:, 0], z_ph[:, 0]
r_ph_h = np.sqrt(x_ph_h**2 + y_ph_h**2 + z_ph_h**2)
r_initial_m, x_ph_m, y_ph_m, z_ph_m = r_initial[:, 1], x_ph[:, 1], y_ph[:, 1], z_ph[:, 1]
r_ph_m = np.sqrt(x_ph_m**2 + y_ph_m**2 + z_ph_m**2)
mid = np.abs(Z) < Vol**(1/3)
X_mid, Y_mid, Z_mid, Den_mid, Vol_mid = make_slices([X, Y, Z, Den, Vol], mid)

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
ax1.scatter(r_initial_h*apo, r_ph_h/apo, s = 400, c = np.arange(len(r_ph_h)), label = 'Healpix', marker = 'o', cmap = 'jet', edgecolors = 'k')
# ax1.scatter(r_initial_m*apo, r_ph_m/apo, s = 500, c = np.arange(len(r_ph_m)), label = 'Mine', marker = '*', cmap = 'jet', edgecolors = 'k')
ax1.set_xlabel(r'R$_{initial}$ [R$_a$]', fontsize = 27)
ax1.set_ylabel(r'R$_{ph}$ [R$_a$]', fontsize = 27)
ax1.set_yscale('log')
ax1.grid()
ax1.tick_params(axis='y', which='major', width = 1.5, length = 8)
ax1.tick_params(axis='y', which='minor', width = 1.5, length = 5)
# ax1.legend(fontsize = 27)

ax2.scatter(X_mid/apo, Y_mid/apo, s = 10, c = Den_mid, cmap = 'gray', norm = colors.LogNorm(vmin=1e-13, vmax=1e-4))
img = ax2.scatter(x_ph_h/apo, y_ph_h/apo, c = np.arange(len(x_ph_h)), s = 400, marker = 'o', cmap = 'jet', edgecolors = 'k', label = 'Healpix')  
# img = plt.scatter(x_ph_m/apo, y_ph_m/apo, c = np.arange(len(x_ph_m)), s = 500, marker = '*', cmap = 'jet', edgecolors = 'k',label = 'Mine')
cbar = plt.colorbar(img, orientation='horizontal', pad = 0.15)
cbar.set_label('Observer Number')
ax2.set_xlim(-9,2.5)
ax2.set_ylim(-5,3)
ax2.set_xlabel(r'X [R$_a$]', fontsize = 27)
ax2.set_ylabel(r'Y [R$_a$]', fontsize = 27)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/Test/photosphere/{snap}/OrbPl/{snap}_DenHeal.png', bbox_inches='tight')

# %%
plt.figure(figsize=(15, 7))
img = plt.scatter(X_mid/apo, Y_mid/apo, c=(Vol_mid**(1/3))/apo, cmap = 'viridis', s = 20, norm = colors.LogNorm(vmin=5e-4, vmax=5e-1))
cbar = plt.colorbar(img)
cbar.set_label(r'$R_{\rm cell} [R_{\rm a}]$', fontsize = 20)
plt.scatter(x_ph_h/apo, y_ph_h/apo, facecolor = 'none', s = 40, marker = 'o', cmap = 'jet', edgecolors = 'r')  
plt.ylim(-5, 3)
plt.xlim(-9, 2.5)
plt.xlabel(r'$X [R_{\rm a}]$', fontsize = 20)
plt.ylabel(r'$Y [R_{\rm a}]$', fontsize = 20)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/Test/photosphere/{snap}/OrbPl/{snap}_cell.png', bbox_inches='tight')
