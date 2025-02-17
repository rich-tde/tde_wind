""" Find photosphere following FLD curve Elad's script. 
Just for the orbital plane, no Helapix."""
#%%
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'
    import matplotlib.pyplot as plt

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
from Utilities.selectors_for_snap import select_snap, select_prefix
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

#%% observers 
observers_xyz_mine = generate_elliptical_observers(num_observers = 200, amb = a_mb, emb = e_mb) # shape: (200, 3)
observers_xyz_mine = np.array(observers_xyz_mine)
x_mine, y_mine, z_mine = observers_xyz_mine[:, 0], observers_xyz_mine[:, 1], observers_xyz_mine[:, 2]
r_mine = np.sqrt(x_mine**2 + y_mine**2 + z_mine**2)

# Opacity Input
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
longitude_moll = theta_heal             
latitude_moll = np.pi / 2 - phi_heal
indecesorbital = np.concatenate(np.where(latitude_moll==0))
long_orb, lat_orb = longitude_moll[indecesorbital], latitude_moll[indecesorbital]
# Plot in 2D using a Mollweide projection
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
img = ax.scatter(long_orb, lat_orb, s=40, c= np.arange(len(long_orb)))
plt.colorbar(img, ax=ax, label='Observer Number')
ax.grid(True)
ax.set_xticks(np.radians(np.linspace(-180, 180, 9)))
ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/Test/photosphere/348/observersOrbPl.png')
plt.show()

#%% Tree ----------------------------------------------------------------------
observers_xyz = observers_xyz_mine #np.array([x_mine, y_mine, z_mine]).T

xyz = np.array([X, Y, Z]).T
N_ray = 5_000

# Dynamic Box -----------------------------------------------------------------
reds = np.zeros(len(observers_xyz))
## just to check photosphere
x_ph = np.zeros(len(observers_xyz))
y_ph = np.zeros(len(observers_xyz))
z_ph = np.zeros(len(observers_xyz))
ph_idx = np.zeros(len(observers_xyz))
fluxes = np.zeros(len(observers_xyz))
r_initial = np.zeros(len(observers_xyz))
for i in range(len(observers_xyz)):
    # Progress 
    print(f'Obs: {i}', flush=False)
    sys.stdout.flush()

    mu_x = observers_xyz[i][0] # mu_x = x_heal[i]
    mu_y = observers_xyz[i][1] # mu_y = y_heal[i]
    mu_z = observers_xyz[i][2] # mu_z = z_heal[i]
    # mu_x_mine = x_mine[i]
    # mu_y_mine = y_mine[i]
    # mu_z_mine = z_mine[i]

    # Box is for dynamic ray making
    # box gives -x, -y, -z, +x, +y, +z
    if mu_x < 0:
        rmax = box[0] / mu_x
        # rmax_mine = box[0] / mu_x_mine
    else:
        rmax = box[3] / mu_x
        # rmax_mine = box[3] / mu_x_mine
    if mu_y < 0:
        rmax = min(rmax, box[1] / mu_y)
        # # rmax_mine = min(rmax_mine, box[1] / mu_y_mine)
    else:
        rmax = min(rmax, box[4] / mu_y)
        # # rmax_mine = min(rmax_mine, box[4] / mu_y_mine)
    if mu_z < 0:
        rmax = min(rmax, box[2] / mu_z)
        # # rmax_mine = min(rmax_mine, box[2] / mu_z_mine)
    else:
        rmax = min(rmax, box[5] / mu_z)
        # # rmax_mine = min(rmax_mine, box[5] / mu_z_mine)

    # we want rmax = rmax_mine*Robsmax_mine where Robs = sqrt(mu_ x_mine**2 + mu_ y_mine**2 + mu_ z_mine**2)
    # rmax_new = rmax_mine * np.sqrt(mu_x_mine**2 + mu_y_mine**2 + mu_z_mine**2)
    # rs_max = [rmax, rmax_new]
    # label_rs = ['R_Healp', 'R_mine']
    # linestyle_rs = ['-', 'solid']
    # marker_rs = ['o', '*']
    
    # img, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 15))
    # for j, rmax_chosen in enumerate(rs_max):
    r = np.logspace( -0.25, np.log10(rmax), N_ray)
    # r_initial[i] = rmax_mine
    alpha = (r[1] - r[0]) / (0.5 * ( r[0] + r[1]))
    dr = alpha * r

    x = r*mu_x
    y = r*mu_y
    z = r*mu_z
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
    xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
    _, idxnew = tree.query(xyz3, k=20)
    idxnew = np.unique(idxnew).T
    dx = 0.5 * Vol[idx]**(1/3) # Cell radius #the constant should be 0.62

    # Get the Grads
    # sphere and get the gradient on them. Is it neccecery to re-interpolate?
    # scattered interpolant returns a function
    # griddata DEMANDS that you pass it the values you want to eval at
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
    reds[i] = Lphoto
    # just to check photosphere
    ph_idx[i] = idx[b]
    print(f'Photosphere: {ph_idx[i]}')
    x_ph[i], y_ph[i], z_ph[i] = X[idx][b], Y[idx][b], Z[idx][b]
    fluxes[i] = Lphoto / (4*np.pi*(r[b]*prel.Rsol_cgs)**2)
    
    del smoothed_flux, R_lamda, fld_factor, EEr
    gc.collect()
    Lphoto_snap = np.mean(reds)

    # ax1.plot(r/apo,d, c = 'g', linestyle = linestyle_rs[j])
    # ax2.plot(r/apo,t, c ='r',  linestyle = linestyle_rs[j])
    # ax3.plot(r/apo,los, c ='b', linestyle = linestyle_rs[j])
    # for ax in [ax1, ax2, ax3]:
    #     ax.axvline(r[b]/apo, c = 'k', linestyle = linestyle_rs[j], label = f'Photosphere {label_rs[j]}')
    # ax4.scatter(x_ph/apo, y_ph/apo, marker = marker_rs[j])
  
    # ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
    # ax2.set_ylabel(r'T [K]')
    # ax3.set_xlabel(r'R [R$_a]$')
    # ax3.set_ylabel(r'$\tau$')
    # ax3.axhline(2/3, c = 'r', ls = '--')
    # ax3.legend()
    # ax4.set_xlim(-9, 3)
    # ax4.set_ylim(-5,3)
    # for ax in [ax1, ax2, ax3]:
    #     ax.loglog()
    # plt.suptitle(f'Observer: {i}', fontsize=16)
    # plt.tight_layout()
    # plt.savefig(f'{abspath}/Figs/Test/photosphere/348/348_ray{i}.png')
    # plt.close()
#%%
plt.figure(figsize = (12,5))
plt.scatter(x_ph/apo, y_ph/apo, c = 'r', cmap = 'jet')  
plt.xlim(-9, 3)
plt.ylim(-5,3)

#%% Save red of the single snap
save = True
if save:
    pre_saving = f'{abspath}data/{folder}/testRph'
    # save observers position
    np.savetxt(f'{pre_saving}/small/{check}_observers200LOCAL_{snap}.txt', [r_initial, observers_xyz[:,0], observers_xyz[:,1], observers_xyz[:,2], ph_idx])
eng.exit()

# %%
alicedata = np.loadtxt(f'{abspath}data/{folder}/testRph/small/{check}_photo{snap}_200AGAIN.txt')
# alicedata = np.loadtxt(f'{abspath}data/{folder}/photo/{check}_photo{snap}_OrbPl200.txt')
xalice, yalice, zalice = alicedata[0], alicedata[1], alicedata[2]
plt.figure(figsize = (12,5))
plt.scatter(x_ph/apo, y_ph/apo, c = 'r', s=20, label = 'local')
plt.scatter(xalice/apo, yalice/apo, c = 'b', s=10, label = 'Alice')
plt.xlim(-9, 3)
plt.ylim(-5,3)
plt.legend()

# %%
