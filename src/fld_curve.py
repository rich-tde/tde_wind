""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
import sys
sys.path.append('/Users/paolamartire/shocks')
# import resource
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
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import first_rich_extrap, linear_rich
from scipy.ndimage import uniform_filter1d

import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import matplotlib.pyplot as plt
import src.orbits as orb

#%% Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'LowResNewAMR' # 

## Snapshots stuff
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
print('we are in: ', pre)

#%% Opacities: load and interpolate ----------------------------------------------------------------
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
if check in ['LowRes', '', 'HiRes', 'LowResNewAMR', 'LowResNewAMRRemoveCenter']:
    T_cool2, Rho_cool2, rossland2 = first_rich_extrap(T_cool, Rho_cool, rossland, what = 'scattering_limit', slope_length = 7, highT_slope = 0)
if check in ['LowResOpacityNew', 'OpacityNew', 'OpacityNewNewAMR']:
    T_cool2, Rho_cool2, rossland2 = linear_rich(T_cool, Rho_cool, rossland, what = 'scattering_limit', highT_slope = 0)
        
N_ray = 5_000
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

# MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()
Lphoto_all = np.zeros(len(snaps))
for idx_s, snap in enumerate(snaps):
    print('\n Snapshot: ', snap, '\n')
    box = np.zeros(6)
    # Load data -----------------------------------------------------------------
    if alice:
        X = np.load(f'{pre}/snap_{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}/snap_{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}/snap_{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}/snap_{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}/snap_{snap}/Vz_{snap}.npy')
        T = np.load(f'{pre}/snap_{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}/snap_{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}/snap_{snap}/Rad_{snap}.npy') # specific rad energy
        Vol = np.load(f'{pre}/snap_{snap}/Vol_{snap}.npy')
        box = np.load(f'{pre}/snap_{snap}/box_{snap}.npy')
    else:
        X = np.load(f'{pre}/{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}/{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}/{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}/{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}/{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}/{snap}/Vz_{snap}.npy')
        T = np.load(f'{pre}/{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}/{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}/{snap}/Rad_{snap}.npy') # specific rad energy
        Vol = np.load(f'{pre}/{snap}/Vol_{snap}.npy')
        box = np.load(f'{pre}/{snap}/box_{snap}.npy')
    
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Rad, Vol, VX, VY, VZ = make_slices([X, Y, Z, T, Den, Rad, Vol, VX, VY, VZ], denmask)
    Rad_den = np.multiply(Rad,Den) # now you have energy density
    del Rad   
    xyz = np.array([X, Y, Z]).T
    R = np.sqrt(X**2 + Y**2 + Z**2)
    # Cross dot -----------------------------------------------------------------
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz).T # shape: (192, 3)
    num_obs = prel.NPIX # you'll use it for the mean of the observers. It's 192, unless you don't find the photosphere for someone and so decrease of 1

    # Dynamic Box 
    reds = np.zeros(prel.NPIX)
    ph_idx = np.zeros(prel.NPIX)
    xph = np.zeros(prel.NPIX) 
    yph = np.zeros(prel.NPIX)
    zph = np.zeros(prel.NPIX)
    volph = np.zeros(prel.NPIX)
    denph = np.zeros(prel.NPIX) 
    Tempph = np.zeros(prel.NPIX)
    Rad_denph = np.zeros(prel.NPIX)
    Vxph = np.zeros(prel.NPIX) 
    Vyph = np.zeros(prel.NPIX)
    Vzph = np.zeros(prel.NPIX)
    fluxes = np.zeros(prel.NPIX)
    r_initial = np.zeros(prel.NPIX) # initial starting point for Rph
    for i in range(prel.NPIX):
        # Progress 
        print(f'Snap: {snap}, Obs: {i}', flush=False)
        sys.stdout.flush()

        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        # Box is for dynamic ray making
        # box gives -x, -y, -z, +x, +y, +z
        if mu_x < 0:
            rmax = box[0] / mu_x
            # print('x-', rmax)
        else:
            rmax = box[3] / mu_x
            # print('x+', rmax)
        if mu_y < 0:
            rmax = min(rmax, box[1] / mu_y)
            # print('y-', rmax)
        else:
            rmax = min(rmax, box[4] / mu_y)
            # print('y+', rmax)

        if mu_z < 0:
            rmax = min(rmax, box[2] / mu_z)
            # print('z-', rmax)
        else:
            rmax = min(rmax, box[5] / mu_z)
            # print('z+', rmax)

        r = np.logspace(-0.25, np.log10(rmax), N_ray)
        r_initial[i] = rmax # this is true if the observers are nomalized to have |R|=1

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        del x, y, z
        # find the simulation cell corresponding to cells in the wanted ray
        tree = KDTree(xyz, leaf_size=50) 
        _, idx = tree.query(xyz2, k=1)
        idx = [ int(idx[i][0]) for i in range(len(idx))]
        # Quantity corresponding to the ray
        d = Den[idx] * prel.den_converter
        t = T[idx]
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        rad_den = Rad_den[idx]
        volume = Vol[idx]
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        
        # Interpolate opacity 
        sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
        sigma_rossland = np.array(sigma_rossland)[0]
        underflow_mask = sigma_rossland != 0.0
        idx = np.array(idx)
        d, t, r, ray_x, ray_y, ray_z, sigma_rossland, rad_den, volume, ray_vx, ray_vy, ray_vz, idx = \
            make_slices([d, t, r, ray_x, ray_y, ray_z, sigma_rossland, rad_den, volume, ray_vx, ray_vy, ray_vz, idx], underflow_mask)
        sigma_rossland_eval = np.exp(sigma_rossland) # [1/cm]
        del sigma_rossland
        gc.collect()

        # Optical Depth
        r_fuT = np.flipud(r) #.T
        kappa_rossland = np.flipud(sigma_rossland_eval) 
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
        
        # Red 
        # Get 20 unique nearest neighbors to each cell in the wanted ray and use them to compute the gradient along the ray
        xyz3 = np.array([ray_x, ray_y, ray_z]).T
        _, idxnew = tree.query(xyz3, k=20)
        idxnew = np.unique(idxnew) #.T
        dx = 0.5 * volume**(1/3) # Cell radius 
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

        grad = np.sqrt(gradx**2 + grady**2 + gradz**2) # grad = |grad|
        gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz) # projection of the gradient along the radial direction
        del gradx, grady, gradz
        gc.collect()

        # Eq.(28) from Krumholz07.
        R_lamda = grad / ( prel.Rsol_cgs * sigma_rossland_eval* rad_den) # this is the conversion for /r from the gradient. It's dimensionless
        R_lamda[R_lamda < 1e-10] = 1e-10
        # Eq.(27) from Krumholz07.
        fld_factor = (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        # Eq.(26) from Krumholz07. You miss a c, but it's in Lphoto2 for computational reasons.
        # Before it was r.T
        smoothed_flux = -uniform_filter1d(r**2 * fld_factor * gradr / sigma_rossland_eval, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 

        # You can have numerical errors at early times
        try: 
            photosphere = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0] 
        except IndexError: # if you don't find the photosphere, exlude the observer
            num_obs -= 1
            print(f'No photosphere found for observer {i}, now observers are {num_obs}', flush=False)
            sys.stdout.flush()
            continue
        Lphoto2 = 4*np.pi * prel.c_cgs*smoothed_flux[photosphere] * prel.Msol_cgs / (prel.tsol_cgs**2) # you have to convert rad_den*r^2/lenght = energy/lenght^2 = mass/time^2
        if Lphoto2 < 0:
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives
        # free streaming emission
        max_length = 4*np.pi*(r[photosphere]**2) * prel.c_cgs * rad_den[photosphere] * prel.Msol_cgs * prel.Rsol_cgs / (prel.tsol_cgs**2) #the conversion is for rad_den*r^2 = mass*len/time^2
        Lphoto = np.min( [Lphoto2, max_length])
        reds[i] = Lphoto # cgs
        ph_idx[i] = idx[photosphere]
        xph[i] = ray_x[photosphere]
        yph[i] = ray_y[photosphere]
        zph[i] = ray_z[photosphere]
        volph[i] = volume[photosphere]
        denph[i] = d[photosphere]
        Tempph[i] = t[photosphere]
        Rad_denph[i] = rad_den[photosphere]
        Vxph[i] = ray_vx[photosphere]
        Vyph[i] = ray_vy[photosphere]
        Vzph[i] = ray_vz[photosphere]
        fluxes[i] = Lphoto / (4*np.pi*(r[photosphere]*prel.Rsol_cgs)**2)

        del smoothed_flux, R_lamda, fld_factor, rad_den
        gc.collect()
    Lphoto_snap = np.sum(reds)/num_obs # take the mean
    print(Lphoto_snap, flush=True)
    sys.stdout.flush()

    if save:
        # Save red of the single snap
        pre_saving = f'{abspath}/data/{folder}'
        data = [snap, tfb[idx_s], Lphoto_snap]
        with open(f'{pre_saving}/{check}_red.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()

        # save Rph index and fluxes for each observer in the snapshot
        time_rph = np.concatenate([[snap,tfb[idx_s]], ph_idx])
        time_fluxes = np.concatenate([[snap,tfb[idx_s]], fluxes])
        with open(f'{pre_saving}/{check}_phidx_fluxes.txt', 'a') as fileph:
            fileph.write(f'# {folder}_{check}. First data is snap, second time (in t_fb), the rest are the photosphere indices \n')
            fileph.write(' '.join(map(str, time_rph)) + '\n')
            fileph.write(f'# {folder}_{check}. First data is snap, second time (in t_fb), the rest are the fluxes [cgs] for each obs \n')
            fileph.write(' '.join(map(str, time_fluxes)) + '\n')
            fileph.close()
        
        with open(f'{pre_saving}/photo/{check}_photo{snap}.txt', 'w') as f:
            f.write('# Data for the photospere. Lines are: xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph \n # NB d is in CGS \n')
            f.write(' '.join(map(str, xph)) + '\n')
            f.write(' '.join(map(str, yph)) + '\n')
            f.write(' '.join(map(str, zph)) + '\n')
            f.write(' '.join(map(str, volph)) + '\n')
            f.write(' '.join(map(str, denph)) + '\n')
            f.write(' '.join(map(str, Tempph)) + '\n')
            f.write(' '.join(map(str, Rad_denph)) + '\n')
            f.write(' '.join(map(str, Vxph)) + '\n')
            f.write(' '.join(map(str, Vyph)) + '\n')
            f.write(' '.join(map(str, Vzph)) + '\n')
            f.close()
        
eng.exit()
# usage = resource.getrusage(resource.RUSAGE_SELF)
# print(f"Peak RAM usage: {usage.ru_maxrss / 1024**2:.2f} MB")
# %%
