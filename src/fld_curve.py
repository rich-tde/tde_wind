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
from src.Opacity.interpolator_vectorized import calc_planck_opacity_vectorized, calc_ross_opacity_vectorized, calc_scattering_opacity_vectorized

from scipy.ndimage import uniform_filter1d
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
from Utilities.operators import make_tree

def fld_lightcurve(params, compton, check, N_ray):
    m, Rstar, mstar, beta, n, compton = params
    Mbh = 10**m
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
    pre_saving = f'{abspath}/data/{folder}'

    # observers 
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) # shape: (3, 192)
    observers_xyz = np.array(observers_xyz).T # shape: (192, 3)

    for idx_s, snap in enumerate(snaps):
        if snap not in [76]: 
            continue
        print(f'Snap: {snap}', flush=True)
        if alice:
            loadpath = f'{pre}/snap_{snap}'
        else:
            loadpath = f'{pre}/{snap}'
        Lphoto_snap, photosphere, colorsphere, freqs, L_col = single_fld(loadpath, snap, observers_xyz, N_ray)
        data = [snap, tfb[idx_s], Lphoto_snap]
        if save:
            with open(f'{pre_saving}/{check}_red.csv', 'a', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            file.close()

            np.savez(f"{pre_saving}/photo/{check}_photo{snap}.npz", **photosphere)
            # Save spectrum
            np.savez(f"{pre_saving}/spectra/{check}_Rcol{snap}.npz", **colorsphere)
            np.savetxt(f'{pre_saving}/spectra/freqs.txt', freqs)
            np.savetxt(f'{pre_saving}/spectra/{check}_spectra{snap}.txt', L_col)
        del Lphoto_snap, photosphere, colorsphere, L_col
        gc.collect()

def single_fld(loadpath, snap, observers_xyz, N_ray):
    num_obs = len(observers_xyz)
    data = make_tree(loadpath, snap)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Rad, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        make_slices([X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den], denmask)
    xyz = np.array([X, Y, Z]).T
    R = np.sqrt(X**2 + Y**2 + Z**2)

    photosphere = {'idx': [], 'x': [], 'y': [], 'z': [], 'vol': [], 'den': [], 'temp': [], 'radden': [], 'vx': [], 'vy': [], 'vz': [], 'P': [], 'ieden': [], 'los': [], 'los_scatt': [], 'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [], 'r': [], 'Lum': [], 'Fx': [], 'Fy': [], 'Fz': []}
    colorsphere = {'idx': [], 'x': [], 'y': [], 'z': [], 'vol': [], 'den': [], 'temp': [], 'radden': [], 'vx': [], 'vy': [], 'vz': [], 'P': [], 'ieden': [], 'los': [], 'los_scatt': [], 'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [], 'alpha_eff': []}
    freqs = prel.freqs
    L_col = np.zeros((num_obs, len(prel.freqs)))
    for i in range(num_obs):
        # if i not in [0, 90, 100, 130]:
        #     continue
        print(f'Obs: {i}', flush=True)

        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        # be sure is normalized, but it should be because it's a healpy vector
        norm_mu = np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
        if norm_mu != 1.:
            print('normalizing observers direction. norm:', norm_mu, flush=True)
            mu_x /= norm_mu
            mu_y /= norm_mu
            mu_z /= norm_mu

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

        r = np.logspace(-0.25, np.log10(rmax), N_ray)

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        del x, y, z
        # find the simulation cell corresponding to cells in the wanted ray
        tree = KDTree(xyz, leaf_size=50) 
        _, idx = tree.query(xyz2, k=1)
        idx = idx.ravel()
        # idx = np.unique(idx).astype(np.int64) #####
        # Quantity corresponding to the ray
        d = Den[idx] * prel.den_converter
        t = T[idx]
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        ray_radDen = Rad_den[idx]
        volume = Vol[idx]
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        ray_press = Press[idx]
        ray_ie_den = IE_den[idx]
        # r = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2) 

        alpha_scatter = calc_scattering_opacity_vectorized(T_cool, Rho_cool, scattering, np.log(t), np.log(d))
        alpha_scatter = np.array(alpha_scatter)
        alpha_rossland = calc_ross_opacity_vectorized(T_cool, Rho_cool, rossland, scattering, np.log(t), np.log(d))
        alpha_rossland = np.array(alpha_rossland)
        alpha_planck = calc_planck_opacity_vectorized(T_cool, Rho_cool, planck, np.log(t), np.log(d))
        alpha_planck = np.array(alpha_planck)
        
        underflow_mask = np.logical_and(np.logical_and(np.log(alpha_rossland) != 0.0, np.log(alpha_planck) != 0.0), np.log(alpha_scatter) != 0.0)
        d, t, r, ray_x, ray_y, ray_z, alpha_rossland, alpha_planck, alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx = \
            make_slices([d, t, r, ray_x, ray_y, ray_z, alpha_rossland, alpha_planck, alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx], underflow_mask)
        idx = np.array(idx)

        # Optical depth
        r_fuT = np.flipud(r) #.T
        alpha_rossland_fuT = np.flipud(alpha_rossland) 
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        los = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r

        alpha_scatter_fuT = np.flipud(alpha_scatter) 
        los_scatt = - np.flipud(sci.cumulative_trapezoid(alpha_scatter_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
        
        alpha_effective = np.sqrt(3 * np.minimum(alpha_planck, alpha_rossland) * alpha_rossland)
        alpha_effective_fuT = np.flipud(alpha_effective)
        los_effective = - np.flipud(sci.cumulative_trapezoid(alpha_effective_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs
        los_effective[los_effective > 30] = 30
        del r_fuT, alpha_rossland_fuT, alpha_scatter_fuT, alpha_effective_fuT

        # FLD curve 
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
        # gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz) # projection of the gradient along the radial direction
        gc.collect()

        # Eq.(28) from Krumholz07.
        R_lamda = grad / ( prel.Rsol_cgs * alpha_rossland* ray_radDen) # this is the conversion for /r from the gradient. It's dimensionless
        R_lamda[R_lamda < 1e-10] = 1e-10
        # Eq.(27) from Krumholz07.
        fld_factor = (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        # Eq.(26) from Krumholz07. 
        Fx = -prel.c_cgs * fld_factor * gradx * (prel.en_den_converter/prel.Rsol_cgs)/ alpha_rossland # CGS
        Fy = -prel.c_cgs * fld_factor * grady * (prel.en_den_converter/prel.Rsol_cgs)/ alpha_rossland
        Fz = -prel.c_cgs * fld_factor * gradz * (prel.en_den_converter/prel.Rsol_cgs)/ alpha_rossland
        Fr = (mu_x * Fx) + (mu_y*Fy) + (mu_z*Fz)
        # smoothed_flux_r2 = -prel.c_cgs * uniform_filter1d(r**2 * fld_factor * gradr / alpha_rossland, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 
        smoothed_flux_r2 = uniform_filter1d(r**2 * Fr, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 
        del gradx, grady, gradz, R_lamda, fld_factor

        try: 
            photo_idx = np.where( ((smoothed_flux_r2>0) & (los<2/3) ))[0][0] 
        except IndexError: # if you don't find the photosphere, skip the observer
            print(f'No photosphere found for observer {i}', flush=True)
            continue
        # Lphoto2 = 4*np.pi * smoothed_flux_r2[photosphere] * prel.Msol_cgs / (prel.tsol_cgs**2) # you have to convert ray_radDen*r^2/lenght = energy/lenght^2 = mass/time^2
        Lphoto2 = 4*np.pi * smoothed_flux_r2[photo_idx] * prel.Rsol_cgs**2 # you have to convert the r^2 in smoothed_flux_r2
        if Lphoto2 < 0: 
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives
        # free streaming emission
        max_length = 4*np.pi*(r[photo_idx]**2) * prel.c_cgs * ray_radDen[photo_idx] * prel.Msol_cgs * prel.Rsol_cgs / (prel.tsol_cgs**2) #the conversion is for ray_radDen*r^2 = mass*len/time^2
        Lphoto = np.min( [Lphoto2, max_length]) #that's usually Lphoto2
        photosphere['idx'].append(idx[photo_idx])
        photosphere['x'].append(ray_x[photo_idx])
        photosphere['y'].append(ray_y[photo_idx])
        photosphere['z'].append(ray_z[photo_idx])
        photosphere['vol'].append(volume[photo_idx])
        photosphere['den'].append(d[photo_idx])
        photosphere['temp'].append(t[photo_idx])
        photosphere['radden'].append(ray_radDen[photo_idx])
        photosphere['vx'].append(ray_vx[photo_idx])
        photosphere['vy'].append(ray_vy[photo_idx])
        photosphere['vz'].append(ray_vz[photo_idx])
        photosphere['P'].append(ray_press[photo_idx])
        photosphere['ieden'].append(ray_ie_den[photo_idx])
        photosphere['los'].append(los[photo_idx])
        photosphere['los_scatt'].append(los_scatt[photo_idx])
        photosphere['alpha_rossland'].append(alpha_rossland[photo_idx])
        photosphere['alpha_scatter'].append(alpha_scatter[photo_idx])
        photosphere['alpha_abs'].append(alpha_planck[photo_idx])
        photosphere['r'].append(r[photo_idx])
        photosphere['Fx'].append(Fx[photo_idx])
        photosphere['Fy'].append(Fy[photo_idx])
        photosphere['Fz'].append(Fz[photo_idx])
        photosphere['Lum'].append(Lphoto) # fluxes was from here as L/4pi r[photo)idx]**2

        # Spectra
        try: 
            color_idx = np.where(los_effective<5)[0][0]
        except IndexError:
            print(f'No color index found for observer {i}', flush=True)
            continue
        colorsphere['idx'].append(idx[color_idx])
        colorsphere['x'].append(ray_x[color_idx])
        colorsphere['y'].append(ray_y[color_idx])
        colorsphere['z'].append(ray_z[color_idx])
        colorsphere['vol'].append(volume[color_idx])
        colorsphere['den'].append(d[color_idx])
        colorsphere['temp'].append(t[color_idx])
        colorsphere['radden'].append(ray_radDen[color_idx])
        colorsphere['vx'].append(ray_vx[color_idx])
        colorsphere['vy'].append(ray_vy[color_idx])
        colorsphere['vz'].append(ray_vz[color_idx])
        colorsphere['P'].append(ray_press[color_idx])
        colorsphere['ieden'].append(ray_ie_den[color_idx])
        colorsphere['los_scatt'].append(los_scatt[color_idx])
        colorsphere['alpha_rossland'].append(alpha_rossland[color_idx])
        colorsphere['alpha_scatter'].append(alpha_scatter[color_idx])
        colorsphere['alpha_abs'].append(alpha_planck[color_idx])
        colorsphere['alpha_eff'].append(alpha_effective[color_idx])

        # Spectra ---
        for k in range(color_idx, len(r)):
            # if k == 0:
            #     continue
            dr = r[k]-r[k-1]
            Vcell =  r[k]**2 * dr # there should be a (4 * np.pi / 192)*, but doesn't matter because we normalize
            wien = np.exp(prel.h_cgs * freqs / (prel.Kb_cgs * t[k])) - 1
            black_body = freqs**3 / wien # There should be a 2 * prel.h_cgs/c^2, but it doesn't matter because we normalize. BB udm: erg/s/cm^2/Hz/ster. 
            L_col[i,:] += alpha_planck[k] * Vcell * np.exp(-los_effective[k]) * black_body # erg/s/Hz.
        
        norm = Lphoto / np.trapezoid(L_col[i,:], freqs)
        L_col[i,:] *= norm

        del smoothed_flux_r2, ray_radDen, alpha_rossland, alpha_planck, alpha_scatter, los, los_effective, tree, volume, ray_x, ray_y, ray_z, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den 
        gc.collect()

    Lphoto_snap = np.mean(photosphere['Lum'])
    print('L :', Lphoto_snap, flush=True)

    return Lphoto_snap, photosphere, colorsphere, freqs, L_col

if __name__ == "__main__":
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'HiResNewAMR' 
    N_ray = 5_000
    params = [m, Rstar, mstar, beta, n, compton]
    
    # Load opacity tables
    opac_path = f'{abspath}/src/Opacity'
    T_cool = np.loadtxt(f'{opac_path}/T.txt')
    Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
    rossland = np.loadtxt(f'{opac_path}/ross.txt')
    planck = np.loadtxt(f'{opac_path}/planck.txt')
    scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm

    #MATLAB, thanks Cindy.
    eng = matlab.engine.start_matlab()

    fld_lightcurve(params, compton, check, N_ray)