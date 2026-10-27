""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
import sys
sys.path.append('/Users/paolamartire/shocks')

abspath = '/Users/paolamartire/shocks'
import numpy as np
import healpy as hp
import scipy.integrate as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
mpl.rcParams['savefig.transparent'] = False 
mpl.rcParams['figure.facecolor'] = 'white'    # Figure when displayed
mpl.rcParams['axes.facecolor'] = 'white'  
from src.Opacity.interpolator_vectorized import calc_planck_opacity_vectorized, calc_ross_opacity_vectorized, calc_scattering_opacity_vectorized
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
from Utilities.operators import make_tree, sort_list
import gc
import warnings
warnings.filterwarnings('ignore')

compute = False
save = True

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
        loadpath = f'{pre}/{snap}'
        colorsphere_groups, colorsphere, freqs, L_col = single_fld(loadpath, snap, observers_xyz, N_ray)

        if save:
            # Save spectrum
            # np.savez(f"{pre_saving}/spectra/mixMG/{check}_Rcolgroup_mixMG{snap}.npz", **colorsphere_groups)
            np.savez(f"{pre_saving}/spectra/mixMG/{check}_Rcol_mixMG{snap}.npz", **colorsphere)
            np.savetxt(f'{pre_saving}/spectra/mixMG/freqs_mixMG.txt', freqs)
            np.savetxt(f'{pre_saving}/spectra/mixMG/{check}_spectra_mixMG{snap}.txt', L_col)
        del colorsphere, L_col
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

    colorsphere = {'idx': [], 'x': [], 'y': [], 'z': [], 'vol': [], 'den': [], 'temp': [], 'radden': [], 'vx': [], 'vy': [], 'vz': [], 'P': []} # each list have length num_obs
    colorsphere_groups = {g: {
                            'idx': [], 'x': [], 'y': [], 'z': [],
                            'vol': [], 'den': [], 'temp': [], 'radden': [],
                            'vx': [], 'vy': [], 'vz': [], 'P': [],
                            'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [], 'alpha_eff': []}
                        for g in range(Ng)}
    freqs = prel.freqs
    L_col = np.zeros((num_obs, len(prel.freqs)))

    for i in range(num_obs):
        # if i not in [0, 10, 20, 30, 40, 70, 80, 90, 100, 110, 140, 150, 180]:
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
        idx = np.array(idx)
        # idx = np.unique(idx).astype(np.int64)
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

        # Optical depth
        r_fuT = np.flipud(r) #.T
        alpha_planck = calc_planck_opacity_vectorized(T_cool, Rho_cool, planck, np.log(t), np.log(d))
        alpha_planck = np.array(alpha_planck)
        alpha_planck_groups = np.zeros((Ng, len(r)))
        alpha_rossland_groups = np.zeros((Ng, len(r)))
        los_effective_groups = np.zeros((Ng, len(r)))
        # print(np.log(t))
        for g in range(Ng):
            rossland_kappa_g_table = np.loadtxt(f'{opac_path_mg}/sigma_rossland_{g+1}.txt').T # they are NOT in log, are in cm2/g
            planck_kappa_g_table = np.loadtxt(f'{opac_path_mg}/sigma_absorption_rossland_{g+1}.txt').T
            scattering_kappa_g_table = np.loadtxt(f'{opac_path_mg}/sigma_scattering_planck_{g+1}.txt').T
            rossland_g_table = rossland_kappa_g_table * Rho_cool_mg[:, None]
            planck_g_table = planck_kappa_g_table * Rho_cool_mg[:, None]
            scattering_g_table = scattering_kappa_g_table * Rho_cool_mg[:, None]
            Rho_cool_mg_log = np.log(Rho_cool_mg)
            T_cool_mg_log = np.log(T_cool_mg)
            rossland_g_table = np.log(rossland_g_table)
            planck_g_table = np.log(planck_g_table)
            scattering_g_table = np.log(scattering_g_table)

            alpha_rossland_g = calc_ross_opacity_vectorized(
                                T_cool_mg_log,
                                Rho_cool_mg_log,
                                rossland_g_table,
                                scattering_g_table,
                                np.log(t),
                                np.log(d))

            alpha_planck_g = calc_planck_opacity_vectorized(
                                T_cool_mg_log,
                                Rho_cool_mg_log,
                                planck_g_table,
                                np.log(t),
                                np.log(d))

            alpha_rossland_g = np.asarray(alpha_rossland_g)
            alpha_planck_g = np.asarray(alpha_planck_g)


            # underflow_mask = np.logical_and(np.log(alpha_rossland_g) != 0.0, np.log(alpha_planck_g) != 0.0)
            # alpha_rossland_g[~underflow_mask] = 0.0
            # alpha_planck_g[~underflow_mask] = 0.0

            alpha_rossland_groups[g] = alpha_rossland_g
            alpha_planck_groups[g] = alpha_planck_g

            alpha_effective_g = np.sqrt(3.0 * np.minimum(alpha_planck_g, alpha_rossland_g) * alpha_rossland_g)
            # print(f'alpha_effective_g for group {g}:', alpha_effective_g, flush=True)
            
            los_effective_groups[g] = -np.flipud(sci.cumulative_trapezoid(np.flipud(alpha_effective_g), r_fuT, initial=0)) * prel.Rsol_cgs

        los_effective_groups[los_effective_groups > 30] = 30
        

        color_idx_groups = np.full(Ng, -1, dtype=int)
        for g in range(Ng):
            idx_g = np.where(los_effective_groups[g] < 5)[0]
            if len(idx_g) == 0:
                print(f'No color index found for observer {i}, group {g}', flush=True)
                colorsphere_groups[g]['idx'].append(np.nan)
                colorsphere_groups[g]['x'].append(np.nan)
                colorsphere_groups[g]['y'].append(np.nan)
                colorsphere_groups[g]['z'].append(np.nan)
                colorsphere_groups[g]['vol'].append(np.nan)
                colorsphere_groups[g]['den'].append(np.nan)
                colorsphere_groups[g]['temp'].append(np.nan)
                colorsphere_groups[g]['radden'].append(np.nan)
                colorsphere_groups[g]['vx'].append(np.nan)
                colorsphere_groups[g]['vy'].append(np.nan)
                colorsphere_groups[g]['vz'].append(np.nan)
                colorsphere_groups[g]['P'].append(np.nan)
                colorsphere_groups[g]['alpha_rossland'].append(np.nan)
                # colorsphere_groups[g]['alpha_scatter'].append(np.nan)
                colorsphere_groups[g]['alpha_abs'].append(np.nan)
                colorsphere_groups[g]['alpha_eff'].append(np.nan)
                continue
            color_idx_groups[g] = idx_g[0]
            colorsphere_groups[g]['idx'].append(idx[idx_g])
            colorsphere_groups[g]['x'].append(ray_x[idx_g])
            colorsphere_groups[g]['y'].append(ray_y[idx_g])
            colorsphere_groups[g]['z'].append(ray_z[idx_g])
            colorsphere_groups[g]['vol'].append(volume[idx_g])
            colorsphere_groups[g]['den'].append(d[idx_g])
            colorsphere_groups[g]['temp'].append(t[idx_g])
            colorsphere_groups[g]['radden'].append(ray_radDen[idx_g])
            colorsphere_groups[g]['vx'].append(ray_vx[idx_g])
            colorsphere_groups[g]['vy'].append(ray_vy[idx_g])
            colorsphere_groups[g]['vz'].append(ray_vz[idx_g])
            colorsphere_groups[g]['P'].append(ray_press[idx_g])
            colorsphere_groups[g]['alpha_rossland'].append(alpha_rossland_groups[g][idx_g])
            # colorsphere_groups[g]['alpha_scatter'].append(scattering_g_table[idx_g])
            colorsphere_groups[g]['alpha_abs'].append(alpha_planck_groups[g][idx_g])
            colorsphere_groups[g]['alpha_eff'].append(alpha_effective_g[idx_g])

        valid = color_idx_groups >= 0
        if not np.any(valid):
            print(f'No color surface found for observer {i}', flush=True)
            continue
        r_th_groups = r[color_idx_groups]
        r_th_min = np.min(r_th_groups)
        color_idx = np.argmin(np.abs(r - r_th_min))

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

        # Spectra ---
        for g in range(Ng):
            # Frequencies belonging to this MG group
            freq_mask = ((freqs >= f_edges[g]) & (freqs < f_edges[g+1]))
            if not np.any(freq_mask):
                continue
            nu = freqs[freq_mask]

            for k in range(max(color_idx, 1), len(r)):
                dr = r[k] - r[k-1]
                Vcell = r[k]**2 * dr
                wien = np.exp(prel.h_cgs * nu / (prel.Kb_cgs * t[k])) - 1
                black_body = nu**3 / wien
                L_col[i, freq_mask] += (
                    alpha_planck_groups[g, k]
                    * Vcell
                    * np.exp(-los_effective_groups[g, k])
                    * black_body)

        del ray_radDen, tree, volume, ray_x, ray_y, ray_z, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den 
        gc.collect()

    return colorsphere_groups, colorsphere, freqs, L_col


def load_fld_data(folder, check):
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter = ",")
    snaps, tfb, luminosity = data[:, 0], data[:, 1], data[:, 2]
    snaps, luminosity, tfb = sort_list([snaps, luminosity, tfb], tfb, unique=True)
    return (np.asarray(snaps, dtype=int),
        np.asarray(tfb, dtype=float),
        np.asarray(luminosity, dtype=float))

if __name__ == "__main__":
    import plotting.paperWind.spectra_lc as sp
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'HiResNewAMR' 
    N_ray = 1_000
    params = [m, Rstar, mstar, beta, n, compton]
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    Ng = 10
    
    # Load opacity tables
    opac_path = f'{abspath}/src/Opacity/fromRICH/'
    T_cool = np.loadtxt(f'{opac_path}/T.txt')
    Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
    rossland = np.loadtxt(f'{opac_path}/ross.txt')
    planck = np.loadtxt(f'{opac_path}/planck.txt')
    T_cool_plot = np.exp(T_cool)
    Rho_cool_plot = np.exp(Rho_cool)
    rossland_plot = np.exp(rossland)
    # print(np.shape(rossland), np.shape(T_cool), np.shape(Rho_cool), flush=True)

    opac_path_mg = f'{abspath}/src/Opacity/fromRICH/MG'
    T_cool_mg = np.loadtxt(f'{opac_path_mg}/T.txt')
    T_cool_mg *= prel.ev_toK
    Rho_cool_mg = np.loadtxt(f'{opac_path_mg}/rho.txt')
    rossland_opt_table = np.loadtxt(f'{opac_path_mg}/sigma_rossland_4.txt').T # they are NOT in log
    # print(np.shape(T_cool_mg), np.shape(Rho_cool_mg))
    f_edges_ev = np.loadtxt(f'{opac_path_mg}/frequency_edges.txt') # [eV] 
    f_edges = f_edges_ev * prel.ev_toHz # [Hz] 

    if compute:
        fld_lightcurve(params, compton, check, N_ray)

    else:
        choice = 'split_stream'
        snaps_spectra = [76, 109, 151]
        # kappa_ross_plot = rossland_plot
        kappa_ross_plot = []
        for i in range(len(T_cool_plot)): 
            kappa_ross_plot.append(rossland_plot[i, :]/Rho_cool_plot)
        kappa_ross_plot = np.array(kappa_ross_plot)

        # fig, (axg, axm) = plt.subplots(1, 2, figsize=(14, 6))
        # axg.pcolormesh(np.log10(T_cool_plot), np.log10(Rho_cool_plot), kappa_ross_plot.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1e7))
        # img = axm.pcolormesh(np.log10(T_cool_mg_plot), np.log10(Rho_cool_mg_plot), rossland_opt_table.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1e7))
        # cbar = plt.colorbar(img)
        # cbar.set_label(r'$\kappa_R$ (cm$^2$/g)', fontsize=18)
        # for ax in [axg, axm]:
        #     ax.set_xlabel(r'$\log_{10}$ Temperature (K)', fontsize=18)
        #     ax.set_xlim(3.5, 9.5)
        #     ax.set_ylim(-14, 2)
        # axg.set_ylabel(r'$\log_{10}$ Density (g/cm$^3$)', fontsize=18)


        # colorsphere_mix = np.load(f"{abspath}/data/{folder}/spectra/{check}_Rcol_mixMG76.npz")
        # x_mix, y_mix, z_mix = colorsphere_mix['x'], colorsphere_mix['y'], colorsphere_mix['z']
        # r_mix = np.sqrt(x_mix**2 + y_mix**2 + z_mix**2)

        # colorsphere = np.load(f"{abspath}/data/{folder}/spectra/{check}_Rcol76.npz")
        # x, y, z = colorsphere['x'], colorsphere['y'], colorsphere['z']
        # r = np.sqrt(x**2 + y**2 + z**2)

        # plt.figure(figsize=(6, 6))
        # plt.scatter(np.arange(len(r))/13, r, color='orange', label='grey')
        # plt.scatter(np.arange(len(r_mix))/13, r_mix, color='blue', label='mixMG')
        # plt.xlabel('Observer index', fontsize=18)
        # plt.ylabel(r'$r_{\rm col} (r_t)$', fontsize=18)
        # plt.yscale('log')
        # plt.legend()

        L_min, L_max = 1e38, 6e42
        T_min, T_max = 1e3, 1e7
        nu_min, nu_max = T_min /prel.Hz_toK, T_max/prel.Hz_toK

        def load_spectrum_mixMG(folder, check, snap):
            ''' Load and normalize the spectrum for a given snapshot. '''
            pre_saving = f'{abspath}/data/{folder}'
            freqs = np.loadtxt(f'{pre_saving}/spectra/mixMG/freqs_mixMG.txt') 
            spectra = np.loadtxt(f'{pre_saving}/spectra/mixMG/{check}_spectra_mixMG{snap}.txt')
            photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
            luminosity = photo["Lum"]
            integrals = np.trapezoid(spectra, freqs, axis=1)
            if np.any(~np.isfinite(integrals)) or np.any(integrals == 0):
                raise ValueError(f"Invalid spectral integral in snapshot {snap}")
            return spectra * (luminosity / integrals)[:, None], luminosity

        def plot_spectra_comp(folder, check, snaps, x_axis, choice, make_plot=False):
            # base = data_path(folder)
            # freqs = np.loadtxt(base / "spectra" / "freqs.txt")
            pre_saving = f'{abspath}/data/{folder}'
            freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt') 
            freqs_mixMG = np.loadtxt(f'{pre_saving}/spectra/mixMG/freqs_mixMG.txt') 
            snaps_fld, tfb, _ = load_fld_data(folder, check)
            xyz, cosine, sectors, labels, colours, central_indices = sp.observer_geometry(choice)
            spectra_mixMG_out = {}
            spectra_out = {}

            if make_plot:
                axes_count = len(snaps)
                fig, axes = plt.subplots(1, axes_count, figsize=(8 * axes_count, 8))
                # axes = axes[0]
                colour_handles, colour_labels = [], []
                handles_T, labels_T = [], []
                x_values = freqs * prel.Hz_toK if x_axis == "Temp" else freqs
                x_values_mixMG = freqs_mixMG * prel.Hz_toK if x_axis == "Temp" else freqs_mixMG

            for s, snap in enumerate(snaps):
                try:
                    np.loadtxt(f'{pre_saving}/spectra/mixMG/{check}_spectra_mixMG{snap}.txt')
                except OSError:
                    # print(f"Snapshot {snap} not found for mixMG, skipping.")
                    continue
                time = tfb[snaps_fld == snap][0]
                spectra, _ = sp.load_spectrum(folder, check, snap)
                spectra = sp.angular_average(spectra, cosine)

                spectra_mixMG, _ = load_spectrum_mixMG(folder, check, snap)
                spectra_mixMG = sp.angular_average(spectra_mixMG, cosine)

                sector_spectra = np.full(
                    (len(sectors), len(freqs)),
                    np.nan)

                sector_spectra_mixMG = np.full(
                                    (len(sectors), len(freqs_mixMG)),
                                    np.nan)
                 
                if make_plot:
                    sp.add_spectral_regions(axes[s], x_axis, text_band=True if s == 0 else False)
                    axes[s].set_title(rf"$t={time:.1f}\,t_{{\rm fb}}$", fontsize=30, y=1.17)
                    axes[s].set_ylim(L_min, 1e42)
                    axes[s].loglog()
                    axes[s].tick_params(axis='both', which='major', length=8, width=1.2)
                    axes[s].tick_params(axis='both', which='minor', length=5, width=1)
                    for f, freq_mg in enumerate(f_edges):
                        if f in [0,2,4,6,7,9]: 
                            continue
                        axes[s].axvline(freq_mg * prel.Hz_toK if x_axis == "Temp" else freq_mg, color='gray')
                            
                    if s == 0:
                        legend_local = axes[s].legend(
                                        handles=handles_T,      
                                        labels=labels_T,
                                        loc='upper left',         
                                        fontsize=14)
                    else:
                        axes[s].legend(fontsize=15, loc = 'upper left' if s == 1 else 'upper right')

                    top = axes[s].twiny()
                    primary_ticks = np.logspace(np.log10(T_min if x_axis == "Temp" else nu_min), np.log10(T_max if x_axis == "Temp" else nu_max), 5)
                    frequencies = primary_ticks / prel.Hz_toK if x_axis == "Temp" else primary_ticks
                    wavelengths = prel.c_cgs * 1e8 / frequencies
                    top.set_xticks(wavelengths, [f"{x:.2f}" for x in wavelengths])
                    top.set_xlim(wavelengths.max(), wavelengths.min())
                    top.set_xscale("log")
                    top.set_xlabel(r"$\lambda\;(\AA)$", fontsize=30)


                for k, idx in enumerate(sectors):
                    if labels[k] in {"South pole"}:
                        continue
                    luminosity = np.mean(spectra[idx], axis=0)
                    luminosity_mixMG = np.mean(spectra_mixMG[idx], axis=0)
                    if np.any(~np.isfinite(luminosity)):
                        print(f"Skipping non-finite sector {labels[k]} at snapshot {snap}")
                        continue

                    sector_spectra[k] = luminosity
                    sector_spectra_mixMG[k] = luminosity_mixMG
                    if make_plot:
                        line = axes[s].plot(x_values, freqs * luminosity, color=colours[k], label = 'gray' if np.logical_and(s == 0, k == 0) else None)[0]
                        axes[s].plot(x_values_mixMG, freqs_mixMG * luminosity_mixMG, color=colours[k], ls="--", label = 'mix MG' if np.logical_and(s == 0, k == 0) else None)[0]
                        if s == 0:
                            colour_handles.append(line)
                            colour_labels.append(labels[k])

                spectra_out[snap] = sector_spectra
                spectra_mixMG_out[snap] = sector_spectra_mixMG

            if make_plot:
                axes[0].set_ylabel(r"$\nu L_\nu$ (erg/s)", fontsize=30)
                fig.legend(colour_handles, colour_labels, loc="lower center", bbox_to_anchor=(0.525, -0.09), ncol=len(colour_labels), fontsize=20)
                fig.tight_layout()
            else:
                return freqs_mixMG, spectra_mixMG_out, freqs, spectra_out

                            
        def plot_light_curves(folder, check, choice):
            pre_saving = f'{abspath}/data/{folder}'
            freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt')
            bands_idx = sp.band_indices(freqs)
            snaps, tfb, luminosity_fld = load_fld_data(folder, check)
            time_singlesnap, curves_gray, curves_mixMG= light_curve_from_spectra(folder, check, choice)

            _, cosine, sectors, labels, colours, central_indices = sp.observer_geometry(choice)
            _, _, sectors_mg, _, _, _ = sp.observer_geometry(choice, nside=8)
            n_sectors, n_times = len(sectors), len(snaps)
            fld_sector = np.zeros((n_sectors, n_times))
            curves = {name: np.zeros((n_sectors, n_times)) for name in ("optical", "UV", "Xray", "EUVXray")}

            for s, snap in enumerate(snaps):
                spectra, luminosity_photo = sp.load_spectrum(folder, check, snap)
                spectra, _ = sp.load_spectrum(folder, check, snap)
                spectra = sp.angular_average(spectra, cosine)
                integrated = sp.integrate_bands(spectra, freqs, bands_idx)
                luminosity_photo = sp.angular_average(luminosity_photo, cosine)
                # fld_sector[:, s] = luminosity_photo[central_indices]
                fld_sector[:, s] = sp.sector_average(luminosity_photo, sectors)
                for band in curves: 
                    curves[band][:, s] = sp.sector_average(integrated[band], sectors)
        
            plotted = [k for k, label in enumerate(labels) if label not in {"South pole", r"-$\hat{z}$"}]
            
            fig, (ax_opt, ax_uv, ax_x) = plt.subplots(1, 3, figsize=(24, 7))
            axes = [ax_opt, ax_uv, ax_x]
            for k in plotted:
                # ax_bol.plot(tfb, curves_fld[k], color=colours[k], lw = 2, label=labels[k], zorder = 4)
                # ax_bol.scatter(time_singlesnap, curves_gray["bolometric"][k], color=colours[k], s = 200) 
                # ax_bol.scatter(time_singlesnap, curves_mixMG["bolometric"][k], color=colours[k], s = 50, zorder = 4, edgecolors = 'k', marker='s')
                ax_opt.plot(tfb, curves["optical"][k], color=colours[k], lw = 2, label=labels[k], zorder = 4)
                ax_opt.scatter(time_singlesnap, curves_gray["optical"][k], color=colours[k], s = 200)
                ax_opt.scatter(time_singlesnap, curves_mixMG["optical"][k], color=colours[k], s = 50, edgecolors = 'k', zorder = 4, marker='s')
                ax_uv.plot(tfb, curves["UV"][k], color=colours[k], lw = 3, zorder = 4)
                ax_uv.scatter(time_singlesnap, curves_gray["UV"][k], color=colours[k], s = 200)
                ax_uv.scatter(time_singlesnap, curves_mixMG["UV"][k], color=colours[k], s = 50, edgecolors = 'k', zorder = 4, marker='s')
                ax_x.plot(tfb, curves["EUVXray"][k], color=colours[k], label=labels[k], lw = 2, zorder = 4)
                ax_x.scatter(time_singlesnap, curves_gray["EUVXray"][k], color=colours[k], s = 200)
                ax_x.scatter(time_singlesnap, curves_mixMG["EUVXray"][k], color=colours[k], s = 50, edgecolors = 'k', zorder = 4, marker='s')
           
            ax_opt.text(0.1, 2e42, "Optical", fontsize=25)
            ax_uv.text(0.1, 2e42, "UV", fontsize=25)
            # ax_bol.text(0.1, 2e42, "Bolometric", fontsize=25)
            ax_x.text(0.1, 2e42, "EUV+Xray", fontsize=25)
            ax_opt.set_ylabel(r"$L$ (erg/s)", fontsize=30)
            # ax_bol.set_ylabel(r"$L$ (erg/s)", fontsize=30)
            # ax_x.set_ylabel(r"$\nu L_\nu$ (erg/s)", fontsize=30)
            ax_opt.legend(fontsize=15)
            # ax_x.legend(fontsize=15, loc = 'upper left')
             
            original_ticks = axes[0].get_xticks()
            sp.format_time_axes(axes, original_ticks, [])
            ax_x.set_ylim(1e36, L_max)
            fig.tight_layout()
            # fig.savefig(f'{abspath}/Figs/2.paperWind/LCs_{choice}_{group}.pdf', dpi=300, bbox_inches="tight")
            # fig_x.savefig(f'{abspath}/Figs/2.paperWind/LCsXray_{choice}_{group}.pdf', dpi=200)

            
        def light_curve_from_spectra(folder, check, choice):
            f_edges = np.loadtxt(f'{opac_path_mg}/frequency_edges.txt') * prel.ev_toHz # [Hz]
            
            snaps, tfb, _ = load_fld_data(folder, check)

            _, _, sectors, labels, colours, _ = sp.observer_geometry(choice)

            freqs_mixMG, spectra_mixMG_out, freqs, spectra_out = plot_spectra_comp(
                folder,
                check,
                snaps,
                x_axis="freq",
                choice=choice,
                make_plot=False)

            # Indices on the mixMG spectral frequency grid
            bands_idx = {
                "optical": np.where(
                    (freqs_mixMG >= f_edges[1]) &
                    (freqs_mixMG <= f_edges[3])
                )[0],

                "UV": np.where(
                    (freqs_mixMG >= f_edges[3]) &
                    (freqs_mixMG <= f_edges[5])
                )[0],

                "EUVXray": np.where(
                    (freqs_mixMG >= f_edges[5]) &
                    (freqs_mixMG <= f_edges[10])
                )[0],

                "bolometric": np.arange(len(freqs_mixMG))
            }

            curves = {band: np.full((len(sectors), len(snaps)), np.nan) for band in bands_idx}
            curves_mixMG = {band: np.full((len(sectors), len(snaps)), np.nan) for band in bands_idx}


            for s, snap in enumerate(snaps):
                if snap not in spectra_mixMG_out:
                    continue
                sector_spectra = spectra_out[snap]
                sector_spectra_mixMG = spectra_mixMG_out[snap]

                integrated = sp.integrate_bands(
                    sector_spectra,
                    freqs,
                    bands_idx
                )

                integrated_mixMG = sp.integrate_bands(
                    sector_spectra_mixMG,
                    freqs_mixMG,
                    bands_idx
                )

                for band in curves:
                    curves[band][:, s] = integrated[band]
                    curves_mixMG[band][:, s] = integrated_mixMG[band]

            return tfb, curves, curves_mixMG


        plot_spectra_comp(folder, check, snaps_spectra, x_axis = 'freq', choice=choice, make_plot=True)
        plot_light_curves(folder, check, choice)