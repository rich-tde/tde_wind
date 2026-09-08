"""Post-process and plot spectra produced by ``fld_curve.py``.
The script normalizes spectra to the FLD photospheric luminosity, 
performs a cosine-weighted angular average, integrates standard bands, 
compares with the MG calculation, 
and fits optical/UV blackbodies.
"""
import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import astropy.units as u
import numpy as np
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
import src.orbits as orb
from lmfit import Model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import lines as mlines
import matplotlib.colors as colors
from astropy.cosmology import FlatLambdaCDM
import Utilities.prelude as prel
from Utilities.operators import choose_observers, sort_list
from src.fld_curve import load_fld_data
from src.Paper1.predictions_for_obs import find_horizon
from plotting.paperEdd.IHopeIsTheLast import ratio_BigOverSmall


m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
choice = 'split_stream' #
x_axis = 'Temp'  # 'Freq' or 'Temp'
snaps_spectra = [76, 109, 151]

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
t_fb_days = things['t_fb_days']
Rt = things['Rt']
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # implies Omega_Lambda = 0.7
# Visible: 4.8e14-7.5e14 Hz  // UV: 7.5e14-3e15 // Xray: 3e15-3e19 Hz (tera:1e12, peta: 1e14, exa: 1e18)
BANDS = {
    "optical": (1.6767 * prel.ev_toHz, 3.358 * prel.ev_toHz), 
    "ZTF_g": (prel.c_cgs/ (prel.ztf_g_band[1] * 1e-8), prel.c_cgs/ (prel.ztf_g_band[0] * 1e-8)),
    "Rubin_g": (prel.c_cgs/ (prel.Rubin_g_band[1] * 1e-8), prel.c_cgs/ (prel.Rubin_g_band[0] * 1e-8)),
    # "ZTF_r": (prel.c_cgs/ (prel.ztf_g_band[1] * 1e-8), prel.c_cgs/ (prel.ztf_g_band[0] * 1e-8)),
    # "ZTF_i": (prel.c_cgs/ (prel.ztf_g_band[1] * 1e-8), prel.c_cgs/ (prel.ztf_g_band[0] * 1e-8)),
    "UV": (3.358 * prel.ev_toHz, 7.7488 * prel.ev_toHz),
    "ULTRASAT": (prel.c_cgs/ (prel.lam_ULTR_max * 1e-8), prel.c_cgs/ (prel.lam_ULTR_min * 1e-8)),
    "EUV": (7.7488 * prel.ev_toHz, 300 * prel.ev_toHz),
    "Xray": (300 * prel.ev_toHz, 2e4 * prel.ev_toHz),
    "eROSITA": (200 * prel.ev_toHz, 2300 * prel.ev_toHz),
}

L_min, L_max = 1e38, 6e42
T_min, T_max = 1e3, 1e7
tfb_min, tfb_max = -0.05, 2.24
nu_min, nu_max = T_min /prel.Hz_toK, T_max/prel.Hz_toK

SURVEY_LIMITS = {
    "Einstein": 3.5e43,
    "eROSITA": 1e41,
    "Rubin": 4e39,
    "ZTF": 1.3e41,
    "ULTRASAT": 5e40,
}

Ledd_sol, _ = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter / prel.tsol_cgs

# -----------------------------------------------------------------------------
# Data and numerical helpers
# -----------------------------------------------------------------------------

def magnitude_ab(Lnu):
    m_ab = -2.5 * np.log10(Lnu) + 51.60
    return m_ab

def planck_nu(nu, T):
    x = prel.h_cgs * nu / (prel.Kb_cgs * T)
    return 2 * prel.h_cgs * nu**3 / prel.c_cgs**2 / np.expm1(x)

def blackbody_lnu(nu, R, T):
    return 4 * np.pi**2 * R**2 * planck_nu(nu, T)

BB_MODEL = Model(blackbody_lnu, independent_vars=["nu"])

def fit_blackbody(freqs, luminosity, fit_indices):
    params = BB_MODEL.make_params(R=1e13, T=1e4)
    params["R"].min = 0.0
    params["T"].min = 0.0
    fit = BB_MODEL.fit(
        luminosity[fit_indices], nu=freqs[fit_indices], params=params
    )
    return fit.params["R"].value, fit.params["T"].value

def band_indices(freqs):
    return {
        name: np.flatnonzero((freqs > lower) & (freqs < upper))
        for name, (lower, upper) in BANDS.items()
    }

def wavelength_indices(freqs, wavelength_band):
    """Return frequency indices for a wavelength interval in Ångström."""
    wavelength_min, wavelength_max = wavelength_band

    # Since ν = c/λ, lambda_max gives nu_min and vice versa.
    nu_min = prel.c_cgs / (wavelength_max * 1e-8)
    nu_max = prel.c_cgs / (wavelength_min * 1e-8)
    return np.flatnonzero((freqs > nu_min) & (freqs < nu_max))

def blackbody_fit_indices(freqs):
    bands = (
        prel.ztf_r_band,
        prel.ztf_i_band,
        prel.swift_u_band,
        prel.swift_b_band,
        prel.swift_v_band,
        prel.swift_uvw1_band,
        prel.swift_uvm2_band,
        prel.swift_uvw2_band,
    )
    return np.unique(np.concatenate([wavelength_indices(freqs, b) for b in bands]))

def observer_geometry(choice, nside=None):
    nside = prel.NSIDE if nside is None else nside
    npix = hp.nside2npix(nside)
    xyz = np.asarray(hp.pix2vec(nside, np.arange(npix)))
    sector_indices, labels, colours, _, _, central_indices = choose_observers(xyz, choice=choice)
    cosine = np.clip(xyz.T @ xyz, 0.0, None) 
    return xyz, cosine, sector_indices, labels, colours, central_indices

def load_spectrum(folder, check, snap):
    ''' Load and normalize the spectrum for a given snapshot. '''
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt') 
    spectra = np.loadtxt(f'{pre_saving}/spectra/{check}_spectra{snap}.txt')
    luminosity = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')["Lum"]
    integrals = np.trapezoid(spectra, freqs, axis=1)
    if np.any(~np.isfinite(integrals)) or np.any(integrals == 0):
        raise ValueError(f"Invalid spectral integral in snapshot {snap}")
    return spectra * (luminosity / integrals)[:, None], luminosity

def angular_average(spectra, cosine):
    weights = cosine / cosine.sum(axis=1, keepdims=True)
    return weights @ spectra

def integrate_bands(spectra, freqs, indices):
    return {
        name: np.trapezoid(spectra[:, idx], freqs[idx], axis=1)
        for name, idx in indices.items()
    }

def sector_average(values, sector_indices):
    return np.asarray([np.mean(values[idx]) for idx in sector_indices])

def load_mg_lightcurves(folder, check, choice):
    pre_saving = f'{abspath}/data/{folder}'
    _, cosine, sectors, _, _, central_indices = observer_geometry(choice, nside=8)
    table = np.loadtxt(f'{pre_saving}/MG/{check}_timesMG.csv', delimiter=',', dtype=float)
    snaps, times = table[:, 0].astype(int), table[:, 1]

    curves = {name: [] for name in ("optical", "UV", "Xray")}
    mean_all_Xray = np.zeros(len(snaps))

    curves_x = []
    for s, snap in enumerate(snaps):
        values = np.loadtxt(f'{pre_saving}/MG/snap_{snap}/L_snap_{snap}.txt')
        bands = {
            "optical": values[:, 1:3].sum(axis=1),
            "UV": values[:, 3:5].sum(axis=1),
            "Xray": values[:, 8:].sum(axis=1),
        }
        curves_x.append(bands["Xray"])
        for name, luminosity in bands.items():
            luminosity = angular_average(luminosity, cosine)
            curves[name].append(sector_average(luminosity, sectors))
            # curves[name].append(luminosity[central_indices])
        mean_all_Xray[s] = np.mean(bands["Xray"])
    curves_x = np.asarray(curves_x).T
    curves = {name: np.asarray(value).T for name, value in curves.items()}
    return times, curves, curves_x, mean_all_Xray


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def add_spectral_regions(ax, x_axis, text_band = False):
    colours = {"optical": "bisque", "UV": "#ffc6ff", "EUV": "lightsteelblue", "Xray": "#c77dff"}
    for name, (lower, upper) in BANDS.items():
        if name in ["ZTF_g", "Rubin_g", "ULTRASAT", "eROSITA"]:
            continue
        left, right = (lower * prel.Hz_toK, upper * prel.Hz_toK) if x_axis == "Temp" else (lower, upper)
        ax.axvspan(left, right, color=colours[name], alpha=0.2)
    if x_axis == "Temp":
        ax.set_xlabel("Temperature (K)", fontsize=30)
        ax.set_xlim(T_min, T_max)
        
    else:
        ax.set_xlabel("Frequency (Hz)", fontsize=30)
        ax.set_xlim(nu_min, nu_max)
    if text_band:
        for name, (lower, upper) in BANDS.items():
            if name in ["ZTF_g", "Rubin_g", "ULTRASAT", "eROSITA"]:
                continue
            left, right = (lower * prel.Hz_toK, upper * prel.Hz_toK) if x_axis == "Temp" else (lower, upper)
            mid = 0.6 * right if name in ["optical", "UV"] else 1.2 * left 
            ax.text(mid, L_max/35, name, fontsize=20, rotation = 90)

def format_time_axes(axes, original_ticks, ratio_axes=()):
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    ticks = np.sort(np.concatenate([original_ticks, midpoints]))
    labels = [f"{x:.2f}" if x in original_ticks else "" for x in ticks]
    day_ticks = ticks * t_fb_days
    day_labels = [f"{d:.2f}" if x in original_ticks else "" for x, d in zip(ticks, day_ticks)]

    for ax in axes:
        ax.set_xticks(ticks, labels)
        ax.set_xlabel(r"$t/t_{\rm fb}$", fontsize=30)
        ax.set_xlim(tfb_min, tfb_max)
        ax.set_ylim((1, 20) if ax in ratio_axes else (L_min, L_max))
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", width=1.2, length=10)
        ax.tick_params(axis="y", which="minor", width=1, length=6)
        ax.grid()
        ax_days = ax.twiny()
        ax_days.set_xticks(day_ticks, day_labels)
        ax_days.set_xlim(tfb_min * t_fb_days, tfb_max * t_fb_days)
        ax_days.set_xlabel(r"$t$ (days)", fontsize=30)

# -----------------------------------------------------------------------------
# Main plots
# -----------------------------------------------------------------------------
def plot_spectra(folder, check, snaps, x_axis, choice, in_moll=False):
    # base = data_path(folder)
    # freqs = np.loadtxt(base / "spectra" / "freqs.txt")
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt') 
    bands_idx = band_indices(freqs)
    fit_idx = blackbody_fit_indices(freqs)
    snaps_fld, tfb, _ = load_fld_data(folder, check)
    xyz, cosine, sectors, labels, colours, central_indices = observer_geometry(choice)
    longitude = np.arctan2(xyz[1], xyz[0])
    latitude = np.pi / 2 - np.arccos(xyz[2])

    axes_count = len(snaps)
    fig, axes = plt.subplots(1, axes_count, figsize=(8 * axes_count, 8))
    # axes = axes[0]
    colour_handles, colour_labels = [], []
    handles_T, labels_T = [], []

    if in_moll:
        lon_mesh, lat_mesh = np.meshgrid(
            np.linspace(-np.pi, np.pi, 360), np.linspace(-np.pi / 2, np.pi / 2, 180)
        )
        moll_figs = {}
        for band in ("optical", "Xray"):
            moll_figs[band] = (
                plt.figure(figsize=(11 * axes_count, 7)),
                gridspec.GridSpec(2, axes_count, wspace=0.1, hspace=0, height_ratios=[1, 0.08]),
            )

    for s, snap in enumerate(snaps):
        matches = np.flatnonzero(snaps_fld == snap)
        if not len(matches):
            raise ValueError(f"Snapshot {snap} is absent from the FLD table")
        time = tfb[matches[0]]
        spectra, _ = load_spectrum(folder, check, snap)

        if in_moll:
            integrated = integrate_bands(spectra, freqs, bands_idx)
            for band in ("optical", "Xray"):
                moll_fig, gs = moll_figs[band]
                ax_moll = moll_fig.add_subplot(gs[0, s], projection="mollweide")
                values = griddata(
                    (longitude, latitude), integrated[band], (lon_mesh, lat_mesh), method="linear"
                )
                image = ax_moll.pcolormesh(
                    lon_mesh, lat_mesh, values, cmap="rainbow",
                    norm=mcolors.LogNorm(vmin=L_min, vmax=(10 if band == "optical" else 4) * L_max),
                )
                ax_moll.set_title(rf"${time:.2f}\,t_{{\rm fb}}$", fontsize=24, y=1.15)
                ax_moll.grid()
                moll_figs[band] = moll_fig, gs, image
        
        spectra = angular_average(spectra, cosine)
        x_values = freqs * prel.Hz_toK if x_axis == "Temp" else freqs
        add_spectral_regions(axes[s], x_axis, text_band=True if s == 0 else False)

        fitted = []
        for k, idx in enumerate(sectors):
            if labels[k] in {"South pole"}:
                continue
            luminosity = np.mean(spectra[idx], axis=0)
            observer_index = central_indices[k]
            # luminosity = spectra[observer_index]
            if np.any(~np.isfinite(luminosity)):
                print(f"Skipping non-finite sector {labels[k]} at snapshot {snap}")
                continue
            radius, temperature = fit_blackbody(freqs, luminosity, fit_idx)
            fitted.append((k, radius, temperature))
            print(
                f"At t={time:.1f} t_fb, observer {labels[k]}: "
                f"Tfit={temperature:.2e} K, Rfit={radius/prel.Rsol_cgs:.2e} Rsol"
            )
            line = axes[s].plot(x_values, freqs * luminosity, color=colours[k], label=labels[k] if s == 0 else None)[0]
            if s == 0:
                colour_handles.append(line)
                colour_labels.append(labels[k])

        for k, radius, temperature in fitted:
            bb = blackbody_lnu(freqs, radius, temperature)
            lineB = axes[s].plot(x_values, freqs * bb, color=colours[k], ls="-.", label = f'T={temperature*1e-4:.1f}' + r' $\times 10^4$ K')[0] # if s < 2 else f'T={temperature*1e-3:.1f}' + r' $\times 10^3$ K')[0]
            if s == 0: 
                handles_T.append(lineB)
                labels_T.append(f'T={temperature*1e-4:.1f}' + r' $\times 10^4$ K' )
            

        axes[s].set_title(rf"$t={time:.1f}\,t_{{\rm fb}}$", fontsize=30, y=1.17)
        axes[s].set_ylim(L_min, 1e42)
        axes[s].loglog()
        axes[s].tick_params(axis='both', which='major', length=8, width=1.2)
        axes[s].tick_params(axis='both', which='minor', length=5, width=1)
                
        if s == 0:
            legend_local = axes[s].legend(
                            handles=handles_T,      
                            labels=labels_T,
                            loc='upper left',         
                            fontsize=15)
        else:
            axes[s].legend(fontsize=16, loc = 'upper left' if s == 1 else 'upper right')

        top = axes[s].twiny()
        primary_ticks = np.logspace(np.log10(T_min if x_axis == "Temp" else nu_min), np.log10(T_max if x_axis == "Temp" else nu_max), 5)
        frequencies = primary_ticks / prel.Hz_toK if x_axis == "Temp" else primary_ticks
        wavelengths = prel.c_cgs * 1e8 / frequencies
        top.set_xticks(wavelengths, [f"{x:.2f}" for x in wavelengths])
        top.set_xlim(wavelengths.max(), wavelengths.min())
        top.set_xscale("log")
        top.set_xlabel(r"$\lambda\;(\AA)$", fontsize=30)

    axes[0].set_ylabel(r"$\nu L_\nu$ (erg s$^{-1}$)", fontsize=30)
    fig.legend(colour_handles, colour_labels, loc="lower center", bbox_to_anchor=(0.525, -0.09), ncol=len(colour_labels), fontsize=20)
    fig.tight_layout()
    fig.savefig(f'{abspath}/Figs/2.paperWind/spectra_{choice}.pdf', dpi=300, bbox_inches='tight')

    if in_moll:
        for band, (moll_fig, gs, image) in moll_figs.items():
            cax = moll_fig.add_subplot(gs[1, :])
            cb = moll_fig.colorbar(image, cax=cax, orientation="horizontal")
            cb.set_label(r"$L_{\rm band}$ (erg s$^{-1}$)")
            moll_fig.suptitle("Optical" if band == "optical" else "X-ray", fontsize=24)
            moll_fig.savefig(f'{abspath}/Figs/2.paperWind/moll_{band}_{choice}.pdf', dpi=300, bbox_inches="tight")

def distance_telescope(folder, check, choice):
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt')
    bands_idx = band_indices(freqs)
    snaps, tfb, luminosity_fld = load_fld_data(folder, check)
    idx_maxL = np.argmax(luminosity_fld)
    snap_maxL = snaps[idx_maxL]
    _, mg, curves_x, _ = load_mg_lightcurves(folder, check, choice)
    _, _, sectors, labels, colours, central_indices = observer_geometry(choice)
    n_sectors, n_times = len(sectors), len(snaps)
    fld_sector = np.zeros((n_sectors, n_times))
    curves = {name: np.zeros((n_sectors, n_times)) for name in ("ZTF_g", "Rubin_g", "ULTRASAT", "eROSITA")}

    for s, snap in enumerate(snaps):
        spectra, luminosity_photo = load_spectrum(folder, check, snap)
        integrated = integrate_bands(spectra, freqs, bands_idx)
        # fld_sector[:, s] = luminosity_photo[central_indices]
        for band in curves:
            if band != "eROSITA":
                curves[band][:, s] = sector_average(integrated[band], sectors)
                # curves[band][:, s] = integrated[band][central_indices]
            else:
                curves[band][:, s] = mg["Xray"][:, s]

    plotted = [k for k, label in enumerate(labels) if label not in {"South pole", r"-$\hat{z}$"}]
    for k in plotted:
        # z_horizon_ZTF = find_horizon(np.max(curves["ZTF_g"][k]), -1, -1, prel.mg_lim_ZTF, which_L = 'band', nu_min=BANDS["ZTF_g"][0], nu_max=BANDS["ZTF_g"][1])
        # print(f'#######\nZTF g-band horizon for {labels[k]}: Lum = {np.max(curves["ZTF_g"][k]):.2e}, z = {z_horizon_ZTF:.3f}, in Mpc = {cosmo.luminosity_distance(z_horizon_ZTF).to(u.Mpc).value:.1f}')
        z_horizon_Rubin = find_horizon(np.max(curves["Rubin_g"][k]), -1, -1, prel.mg_lim_Rubin, which_L = 'band', nu_min=BANDS["Rubin_g"][0], nu_max=BANDS["Rubin_g"][1])
        print(f'#######\nRubin g-band horizon for {labels[k]}: Lum = {np.max(curves["Rubin_g"][k]):.2e}, z = {z_horizon_Rubin:.3f}, in Mpc = {cosmo.luminosity_distance(z_horizon_Rubin).to(u.Mpc).value:.1f}')
        # z_horizon_ULTRASAT = find_horizon(np.max(curves["ULTRASAT"][k]), -1, -1, prel.m_lim_ULTRASAT, which_L = 'band', nu_min=BANDS["ULTRASAT"][0], nu_max=BANDS["ULTRASAT"][1])
        # print(f'#######\nULTRASAT horizon for {labels[k]}: z = {z_horizon_ULTRASAT:.3f}, in Mpc = {cosmo.luminosity_distance(z_horizon_ULTRASAT).to(u.Mpc).value:.1f}') 
        # flux_eROS = 1e-13 # erg/s/cm^2
        # distance_eROS_Mpc = np.sqrt(np.max(curves["eROSITA"][k]) / (4 * np.pi * flux_eROS)) / 3.086e24  # in Mpc (https://en.wikipedia.org/wiki/Parsec 1pc = 3.086e16 m)
        # print(f"#######\neROSITA horizon for {labels[k]}: {distance_eROS_Mpc:.1f} Mpc")
        
def plot_light_curves(folder, check, choice, group="bands"):
    if group not in {"sections", "bands", "bandsMG"}:
        raise ValueError("group must be 'sections', 'bands', or 'bandsMG'")

    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt')
    bands_idx = band_indices(freqs)
    snaps, tfb, luminosity_fld = load_fld_data(folder, check)
    idx_maxL = np.argmax(luminosity_fld)
    _, cosine, sectors, labels, colours, central_indices = observer_geometry(choice)
    _, _, sectors_mg, _, _, _ = observer_geometry(choice, nside=8)
    n_sectors, n_times = len(sectors), len(snaps)
    fld_sector = np.zeros((n_sectors, n_times))
    curves = {name: np.zeros((n_sectors, n_times)) for name in ("optical", "UV", "Xray")}
    curves_fld, curves_op, curves_uv = [], [], []

    for s, snap in enumerate(snaps):
        spectra, luminosity_photo = load_spectrum(folder, check, snap)
        spectra, _ = load_spectrum(folder, check, snap)
        spectra = angular_average(spectra, cosine)
        integrated = integrate_bands(spectra, freqs, bands_idx)
        luminosity_photo = angular_average(luminosity_photo, cosine)
        # fld_sector[:, s] = luminosity_photo[central_indices]
        curves_fld.append(luminosity_photo)
        curves_op.append(integrated["optical"])
        curves_uv.append(integrated["UV"]) 
        fld_sector[:, s] = sector_average(luminosity_photo, sectors)
        for band in curves: 
            curves[band][:, s] = sector_average(integrated[band], sectors)
            # curves[band][:, s] = integrated[band][central_indices]
    curves_fld = np.asarray(curves_fld).T
    curves_op, curves_uv = np.asarray(curves_op).T, np.asarray(curves_uv).T
 
    time_mg, mg, curves_x, all_xray_mg = load_mg_lightcurves(folder, check, choice)
    # curves_x = []
    # curves_x.append(mg["Xray"]) 
    # curves_x = np.asarray(curves_x).T
    for target in (1.00, 1.54, 2.23):
        i_fld, i_mg = np.argmin(abs(tfb-target)), np.argmin(abs(time_mg-target))
        print(f"For t={tfb[i_fld]:.2f} t_fb, MG time is {time_mg[i_mg]:.2f} t_fb")
        for k, label in enumerate(labels):
            optical = mg["optical"][k, i_mg] if group == "bandsMG" else curves["optical"][k, i_fld]
            uv = mg["UV"][k, i_mg] if group == "bandsMG" else curves["UV"][k, i_fld]
            print(label, "|| Xray/opt:", mg["Xray"][k, i_mg]/optical, "opt/UV:", optical/uv)

    plotted = [k for k, label in enumerate(labels) if label not in {"South pole", r"-$\hat{z}$"}]
    if group == "sections":
        fig, axes = plt.subplots(1, len(plotted), figsize=(9*len(plotted), 7), squeeze=False)
        axes = list(axes[0])
        for ax, k in zip(axes, plotted):
            ax.plot(tfb, curves["optical"][k], color=colours[k], label="Optical")
            ax.plot(tfb, curves["UV"][k], color=colours[k], ls="--", label="UV")
            ax.plot(time_mg, mg["Xray"][k], color=colours[k], ls=":", label="X-ray")
            ax.text(0.1, L_max/5, labels[k], fontsize=24)
        axes[0].legend(fontsize=20)
        axes[0].set_ylabel(r"$L_{\rm band}$ (erg s$^{-1}$)", fontsize=30)
        ratio_axes = ()
    elif group == "bands":
        fig, (ax_bol, ax_opt, ax_uv) = plt.subplots(1, 3, figsize=(24, 7))
        fig_x, ax_x = plt.subplots(figsize=(9, 7))
        axes = [ax_bol, ax_opt, ax_uv, ax_x]
        for k in plotted:
            for sec in sectors[k]:
                ax_bol.plot(tfb, curves_fld[sec], color=colours[k], alpha = 0.1, lw = 1)
                ax_opt.plot(tfb, curves_op[sec], color=colours[k], alpha = 0.1, lw = 1) 
                ax_uv.plot(tfb, curves_uv[sec], color=colours[k], alpha = 0.1, lw = 1)
            for sec in sectors_mg[k]:
                ax_x.plot(time_mg, curves_x[sec], color=colours[k], alpha = 0.05, lw = .5)
            # ax_opt.plot(tfb, np.mean(curves_op[sectors[k]], axis = 0), color='k', ls = '--')
            ax_bol.plot(tfb, fld_sector[k], color=colours[k], label=labels[k], lw = 3,  zorder = 4)
            ax_bol.scatter(tfb[np.argmax(fld_sector[k])], np.max(fld_sector[k]), c = colours[k], s = 250, marker = '*', edgecolors='k', zorder = 5)
            ax_opt.plot(tfb, curves["optical"][k], color=colours[k], lw = 2, zorder = 4)
            ax_opt.scatter(tfb[np.argmax(curves["optical"][k])], np.max(curves["optical"][k]), c = colours[k], s = 250, marker = '*', edgecolors='k', zorder = 5)
            print(f'{labels[k]}: {np.max(curves["optical"][k]):.2e}')
            ax_uv.plot(tfb, curves["UV"][k], color=colours[k], lw = 3, zorder = 4)
            ax_uv.scatter(tfb[np.argmax(curves["UV"][k])], np.max(curves["UV"][k]), c = colours[k], s = 250, marker = '*', edgecolors='k', zorder = 5)
            ax_x.plot(time_mg, mg["Xray"][k], color=colours[k], label=labels[k], lw = 2, zorder = 4)
            ax_x.scatter(time_mg[np.argmax(mg["Xray"][k])], np.max(mg["Xray"][k]), c = colours[k], s = 250, marker = '*', edgecolors='k', zorder = 5)
        ax_bol.axhline(Ledd_cgs, color="gray", ls="-.")
        ax_opt.text(0.05, L_max/3, r'Optical', fontsize = 26)
        ax_uv.text(0.05, L_max/3, r'UV', fontsize = 26)
        ax_x.text(0.05, L_max/3, 'X-ray (Giron+26)', fontsize = 26)
        ax_bol.text(0.08, 1.2*Ledd_cgs, r'$L_{\rm Edd} (\kappa_{\rm p})$', color = 'gray', fontsize = 20)
        ax_bol.text(0.05, L_max/3, r'Bolometric L$_{\rm FLD}$', fontsize = 26)
        ax_opt.axhline(SURVEY_LIMITS["ZTF"], color="gray", ls="-.")
        ax_opt.text(0, 0.6*SURVEY_LIMITS["ZTF"], 'g-ZTF', fontsize = 16, color = 'gray')
        ax_opt.axhline(SURVEY_LIMITS["Rubin"], color="gray", ls="-.")
        ax_opt.text(0, 0.6*SURVEY_LIMITS["Rubin"], ' g-Rubin', fontsize = 16, color = 'gray')
        ax_uv.axhline(SURVEY_LIMITS["ULTRASAT"], color="gray", ls="-.")
        ax_uv.text(0, 0.6*SURVEY_LIMITS["ULTRASAT"], 'ULTRASAT', fontsize = 16, color = 'gray')
        ax_x.axhline(SURVEY_LIMITS["eROSITA"], color="gray", ls="-.")
        ax_x.text(1.75, 0.6*SURVEY_LIMITS["eROSITA"], 'eROSITA', fontsize = 16, color = 'gray')
        ax_x.axhline(SURVEY_LIMITS["Einstein"], color="gray", ls="-.")
        for ax in (ax_bol, ax_opt, ax_uv):
            ax.plot(tfb, luminosity_fld, "k--", label="All")
            ax.scatter(tfb[idx_maxL], luminosity_fld[idx_maxL], c = 'k', s = 250, marker = '*')
        ax_bol.set_ylabel(r"$L$ (erg s$^{-1}$)", fontsize=30)
        ax_x.set_ylabel(r"$\nu L_\nu$ (erg s$^{-1}$)", fontsize=30)
        ax_bol.legend(fontsize=15)
        ax_x.legend(fontsize=15)
        ratio_axes = ()
    else:
        fig, (ax_opt, ax_uv, ax_x) = plt.subplots(1, 3, figsize=(24, 7))
        fig_ratio, (ratio_opt, ratio_uv, ratio_x) = plt.subplots(1, 3, figsize=(24, 7))
        axes = [ax_opt, ax_uv, ax_x, ratio_opt, ratio_uv, ratio_x]
        handles_color, labels_color = [], []
        for k in plotted:
            for ax, band in zip((ax_opt, ax_uv, ax_x), ("optical", "UV", "Xray")):
                line = ax.plot(tfb, curves[band][k], color=colours[k], label=labels[k])[0]
                if ax == ax_opt:
                    handles_color.append(line)
                    labels_color.append(labels[k])
                ax.plot(time_mg, mg[band][k], color=colours[k], ls=":")
            for ax, band in zip((ratio_opt, ratio_uv, ratio_x), ("optical", "UV", "Xray")):
                time_ratio, ratio, _ = ratio_BigOverSmall(tfb, curves[band][k], time_mg, mg[band][k])
                ax.plot(time_ratio, ratio, color=colours[k])
                if band != "Xray":
                    print('Median ratio for ', labels[k], 'in', band, 'band =', np.median(ratio[np.argmin(np.abs(time_ratio-1.5)):]))
        ratio_opt.set_ylabel("This work / Giron+26", fontsize=25)
        ratio_axes = (ratio_opt, ratio_uv, ratio_x)
        ax_opt.text(1.76, L_max/3, r'Optical', fontsize = 26)
        ax_uv.text(0.05, L_max/3, r'UV', fontsize = 26)
        ax_x.text(0.05, L_max/3, 'X-ray', fontsize = 26)
        ax_opt.set_ylabel(r"$\nu L_\nu$ (erg s$^{-1}$)", fontsize=30)

        legend1 = ax_opt.legend(
                    handles=handles_color,
                    labels=labels_color,
                    fontsize=20,
                    loc='upper left')
        ax_opt.add_artist(legend1)  # Add the first legend to the axes
        method_legend = [mlines.Line2D([0], [0], color="k", ls=ls, label=lab) for ls, lab in zip(("-", ":"), ("This work", "Giron+26"))]
        ax_opt.legend(handles=method_legend, fontsize=18, loc='lower right')

    original_ticks = axes[0].get_xticks()
    format_time_axes(axes, original_ticks, ratio_axes)
    fig.tight_layout()
    fig.savefig(f'{abspath}/Figs/2.paperWind/LCs_{choice}_{group}.pdf', dpi=300, bbox_inches="tight")
    if group == "bands":
        fig_x.tight_layout()
        fig_x.savefig(f'{abspath}/Figs/2.paperWind/LCsXray_{choice}_{group}.pdf', dpi=300)
    if group == "bandsMG":
        fig_ratio.tight_layout()
        fig_ratio.savefig(f'{abspath}/Figs/2.paperWind/LCratios_{choice}_{group}.pdf', dpi=300)

def TRfit_in_time(folder, check, choice):
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt') 
    fit_idx = blackbody_fit_indices(freqs)
    snaps, tfb, _ = load_fld_data(folder, check)
    _, cosine, sectors, labels, colours = observer_geometry(choice)
    shape = (len(snaps), len(sectors))
    lbol, radii, temperatures, fitted_luminosity = (np.zeros(shape) for _ in range(4))

    for s, snap in enumerate(snaps):
        spectra, luminosity_photo = load_spectrum(folder, check, snap)
        spectra = angular_average(spectra, cosine)
        for k, idx in enumerate(sectors):
            luminosity = np.mean(spectra[idx], axis=0)
            radii[s, k], temperatures[s, k] = fit_blackbody(freqs, luminosity, fit_idx)
            bb = blackbody_lnu(freqs, radii[s, k], temperatures[s, k])
            fitted_luminosity[s, k] = np.trapezoid(bb, freqs)
            lbol[s, k] = np.mean(luminosity_photo[idx])

    header = ",".join(["t_fb"] + [f"Tfit_sec {x}" for x in labels] + [f"Rfit_sec {x}" for x in labels])
    np.savetxt(f'{abspath}/data/{folder}/wind/Tfit_intime_{choice}.txt', np.column_stack((tfb, temperatures, radii)), delimiter=",", header=header)

    fig, (ax_t, ax_r, ax_l) = plt.subplots(1, 3, figsize=(28, 8))
    for k, label in enumerate(labels):
        if label in {"South pole", r"-$\hat{z}$"}:
            continue
        ax_t.plot(tfb, temperatures[:, k], color=colours[k], label=label)
        ax_r.plot(tfb, radii[:, k], color=colours[k])
        ax_l.plot(tfb, fitted_luminosity[:, k] / lbol[:, k], color=colours[k])
    ax_t.set_ylabel(r"$T_{\rm BB}$ (K)", fontsize=30)
    ax_r.set_ylabel(r"$R_{\rm BB}$ (cm)", fontsize=30)
    ax_l.set_ylabel(r"$L_{\rm BB}/L_{\rm bol}$", fontsize=30)
    ax_t.set_ylim(4e3, 8e4)
    ax_r.set_ylim(1e11, 1e14)
    ax_l.set_ylim(1e-2, 1)
    ax_t.legend(fontsize=18)
    for ax in (ax_t, ax_r, ax_l):
        ax.set_xlabel(r"$t/t_{\rm fb}$", fontsize=30)
        ax.set_yscale("log")
        ax.grid()
    fig.tight_layout()
    plt.savefig(f'{abspath}/Figs/{folder}/Wind/Tfit_intime_{choice}.png', dpi=300)

    
if __name__ == '__main__':
    # plot_spectra(folder, check, snaps_spectra, x_axis, choice)
    # TRfit_in_time(folder, check, choice)
    plot_light_curves(folder, check, choice, group = 'bands')
    # plot_light_curves(folder, check, choice, group = 'sections')
    # plot_light_curves(folder, check, choice, group = 'bandsMG')
    # distance_telescope(folder, check, choice)

    def lumtest(n, T):
        const = 2*prel.h_cgs/prel.c_cgs**2 
        planck = const * n**3 / (np.exp(prel.h_cgs*n/(prel.Kb_cgs*T))-1)
        return planck

    # x = prel.freqs
    # print(f'Min: {np.min(x):.2e}, Max: {np.max(x):.2e}')
    # plt.figure()
    # plt.plot(x*prel.Hz_toK, x*lumtest(x, 1e4), label = '1e4K')
    # plt.plot(x*prel.Hz_toK, x*lumtest(x, 4e4),  label = '4e4K')
    # plt.axvline(13.6*prel.ev_toHz*prel.Hz_toK, c = 'k', ls = '--')
    # plt.loglog()
    # plt.ylim(1e5, 1e15)
    # plt.xlim(1e3, 1e7)
    # plt.legend(fontsize = 15)
