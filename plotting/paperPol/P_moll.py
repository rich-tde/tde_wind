#!/usr/bin/env python3
abspath = '/Users/paolamartire/shocks'
import sys

sys.path.append(abspath)
import numpy as np
import pyvista as pv
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from src import orbits as orb
from Utilities.operators import split_cells, sort_list
from plotting.paperWind.spectra_lc import add_spectral_regions

def extract_poli(
    vtk_file,
    fields=["polarization_degree"]):
    """
    Extract observer positions and a polarization-related field
    from a RICH post-processing VTK file.

    Parameters
    ----------
    vtk_file : str
        Path to the VTK file.

    fields : list of str
        Names of the fields to extract.
        For example:
            "polarization_degree"
            "polarization_angle"
            "stokes_q"
            "stokes_u"

    Returns
    -------
    data : dict
        Dictionary containing:
            x, y, z
            r
            theta
            phi
            field
    """

    mesh = pv.read(vtk_file)

    # Observer positions
    xyz = np.asarray(mesh.points)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)

    # Spherical coordinates
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)

    # Extract requested quantity
    data = {}
    for f in fields:
        if f in mesh.point_data:
            values = np.asarray(mesh.point_data[f])
        elif f in mesh.cell_data:
            values = np.asarray(mesh.cell_data[f])
        else:
            raise KeyError(
                f"Field '{f}' not found.\n"
                f"Available point fields: {list(mesh.point_data.keys())}\n"
                f"Available cell fields: {list(mesh.cell_data.keys())}")

        data[f] = values

    return {
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "theta": theta,
            "phi": phi,
            **data
            }

if __name__ == "__main__":
    m = 5
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'HiResNewAMR' 
    snap = 200
    generat = 2500
    choice = 'split_stream'
    snr_thresh = 5

    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    params = [Mbh, Rstar, mstar, beta]
    things = orb.get_things_about(params)
    tfallback = things['t_fb_days']
    Rt = things['Rt']
    Rt_cgs = Rt * prel.Rsol_cgs
    tfallback_cgs = tfallback * 24 * 3600

    groups = np.arange(10)
    path_file =  f'{abspath}/TDE/{folder}/Polarization/{snap}/tde_{generat}polarization{snap}.vtk'
    f_edges_ev = np.loadtxt(f'{abspath}/src/Opacity/fromRICH/MG/frequency_edges.txt') # [Hz] frequency range for each group. shape: (10, 2)
    f_edges = f_edges_ev * prel.ev_toHz
    f_mid_points = (f_edges[:-1] + f_edges[1:]) / 2
    if check == 'MG':
        time = np.loadtxt(f'{abspath}/data/{folder}/snap_{snap}/Time_snap_{snap}.txt')
        time = time*prel.tsol_cgs/tfallback_cgs
    if check == 'HiResNewAMR':
        folder_fld = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiResNewAMR'
        data_fld = np.loadtxt(f'{abspath}/data/{folder_fld}/HiResNewAMR_red.csv', delimiter=',', dtype=float)
        snaps_fld, tfb_fld, Lum_fld = data_fld[:, 0], data_fld[:, 1], data_fld[:, 2]
        snaps_fld = snaps_fld.astype(int)
        idx_time_fld = np.argmin(np.abs(snaps_fld - snap))
        time = tfb_fld[idx_time_fld]
    print(time)

    folder_fld = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiResNewAMR'
    data_fld = np.loadtxt(f'{abspath}/data/{folder_fld}/HiResNewAMR_red.csv', delimiter=',', dtype=float)
    snaps_fld, tfb_fld, Lum_fld = data_fld[:, 0], data_fld[:, 1], data_fld[:, 2]
    snaps_fld, tfb_fld, Lum_fld = sort_list([snaps_fld, tfb_fld, Lum_fld], tfb_fld, unique=True) 
    snaps_fld = snaps_fld.astype(int)
    idx_time_fld = np.argmin(np.abs(tfb_fld - time))
    snap_fld = snaps_fld[idx_time_fld]
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
                fields=[f"forward_polarization_degree",
                        f"forward_polarization_snr",
                        "forward_polarization_angle",
                        "forward_luminosity",
                        "photosphere_grey_radius_tau_total"])   
    x = data["x"]
    y = data["y"]
    z = data["z"]
    theta = data["theta"]
    phi = data["phi"]
    indices_sec, labels_sec, color_obs = split_cells(x, y, z, choice=choice)
    Lum = data["forward_luminosity"]
    r_ph = data["photosphere_grey_radius_tau_total"]
    P_gray = data["forward_polarization_degree"]
    snr_P_gray = data["forward_polarization_snr"] 
    P_angle = data["forward_polarization_angle"]
    P_gray_med_idx = np.zeros(len(indices_sec))
    snr_P_gray_med_idx = np.zeros(len(indices_sec))
    Lum_med_idx = np.zeros(len(indices_sec)) 
    rph_med_idx = np.zeros(len(indices_sec))
    for i, indices in enumerate(indices_sec):
        P_gray_med_idx[i] = np.median(P_gray[indices])
        snr_P_gray_med_idx[i] = np.median(snr_P_gray[indices])
        Lum_med_idx[i] = np.median(Lum[indices])
        rph_med_idx[i] = np.median(r_ph[indices])

    figc, axC = plt.subplots(1, 1, figsize=(10, 7))
    for i in range(len(indices_sec)):
        axC.scatter(i, rph_med_idx[i]/prel.Rsol_cgs/r_ph_fld_med_idx[i], color=color_obs[i], s = 80, marker='o', label=r'r$_{\rm ph}$' if i == 1 else "")
        axC.scatter(i, len(Lum) * Lum_med_idx[i]/Lum_fld_med_idx[i], color=color_obs[i], s = 80, marker='x', label=f'Lum' if i == 1 else "")
    # axC.loglog()
    axC.grid()
    axC.set_ylabel('MC/fld')
    axC.legend(fontsize=18)
                                    
    P_wave = []
    snr_P_wave = []
    Lum_wave = []
    rph_wave = []
    figmesh = plt.figure(figsize=(40,15))
    for group in groups:
        data = extract_poli(
            path_file,
            fields=[f"photosphere_g{group}_radius_thermalization", 
                    f"photosphere_g{group}_radius_tau_total",
                    f"forward_polarization_degree_group_{group:03d}",
                    f"forward_polarization_snr_group_{group:03d}",
                    f'forward_group_{group}_luminosity'])         
        
        r_th = data[f"photosphere_g{group}_radius_thermalization"]
        r_tau = data[f"photosphere_g{group}_radius_tau_total"]
        # check if any ratio r_th/r_tau > 1
        if np.any(r_th/r_tau > 1):
            idx_wrong = np.where(r_th/r_tau > 1)[0]
            print(f"Warning: r_th/r_tau > 1 for group {group} at indices {idx_wrong}")
            print(f"r_th/r_tau values: {r_th[idx_wrong] / r_tau[idx_wrong]}")
        P = data[f"forward_polarization_degree_group_{group:03d}"]
        snr_P = data[f"forward_polarization_snr_group_{group:03d}"]
        Lum = data[f'forward_group_{group}_luminosity']
        r_th = data[f"photosphere_g{group}_radius_thermalization"]
        r_ph = data[f"photosphere_g{group}_radius_tau_total"]
        # ratioR = r_th/r_ph

        if group == 2:
            longitude_moll = phi 
            latitude_moll = np.pi/2 - theta
            nlon = 360
            nlat = 180
            lon_grid = np.linspace(longitude_moll.min(), longitude_moll.max(), nlon)
            lat_grid = np.linspace(latitude_moll.min(), latitude_moll.max(), nlat)
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            P_grid = griddata(
                points=(longitude_moll, latitude_moll),
                values=P,
                xi=(lon_mesh, lat_mesh),
                method='linear')

            L_grid = griddata(
                    points=(longitude_moll, latitude_moll),
                    values=Lum,
                    xi=(lon_mesh, lat_mesh),
                    method='linear')

            snr_P_grid = griddata(
                            points=(longitude_moll, latitude_moll),
                            values=snr_P,
                            xi=(lon_mesh, lat_mesh),
                            method='linear')
            
            gs = gridspec.GridSpec(2, 2,  height_ratios=[1, .05], hspace=0.1, wspace = 0.2) 
            axPmesh = figmesh.add_subplot(gs[0, 0], projection='mollweide')
            img = axPmesh.pcolormesh(lon_mesh, lat_mesh, P_grid, cmap='viridis', vmin = 1e-3, vmax = 5e-2)
            snr_contour = axPmesh.contour(
                lon_mesh, lat_mesh, np.ma.masked_invalid(snr_P_grid),
                levels=[snr_thresh], colors='white', linewidths=4)
            axPmesh.clabel(
                snr_contour, fmt={snr_thresh: f'S/N $\geq$ {snr_thresh}'},
                inline=True, fontsize=20)
            axPmesh.grid()
            cbar_ax = figmesh.add_subplot(gs[1, 0]) 
            cbar = figmesh.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'P')
            cbar.ax.tick_params(which='major',length = 5)
            cbar.ax.tick_params(which='minor',length = 3)

            axLmesh = figmesh.add_subplot(gs[0, 1], projection='mollweide')
            img = axLmesh.pcolormesh(lon_mesh, lat_mesh, L_grid, cmap='viridis', norm = colors.LogNorm(vmin=1e36, vmax=4e37))
            axLmesh.grid()
            cbar_ax = figmesh.add_subplot(gs[1, 1]) 
            cbar = figmesh.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'Luminosity [erg/s]')
            cbar.ax.tick_params(which='major',length = 5)
            cbar.ax.tick_params(which='minor',length = 3)
            
            figmesh.suptitle(f'Polarization map for group {group} at t={time:.2f}' + r' t$_{{\rm fb}}$' + f' ({generat} GEN)', fontsize = 20, y = 0.85)
            # plt.tight_layout()

        P_med_idx = np.zeros(len(indices_sec))
        snr_P_med_idx = np.zeros(len(indices_sec))
        Lum_med_idx = np.zeros(len(indices_sec))
        rph_med_idx = np.zeros(len(indices_sec))
        
        for i, indices in enumerate(indices_sec):
            P_sector = P[indices]
            snr_P_sector = snr_P[indices]
            P_tocare = P_sector[snr_P_sector > snr_thresh]
            snr_P_tocare = snr_P_sector[snr_P_sector > snr_thresh]
            P_med_idx[i] = np.median(P_tocare) if len(P_tocare) > 0 else np.nan
            snr_P_med_idx[i] = np.median(snr_P_tocare) if len(snr_P_tocare) > 0 else np.nan
            Lum_med_idx[i] = np.median(Lum[indices])
            rph_med_idx[i] = np.median(r_ph[indices])
        P_wave.append(P_med_idx)
        snr_P_wave.append(snr_P_med_idx)
        Lum_wave.append(Lum_med_idx)
        rph_wave.append(rph_med_idx)

    P_wave = np.transpose(P_wave)# shape: (len(groups), len(indices_sec))
    snr_P_wave = np.transpose(snr_P_wave)
    Lum_wave = np.transpose(Lum_wave)
    rph_wave = np.transpose(rph_wave)
    figP, ((axP, axLum), (axSNR, axR)) = plt.subplots(2, 2, figsize=(20, 15)) 
    for i, indices in enumerate(indices_sec):
        axP.plot(f_mid_points, P_wave[i], '-o', label=f'{labels_sec[i]}', color=color_obs[i])
        axSNR.plot(f_mid_points, snr_P_wave[i],  '-o', color=color_obs[i])
        axLum.plot(f_mid_points, Lum_wave[i], '-o', label=f'{labels_sec[i]}', color=color_obs[i])
        axR.plot(f_mid_points, rph_wave[i]/Rt_cgs, '-o', label=f'{labels_sec[i]}', color=color_obs[i])
        # axP.axhline(P_gray_med_idx[i], ls = '--', color=color_obs[i])
        # axSNR.axhline(snr_P_gray_med_idx[i], ls = '--', color=color_obs[i])
    axP.legend(fontsize=15)
    axP.set_ylabel(r'median $P$')
    axSNR.set_ylabel(r'median $S/N$ for $P_{\rm S/N>' + str(snr_thresh) + '}$')
    axLum.set_ylabel(r'median $L$ [erg/s]')
    axR.set_ylabel(r'median $r_{\rm ph}/r_{\rm t}$')
    axP.set_ylim(5e-3, 0.5)
    axSNR.set_ylim(9e-1, 40)
    for ax in [axP, axSNR, axLum, axR]:
        add_spectral_regions(ax, x_axis = 'wavelength', text_band=True if ax == axP else False)
        ax.tick_params(axis='both', which='major', length=8)
        ax.tick_params(axis='both', which='minor', length=6)
        # if ax == axR:
        #     ax.set_xscale('log')
        # else:
        ax.loglog()
        ax.grid() 
        ax.set_xlim(f_edges[0], f_edges[-1])
        top = ax.twiny() 
        frequencies = np.logspace(np.log10(f_edges[0]), np.log10(f_edges[-1]), 5)
        wavelengths = prel.c_cgs * 1e8 / frequencies
        top.set_xticks(wavelengths, [f"{x:.2f}" for x in wavelengths])
        top.set_xlim(wavelengths.max(), wavelengths.min())
        top.set_xscale("log")
        if ax in [axP, axLum]:
            ax.set_xlabel(r'', fontsize=30)
            top.set_xlabel(r"$\lambda\;(\AA)$", fontsize=30)
    figP.suptitle(f't={time:.2f}' + r't$_{\rm fb}$,' + f' {generat} GEN', fontsize = 20)
    figP.tight_layout()



    