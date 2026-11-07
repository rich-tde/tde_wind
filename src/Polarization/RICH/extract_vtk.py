#!/usr/bin/env python3
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from src.Wind.Mdot_Rfixed_sec import split_cells
import Utilities.prelude as prel


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
    # Put phi in [0, 2pi)
    phi = np.mod(phi, 2.0 * np.pi)

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
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'MG' 
    snap = 429
    generat = 800
    choice = 'split_stream'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    # tfb = np.loadtxt(f'{abspath}/TDE/{folder}/{snap}/tfb_{snap}.txt')
    path_file =  f'{abspath}/TDE/{folder}/Polarization/{snap}/tde_{generat}polarization{snap}.vtk'

    figR, axR = plt.subplots(figsize=(8, 6))
    figP, (axP, axSNR) = plt.subplots(2, 1, figsize=(8, 12))
    for g in range(0, 9):
        data = extract_poli(
            path_file,
            fields=[f"photosphere_g{g}_radius_thermalization", 
                   f"photosphere_g{g}_radius_tau_total",
                   f"forward_polarization_degree_group_{g:03d}",
                   f"forward_polarization_snr_group_{g:03d}"])
        
        x = data["x"]
        y = data["y"]
        z = data["z"]
        theta = data["theta"]
        phi = data["phi"]
        r_th = data[f"photosphere_g{g}_radius_thermalization"]
        r_tau = data[f"photosphere_g{g}_radius_tau_total"]
        P = data[f"forward_polarization_degree_group_{g:03d}"]
        snr_P = data[f"forward_polarization_snr_group_{g:03d}"]
        indices_sec, labels_sec, color_obs = split_cells(x, y, z, choice=choice)
        P_med = np.zeros(len(indices_sec))
        snr_P_med = np.zeros(len(indices_sec))
        
        for i, indices in enumerate(indices_sec):
            P_med[i] = np.median(P[indices])
            snr_P_med[i] = np.median(snr_P[indices])

        axR.plot(np.arange(len(r_th)), r_th/r_tau, label=f'group {g}', alpha=0.7)
        axP.scatter(np.arange(len(indices_sec)), P_med, label=f'group {g}')
        axSNR.scatter(np.arange(len(indices_sec)), snr_P_med, label=f'group {g}')

    axR.set_ylabel(r'$r_{\rm th} / r_{\rm ph}$')
    axP.set_xticks(np.arange(len(indices_sec)))
    # axP.set_xticklabels(labels_sec, rotation=45, ha='right', fontsize=10)
    axP.set_xticklabels('')
    axP.set_ylabel(r'median $P$')
    axP.set_ylim(6e-3, 0.8)
    axSNR.set_xticks(np.arange(len(indices_sec)))
    axSNR.set_xticklabels(labels_sec, rotation=45, ha='right', fontsize=10)
    axSNR.set_ylabel(r'median $S/N$')
    axSNR.set_ylim(1, 30)
    for ax in [axR, axP, axSNR]:
        ax.legend()
        ax.tick_params(axis='both', which='major', length=8)
        ax.tick_params(axis='both', which='minor', length=6)
        ax.set_yscale('log')
        if ax != axSNR:
            ax.set_title(f'Snap {snap}, gen {generat}')
    figP.tight_layout()