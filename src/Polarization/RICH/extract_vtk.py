#!/usr/bin/env python3
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
# import Utilities.prelude as prel


def extract_poli(
    vtk_file,
    field="polarization_degree",
):
    """
    Extract observer positions and a polarization-related field
    from a RICH post-processing VTK file.

    Parameters
    ----------
    vtk_file : str
        Path to the VTK file.

    field : str
        Name of the field to extract.
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
    theta = np.arccos(
        np.clip(z / r, -1.0, 1.0)
    )

    phi = np.arctan2(y, x)

    # Put phi in [0, 2pi)
    phi = np.mod(phi, 2.0 * np.pi)

    # Extract requested quantity
    if field in mesh.point_data:
        values = np.asarray(mesh.point_data[field])

    elif field in mesh.cell_data:
        values = np.asarray(mesh.cell_data[field])

    else:
        raise KeyError(
            f"Field '{field}' not found.\n"
            f"Available point fields: {list(mesh.point_data.keys())}\n"
            f"Available cell fields: {list(mesh.cell_data.keys())}"
        )

    return {
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "theta": theta,
        "phi": phi,
        field: values,
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
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

    path_file =  f'{abspath}/TDE/{folder}/Polarization/{snap}/tde_{generat}polarization{snap}.vtk'
    

    # P = data["forward_polarization_degree"]
    # theta = data["theta"]
    # phi = data["phi"]

    plt.figure(figsize=(8, 6))
    for g in range(0, 9):
        dataTh = extract_poli(
            path_file,
            field=f"photosphere_g{g}_radius_thermalization")
        
        r_th = dataTh[f"photosphere_g{g}_radius_thermalization"]

        datatau = extract_poli(
                    path_file,
                    field=f"photosphere_g{g}_radius_tau_total")
        r_tau = datatau[f"photosphere_g{g}_radius_tau_total"]


        plt.plot(np.arange(len(r_th)), r_th/r_tau, label=f'group {g}')
    plt.legend()
    plt.tick_params(axis='both', which='major', length=8)
    plt.tick_params(axis='both', which='minor', length=6)
    plt.yscale('log')