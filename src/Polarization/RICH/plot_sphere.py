#!/usr/bin/env python3
"""Plot any gray or multigroup observer field on a Mollweide sphere map.

This entry point reuses the maintained generic VTK sphere plotter from the
original imc_postprocess_tde run. Examples:

    python3 plot_sphere.py output/tde_gray_mg75polarization.vtk --list-fields
    python3 plot_sphere.py output/tde_gray_mg75polarization.vtk \
        --field forward_luminosity --scale log
    python3 plot_sphere.py output/tde_gray_mg75polarization.vtk \
        --field grey_polarization_degree
"""

from pathlib import Path
import runpy


# From MAor
# PLOTTER = (
#     Path(__file__).resolve().parent.parent
#     / "imc_postprocess_tde"
#     / "plot_moli.py"
# )
PLOTTER = Path(__file__).resolve().parent / "plot_moli.py"

if not PLOTTER.is_file():
    raise SystemExit(f"Shared sphere plotter not found: {PLOTTER}")

runpy.run_path(str(PLOTTER), run_name="__main__")
