# TDE gray/MG luminosity and polarization

This run post-processes one TDE snapshot twice from the same FLD-normalized
thermalization surface:

- multigroup IMC with the current 10-group `data/STA/MG` tables;
- gray IMC with the matching `data/STA` tables;
- Stokes Q/U and polarization degree for both passes;
- bolometric and per-group luminosities, generation error estimates,
  photospheres, adaptive diagnostics, and flux-source comparison data.

The MPI Monte Carlo manager uses two-sided point-to-point communication. This
is the requested lower-memory non-RDMA path. The RDMA manager remains selectable
with `--transport.communication rdma` if a later platform requires it.

## Build

Load the compiler and MPI environment provided by the machine. Then, from the
RICH repository root, build with that environment's MPI build name:

```bash
RICH_BUILD_NAME=yourMPIBuild 
# substitute "$RICH_BUILD_NAME" with gnuReleaseMPI
# test_name tell you in which folder look for the test.cpp
./build_rich.sh "$RICH_BUILD_NAME" \
  --test_name=imc_postprocess_tde_gray_mg_polarization \
  --energy_groups_num=10 \
  --montecarlo-polarization
```

The compile-time group count must remain 10 for the current STA table, whose
`frequency_edges.txt` contains 11 boundaries.

## Submit

Edit only `LEARNING_ITERATIONS` and `STATISTICS_ITERATIONS` in `submit.sh`.
Learning generations update the adaptive source/observer/group proposals and
are discarded. Statistics generations use the learned proposal and are the
only generations accumulated into reported means and standard errors.

Export absolute paths to the snapshot and the executable produced by the build:

```bash
cd runs/imc_postprocess_tde_gray_mg_polarization
export RICH_POSTPROCESS_SNAPSHOT=/home/pmartire/tde_wind/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/snap_151/snap_151.h5
export RICH_EXECUTABLE=/home/pmartire/RICH/build/gnuReleaseMPI/rich
sbatch submit.sh
```

The submit file intentionally contains no partition, node count, task count,
hardware constraint, excluded-node list, MPI installation path, or fabric
setting. Supply resource requests using the `sbatch` options or site wrapper
appropriate to the machine. Inside Slurm the script uses `srun`; when invoked
directly it uses `mpirun`, or `mpiexec` if `mpirun` is unavailable.
Set `RICH_MPI_LAUNCHER` to a launcher executable or site wrapper when the
automatic choice is not appropriate for that MPI installation.

The run writes:

- `output/tde_gray_mg_polarization.h5`, with `/passes/forward` (MG),
  `/passes/grey`, `/comparison`, `/photosphere`, and full effective config;
- `output/tde_gray_mg_polarization.vtk`, with `forward_*` and `grey_*`
  observer fields in one sphere map.

## Sphere plots

Use the included entry point to inspect and plot any combined VTK field:

Use a Python 3 environment with NumPy, SciPy, and Matplotlib available.

``` From MAOR:
bash
python3 plot_sphere.py \
  output/tde_gray_mg_polarization.vtk --list-fields

python3 plot_sphere.py \
  output/tde_gray_mg_polarization.vtk \
  --field forward_luminosity --scale log

python3 plot_sphere.py \
  output/tde_gray_mg_polarization.vtk \
  --field grey_polarization_degree
```

``` WHAT WE DO
python3 plot_sphere.py /Users/paolamartire/shocks/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/Polarization/tde_gray_mg75polarization151.vtk --list-fields

python3 plot_sphere.py \
  /Users/paolamartire/shocks/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/Polarization/tde_gray_mg75polarization151.vtk \
  --field forward_luminosity --scale log

python3 plot_sphere.py \
  /Users/paolamartire/shocks/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/Polarization/tde_gray_mg75polarization151.vtk \
  --field grey_polarization_degree

``` 