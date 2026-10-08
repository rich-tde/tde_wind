# Code (1950) Polarization with Non-Constant Flux

This script extends the plane-parallel polarization treatment based on **Code (1950)** to an atmosphere with a prescribed, depth-dependent thermal structure and therefore **non-constant radiative flux**.

The original constant-flux `Code1950Atmosphere` solver is retained and used for its angular scattering kernel. The new `Code1950NonConstantFluxAtmosphere` class instead solves the transfer problem through a supplied atmosphere described by optical depth, temperature, and absorption/scattering opacities.

The script also provides routines to combine the local solutions from many photospheric patches and compute the net Stokes parameters and polarization for arbitrary observer directions.

## Physical assumptions

The non-constant-flux solver assumes:

- stationary radiative transfer: within each individual post-processing calculation, the radiation field is assumed to adjust instantaneously to the current structure.
NB: the same is done in RICH post-porcessing, which doesn't evolve the fluid
- gray opacities;
- a local plane-parallel atmosphere;
- axial symmetry around the local normal;
- LTE thermal emission;
- coherent Thomson scattering;
- no Compton frequency redistribution.

The local normal of each photospheric patch is approximated using the direction of the radiative flux at the photosphere,


## Main transfer equation

The absorption fraction is

```text
lambda = alpha_abs / (alpha_abs + alpha_scatter).
```

The original constant-flux thermal term is replaced by the local LTE source

```text
lambda(tau) B[T(tau)] / 2,
```

where

```text
B(T) = sigma_SB T^4 / pi.
```

Schematically, the transfer system solved is

```text
mu dy/dtau = y - (1-lambda) C_sc y - lambda B/2.
```

The conservative scattering kernel is inherited from the original `Code1950Atmosphere`.

## Numerical method

`Code1950NonConstantFluxAtmosphere` solves the intensities at all depth nodes simultaneously.

The method uses:

- Gauss-Legendre angular quadrature;
- a refined optical-depth grid;
- short-characteristics integration;
- exact exponential attenuation over each depth interval;
- piecewise-linear source interpolation;
- a sparse global linear system.

The depth resolution is controlled primarily by

```python
subdivisions
max_delta_tau
```

while the angular resolution is controlled by

```python
n
```

Convergence should be checked by increasing `subdivisions`, decreasing `max_delta_tau`, and increasing `n`.

## Input atmosphere

For one photospheric patch, the basic input arrays are

```python
tau
temperature
alpha_abs
alpha_scatter
```

All arrays must describe the same depth samples.

### Optical depth

`tau` must:

- be one-dimensional;
- start at `tau = 0`;
- increase inward;
- be strictly increasing.

### Temperature

`temperature` is the gas temperature in Kelvin.

The LTE source function is constructed internally from

```python
B = sigma_SB * temperature**4 / pi
```

### Opacities

The solver can receive either

```python
lam
```

directly or

```python
alpha_abs
alpha_scatter
```

from which it computes

```python
lam = alpha_abs / (alpha_abs + alpha_scatter)
```

The absorption and scattering coefficients must use consistent units.

### Radiation energy density

`radiation_energy_density` is optional.

It is stored for comparison with the radiation field reconstructed by the transfer solver. It is **not** imposed as an additional radiation field.

The solver computes

```text
u_rad = 4 pi J / c
```

from its own solution.

## Boundary conditions

A bottom boundary intensity must be explicitly supplied:

```python
bottom_intensity
```

It represents the total unpolarized intensity entering upward from the deepest optical-depth point.

It can be either:

```python
bottom_intensity = scalar
```

or a function of positive `mu`.

For example, an isotropic estimate based on the simulation radiation energy density is

```python
I_bottom = c * u_rad[-1] / (4*pi)
```

which is the choice used in the example at the bottom of the script.

An isotropic LTE boundary could instead use

```python
I_bottom = sigma_SB * T[-1]**4 / pi
```

but no boundary condition is selected automatically.

The optional

```python
top_intensity
```

specifies isotropic downward incident radiation at the surface and defaults to zero.

## Basic usage for one patch

The convenience function

```python
solve_simulation_patch(...)
```

constructs a non-constant-flux atmosphere directly from simulation arrays.

Example:

```python
atm = solve_simulation_patch(
    tau=tau,
    temperature=temp,
    alpha_abs=alpha_abs,
    alpha_scatter=alpha_scatter,
    bottom_intensity=I_bottom,
    radiation_energy_density=radden,
    n=3,
    subdivisions=4,
    max_delta_tau=0.05,
)
```

### Emergent radiation

For a viewing cosine `mu`,

```python
Il, Ir, I, Q, P = atm.emergent(mu)
```

returns the two orthogonal linear-polarization intensities and

```text
I = Il + Ir
Q = Ir - Il
P = Q / I
```

Here `P` is signed according to the adopted local Stokes convention.

### Radiation moments

Use

```python
mom = atm.moments()
```

to obtain

```python
mom["tau"]
mom["J"]
mom["F"]
mom["B"]
mom["lam"]
mom["u_rad"]
mom["dF_dtau_exchange"]
```

If simulation radiation energy density was supplied, the dictionary also contains

```python
mom["u_rad_sim"]
```

The flux is calculated as

```text
F = 2 pi integral mu (Il + Ir) dmu.
```

It is therefore an **output**, rather than an imposed constant.

The energy-exchange quantity is

```text
dF/dtau = 4 pi lambda (J - B).
```

## Net polarization over the photosphere

`compute_polarization_code(...)` combines the radiation from all visible photospheric patches.

For each patch, the local flux direction is used as the approximate surface normal:

```python
Fhat = F / |F|
```

For observer direction `n_obs`, the viewing cosine is

```python
eta = Fhat @ n_obs
```

and only patches satisfying

```python
eta > 0
```

are visible.

Each local atmosphere is evaluated at its corresponding `eta`.

The local Stokes `Q` is then rotated into a common observer-plane basis to obtain

```python
Q_sky
U_sky
```

before summing over the photosphere.

The projected patch weight is

```text
w_obs = dA * eta.
```

The final quantities are

```text
I_tot = sum(w_obs I)
Q_tot = sum(w_obs Q_sky)
U_tot = sum(w_obs U_sky)
```

and

```text
P_tot = sqrt(Q_tot^2 + U_tot^2) / I_tot.
```

## Patch areas

If `dA` is supplied, each photospheric contribution is weighted by its physical surface area.

In the example script, the photosphere is sampled in equal HEALPix solid-angle elements,

```python
dOmega = 4*pi / Npatch
```

and the area is estimated from

```python
dA = r**2 * dOmega / abs(n_hat . r_hat)
```

with the local normal approximated by the flux direction.

If

```python
dA = None
```

all patches are assigned equal area.

## Applying the solver to all simulation patches

The high-level function

```python
compute_simulation_nonconstant_flux(...)
```

constructs one atmosphere per photospheric patch and computes the polarization for every supplied observer.

Example:

```python
results, atmospheres = compute_simulation_nonconstant_flux(
    tau_profiles=tau_profiles,
    temperature_profiles=T_profiles,
    alpha_abs_profiles=alpha_abs_profiles,
    alpha_scatter_profiles=alpha_scatter_profiles,
    radiation_energy_density_profiles=u_rad_profiles,
    bottom_intensities=I_bottom,
    Fx=Fx_ph,
    Fy=Fy_ph,
    Fz=Fz_ph,
    observers_xyz=observers_xyz,
    dA=dA,
    n=3,
    subdivisions=4,
    max_delta_tau=0.05,
)
```

The returned dictionary contains

```python
results["P"]          # net polarization fraction for each observer
results["I"]          # total Stokes I
results["Q"]          # total Stokes Q
results["U"]          # total Stokes U
results["F_surface"]  # transfer-solution flux at tau = 0 for every patch
```

The returned `atmospheres` list contains the complete transfer solution for every photospheric patch.

No patch is deliberately omitted and no local flux renormalization is applied in the non-constant-flux calculation.

## Simulation workflow in this script

When executed directly, the script performs the following steps:

1. Defines the TDE simulation and snapshot.
2. Constructs the HEALPix observer directions.
3. Loads the photospheric positions and radiative-flux vectors.
4. Estimates the surface area `dA` of every photospheric patch.
5. Loads the optical-depth profiles stored in `flux_along_patch<snapshot>.npz`.
6. Extracts `tau`, temperature, radiation energy density, absorption opacity, and scattering opacity for every patch.
7. Reverses the profiles so that optical depth increases inward.
8. Defines the bottom intensity from the deepest simulation radiation energy density.
9. Constructs one non-constant-flux atmosphere per photospheric patch.
10. Computes net `I`, `Q`, `U`, and `P` for all HEALPix observers.
11. Interpolates the polarization values onto a regular longitude-latitude grid.
12. Produces a Mollweide map of the polarization degree.

## Expected simulation files

The example expects a photosphere file of the form

```text
data/<folder>/photo/<check>_photo<snap>.npz
```

containing at least

```text
x
y
z
Fx
Fy
Fz
```

and a ray/profile file

```text
data/<folder>/flux_along_patch<snap>.npz
```

where each patch contains

```text
tau
temp
radden
alpha_abs
alpha_scatter
```

## Dependencies

The script requires:

```text
numpy
scipy
matplotlib
healpy
```

and project-specific modules

```python
from Utilities.isalice import isalice
from Utilities import prelude as prel
from src.Polarization.code1950 import Code1950Atmosphere
```

The project root must therefore be available on `PYTHONPATH`, or added to `sys.path`.

## Important normalization distinction

The original constant-flux `Code1950Atmosphere` may be constructed with a unit flux and subsequently scaled by the local simulation `|F|`.

That is controlled by

```python
atmospheres_are_per_unit_flux=True
```

inside `compute_polarization_code`.

The non-constant-flux atmosphere instead produces **physically normalized intensities from the prescribed thermal structure and boundary conditions**. Therefore it must be used with

```python
atmospheres_are_per_unit_flux=False
```

and must not be multiplied again by the simulation flux magnitude.

The code explicitly checks for this case.

## Output interpretation

`P` is dimensionless.

The returned `I`, `Q`, and `U` are area-weighted photospheric Stokes sums under the conventions used by this script. They are **not directly the observed flux at Earth**; conversion to an observer flux requires the appropriate distance factor.

## Notes

The calculation treats each photospheric patch as an independent local plane-parallel atmosphere. The 3D geometry enters through:

- the photospheric patch positions;
- the local flux directions;
- the patch areas;
- the observer direction;
- the rotation of each local polarization basis onto a common sky basis.

The vertical atmospheric structure enters through the individual `tau`, temperature, absorption, scattering, and optional radiation-energy-density profiles.
