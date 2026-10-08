# TDE gray/MG  polarization

This run post-processes one TDE snapshot producing:
- Stokes Q/U and polarization degree for both gray (`data/STA` tables) and MG (10-group `data/STA/MG` tables);
- bolometric and per-group luminosities, generation error estimates,
  photospheres, adaptive diagnostics, and flux-source comparison data.

The MPI Monte Carlo manager uses two-sided point-to-point communication. This
is the requested lower-memory non-RDMA path. The RDMA manager remains selectable with `--transport.communication rdma` if a later platform requires it.

## Build

Load the compiler and MPI environment provided by the machine. Then, from the
RICH repository root, build with that environment's MPI build name:

```bash
RICH_BUILD_NAME=yourMPIBuild # substitute "$RICH_BUILD_NAME" with gnuReleaseMPI
# test_name tell you in which folder look for the test.cpp
./build_rich.sh "$RICH_BUILD_NAME" \
  --test_name=imc_postprocess_tde_gray_mg_polarization \
  --energy_groups_num=10 \
  --montecarlo-polarization
```

The compile-time group count must remain 10 for the current STA table, whose
`frequency_edges.txt` contains 11 boundaries.

## Overview of the process
512 observers
↗    ↑    ↑    ↖
photons escape
↑
$\tau_{\rm cell}\leq$ 5 region: scattering - absorption (Monte Carlo)
↑ 
$\tau_{\rm cell}>$ 5: DDMC transport 
thermal volume emission can occur (from cells whos thermalization radius is more superficial)
↑ common deep thermalization ($\tau_{\rm therm}\approx5$)
Planckian radiation from deep surface

### How this happens
submit.sh
   ↓
mpirun / srun <RICH executable>
   ↓
runs/PaolaPolarization/test.cpp
   ↓
source/3D/radiation/postprocess/IMCPostProcess.hpp
   ↓
source/3D/radiation/postprocess/PostProcessApplication.cpp
which enables the polarization computation
   ↓
Grey / multigroup calculation
source/3D/radiation/postprocess/GreyCalculation.cpp
   ↓
source/3D/radiation/RadiationIMC.hpp that wraps in STORM
the polarization calculation which happens in 
monte/radiation/Polarization.hpp
(activated by monte/radiation/RadiationIMC.hpp)
and forwards the photon Stokes values to SphericalObserver
   ↓
IMC / DDMC transport engine
source/monte/radiation/ddmc/DDMCEngine


### Setup in test.cpp
Some notes:
- $\underline {\rm Thickness}$:
      1) $\tau_{\rm cell}$ is the thickness of a cell, defined in source/monte/radiation/ddmc/DDMCEngineas as
      \[\tau_{\rm cell,g}=\sigma_{T,g}\,\ell_{\rm chord},\]
      where \(\sigma_{T,g}\) is the relevant extinction/transport coefficient for energy group \(g\),
      $\ell_{\rm chord}$ is the characteristic path length through that cell.
      Its threshold for the DDMC is set in config.transport.ddmcMinCellOpticalDepth. 
      2) $\tau_{\rm therm}$ is the cumulative thermalization depth toward the exterior (i.e. $\tau_{\rm therm}(s) =\int_s^{\rm outside}\alpha_{\rm therm}(s')\,ds'$).
- $\underline {\rm MC \,\,types}$:
      1) IMC (Implicit Monte Carlo) approach: rather than explicitly transferring all that energy back and forth between radiation and matter during a timestep, treat some of those absorption–re-emission cycles as if the packet had scattered. The photon packet therefore survives instead of being destroyed and replaced. The Fleck factor \(f\) determines how much is treated as real absorption versus effective scattering. Schematically: \[\alpha_{\rm abs} =f\alpha_{\rm abs}+ (1-f)\alpha_{\rm abs}.\]
      So \(f=1\) (which is what we use) means all absorption is real absorption, no IMC.
      2) For $\tau\geq\bar\tau$  (for us: 5), we switch to Discrete Diffusion Monte Carlo (DDMC): switch to a diffusion approximation in optically thick cells, which is much faster than random-walking packets through them.
      3) We build an internal surface from which radiation is injected: we don't want Monte-Carlo packets to random-walk from extremely deep inside an optically thick simulation. Instead, we find a sufficiently deep surface and replace the deeper radiation field with a boundary condition, considering that below this surface radiation is sufficiently thermalized that we can treat the radiation  as approximately Planckian (so there no MC transport).


## Polarization.hpp
It defines how a packet’s Stokes polarization state is initialized, rotated, updated by Thomson scattering.
Once the interaction was identified as a scattering in RadiationIMC, it chooses where the photon goes (samplePolarizedThomsonDirection()) and computes what happens to Q and U (applyThomsonScatter()).

### Functions

#### InitializeIfNeeded(p):
If this photon doesn't have polarization information yet, initialize it as unpolarized and creates a reference basis perpendicular to its direction through choosePerpendicularBasis(direction)
NB: it gives the normalized Stokes parameters q=Q/I and u=U/I.

####  choosePerpendicularBasis(direction)
It normalizes the propagation direction $\mathbf k$ and chooses some helper vector that is not parallel to it. It then removes the component parallel to $\mathbf k$

#### ProjectBasisToDirection(basis, direction)
Takes a ${\emph basis}$ vector and makes sure it is perpendicular to a ${\emph direction}$, removing the component of the former parallel to the latter.

#### ProjectToBasis(p, direction, basis):
Takes ${\emph basis}$ vector and the polarization vector of $p$ and makes sure they are perpendicular to a propagation ${\emph direction}$, removing the component of the formers parallel to the latter.


#### ApplyThomsonScatter(...) 
Given the old \(Q,U\) for a photon packet which was moving along $\mathbf{k}_{\rm in}$, calculate its new \(Q,U\) after it is Thomson-scatters into $\mathbf{k}_{\rm out}$.
Iter:
1) If the packet has never had its polarization, initialize
2) Normalize the incoming and outgoing directions.
NB: The second argument to SafeNormalize is just a fallback if the vector has essentially zero length (z axis for incoming, $\mathbf{k}_{\rm in}$ for $\mathbf{k}_{\rm out}$)
3) Build the scattering plane from the incoming and outgoing photon directions ($\mathbf n_{\rm plane}=\hat{\mathbf k}_{\rm in} \times\hat{\mathbf k}_{\rm out}$, while $\hat{\mathbf k}_{\rm in}\cdot\hat{\mathbf k}_{\rm out} = \cos\theta=\mu$).
If $|\mathbf k_{\rm in}\times\mathbf k_{\rm out}|\approx0$, it's a forward/backward scattering and the code updates the direction, adjusts the existing polarization basis so that it exist and it remains perpendicular to the new direction, and stop executing the function
4) Normalize $\mathbf n_{\rm plane}$ and use it as direction reference for the polarization axis
5) Rotate the old \(Q,U\) into the scattering-plane basis (it projects both the old stored polarization reference vector and planeNormal, which gives the new polarization reference vector, into the plane perpendicular to \(\hat{\mathbf k}_{\rm in}\)) and use these values as q,u.

6) Apply the Thomson-scattering Mueller matrix (here the convention is Q positive is polarization is perpendicular to the scattering plane)
\[
\begin{pmatrix}
I'\\
Q'\\
U'
\end{pmatrix}
=
\begin{pmatrix}
1+\mu^2 & 1-\mu^2 & 0\\
1-\mu^2 & 1+\mu^2 & 0\\
0&0&2\mu
\end{pmatrix}
\begin{pmatrix}
I\\
Q\\
U
\end{pmatrix}.
\]

      in that scattering-plane basis, so:
\[ I'/I = (1+\mu^2) + (1-\mu^2)q, \]
\[ Q'/I =(1-\mu^2)+(1+\mu^2)q,\]
\[U'/I = 2\mu u,\]
and stores $q_{\rm new}=Q'/I',\, u_{\rm new}=U'/I'$.

7) Update the packet velocity with the its actual outgoing velocity
8) Mark polarization as initialized

#### SamplePolarizedThomsonDirection(...) 
It is a rejection-sampling algorithm that propose a Thomson scattering direction and correct its probability for the incoming direction. 
NB: This function only chooses the direction, doesn't modify the packet (which will be done in ApplyThomsonScatter()).
There are few steps:
1) With sampleSyntheticThomsonDirection() proposes a candidate outgoing direction from the unpolarized Thomson distribution ($d\sigma/d\Omega\propto1+\cos^2\theta$). Then it samples an azimuth $\phi$ and construct the new direction as $k \mu + e1 \sin\theta \cos\phi+ e2 \sin\theta \sin\phi$, normalizes it and multiply for the magnitude of the ingoing velocity $k$. Here e1, e2 are chosen to form a orthonormal basis with $\hat{\mathbf k}$.
2) It build and normalized $\mathbf{n_{\rm plane}}=
\hat{\mathbf k}_{\rm in}\times \hat{\mathbf k}_{\rm out}$, with the normalized incoming and outcoming direction.
3) With projectToBasis(p, oldVelocity, planeNormal), it rotates the packet's stored \(q,u\) into a basis orthogonal to oldVelocity (actually we just care of q, since the Thomson differential cross section for a given scattering plane depends on the linear polarization component aligned/perpendicular to that plane). 
How this rotation happens? Before scattering, the packet already carries $q,u$ defined relative to some stored reference polarization vector, which we project as $\mathbf e_{\rm old} = Proj_{\perp \rm k}(\texttt{p.polarizationBasis})$ that lies in the plane perpendicular to the incoming photon direction (i.e. $\mathbf e_{\rm old}\cdot\hat{\mathbf k}_{\rm in}=0$). We project as well $\mathbf n_{\rm plane}$, obtaining $\mathbf e_{\rm new}$. So both $\mathbf e_{\rm old}$ and $\mathbf e_{\rm new}$ live in the same 2D plane perpendicular to the incoming photon velocity, separated by an angle $\psi$. We don't really need this angle: it's enought to project: 
\[ \cos\psi = \mathbf e_{\rm old}\cdot\mathbf e_{\rm new}, \quad 
\sin\psi = \hat{\mathbf k}_{\rm in}\cdot
(\mathbf e_{\rm old}\times\mathbf e_{\rm new}),
\]
from which
\[
q' = q\cos2\psi + u\sin2\psi, \quad
u' = -q\sin2\psi + u\cos2\psi.
\]
The \(2\psi\) appears because linear polarization has a \(180^\circ\), rather than \(360^\circ\), symmetry (you see it better from the definition of the Stokes param).

4) Use the new q to modify the acceptance probability with
\[ {\rm ratio} = 1.0 + \frac{1.0 - \mu^2}{1.0 + \mu^2} q\]
and check the candidate (i.e. it's another rejection step).

MC:
new photon
Q = U = 0
      ↓
propagate
      ↓
Thomson scatter
      ↓
choose new direction depending on Q,U
      ↓
apply Thomson Mueller matrix
      ↓
update Q,U and polarization basis
      ↓
more scatterings
      ↓
observer

In a DDMC region, the code might skip thousands of individual physical scatterings. But if you skip them all, you cannot simply leave \(Q,U\) unchanged, because many scatterings tend to erase polarization memory.

### DDMC:
enter optically thick DDMC region
      ↓
do not resolve all microscopic scatterings:
estimate how many scatterings occurred
      ↓
damp Q,U to account for lost polarization memory
      ↓
simulate the last few scatterings explicitly
      ↓
exit DDMC region

More in detail:
The code stores in PendingMeanScatterings an expected number of unresolved scatterings, estimated with accumulateAcceleratedPolarizationHistory() as  
\[\langle N_{\rm scat}\rangle\simeq\,c\sigma_{\rm scat}\,\Delta t.\]
Then when the packet exits the accelerated region, FinalizeAcceleratedPolarizationHistory() does two things: 
- it damps the existing polarization as
\[
Q\rightarrow Q\,e^{-N/N_{\rm pol}},
\]
\[
U\rightarrow U\,e^{-N/N_{\rm pol}},
\]
where depolarizationScatterings controls the damping scale (with \(N_{\rm pol}\), a characteristic number of unresolved scatterings needed to erase polarization memory).
- it explicitly simulates the last few scatterings with applyManualSyntheticScatterings().

There is also this branch:
if(resetOccurred)
{
    p.velocity = SampleIsotropicVelocity(...);
    ResetUnpolarized(p);
}

This represents an effective absorption/re-emission event during accelerated transport. Such an event destroys the incoming polarization memory, so it resets $Q=U=0$ and also randomizes the microscopic photon direction.

## RadiationIMC.hpp

This is the file that decide what happens next and, in case, activate MC.
Basically, we compute four competing times ($t_{\rm intersection}, t_{\rm scattering}, t_{\rm timeleft}, t_{\rm observer}$) and pick the minimum to decide which event happens. They are defined as:
#### scattering time
This comes directly from sampling an optical depth $d\tau=\alpha ds$ (and $ds=cdt$ for a photon)

\[ t_{\rm scat} = \frac{-\ln \xi}{\alpha_{\rm event}\,D\,|\mathbf v_{\gamma}|}\approx \frac{-\ln \xi}{\alpha_{\rm event}c},\]
where \(\xi\) is a uniform random number between 0 and 1 and 
\[\alpha_{\rm event}=
\alpha_{\rm scat}
+\alpha_{\rm eff.abs}
+\alpha_{\rm Compton}.
\]

#### intersection time
\[t_{\rm intersection}=\frac{d_{\rm boundary}}{|\mathbf v_\gamma|}\simeq
\frac{d_{\rm boundary}}{c}.\]

####  time left
If the photon has already spent \(t_{\rm elapsed}\) travelling during that step, then
\[t_{\rm left}=\Delta t-t_{\rm elapsed}.\]

####  observer left
Schematically,
\[\mathbf r(t)=\mathbf r_0+\mathbf v_\gamma t\]
and it solves
\[|\mathbf r(t)|=R_{\rm obs}.\]

The packet is then propagated for exactly that time.

### When scatttering happens
Inside `else if(min.first == SCATTERING)`, 
it draws another random number with 
`double eventRandom = this->randomUnitOpen(particle) * eventOpacity`;
Then it partitions the total opacity.
`if(eventRandom < elasticScatteringOpacity)` and polarization is enabled, applies samplePolarizedThomsonDirection() and applyThomsonScatter(). 
Otherwise 
`if(
    (eventRandom -= elasticScatteringOpacity)
    < effectiveAbsorptionOpacity)`,
which is an effective scattering arising from IMC absorption/re-emission treatment, not an elastic Thomson event. It samples a new direction with and sets isEffectiveScatter = true.
Otherwise corresponds to a Compton event.



## Submit

Edit GENERATIONS in submit.sh to control the number of final statistics generations.
The run automatically adds 21 fixed burn-in/probe generations before the statistics generations. Only the final GENERATIONS are accumulated into the reported means and statistical errors, which scale approximately as (1/\sqrt{\mathrm{GENERATIONS}}).
The packet budgets per generation are controlled by:
MG_LEARNED_BUDGET=1600
GREY_LEARNED_BUDGET=500
LEARNED_MAX_PER_CELL=50000
The snapshot path and RICH installation paths are set directly in submit.sh. Before submitting, make sure that 
1. SNAPSHOT, 
2. here, 
3. root 
point to 
1. the desired snapshot h5 file, 
2. RICH build directory,  
3. RICH repository.

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

``` ON MY LOCAL MACHINE

python3 plot_sphere.py /Users/paolamartire/shocks/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/Polarization/tde_gray_75polarization151.vtk --list-fields

python3 plot_sphere.py \
  /Users/paolamartire/shocks/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonMG/Polarization/tde_mg_1500polarization429.vtk \
  --field forward_luminosity --scale log

python3 plot_sphere.py \
  /Users/paolamartire/shocks/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonMG/Polarization/tde_mg_1500polarization429.vtk \
  --field grey_polarization_degree

python3 plot_sphere.py \
  /Users/paolamartire/shocks/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonMG/Polarization/tde_mg_1500polarization429.vtk \
  --field polarization_degree

``` 

