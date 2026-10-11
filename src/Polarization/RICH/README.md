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
```bash
if(resetOccurred)
{
    p.velocity = SampleIsotropicVelocity(...);
    ResetUnpolarized(p);
}
```
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


## Which values are saved for --field:
In general source/3D/radiation/SphericalObserver.cpp defines the raw field names, and source/3D/radiation/postprocess/PostProcessResultWriter.cpp is where the final forward_ and grey_ prefixes are explicitly added.

### forward_luminosity
It's the bolometric/multigroup Monte Carlo escaping luminosity per observer patch.
##### How is computed
In source/3D/radiation/postprocess/ForwardCalculation.cpp, the multigroup/forward calculation uses 
`auto& observer = runtime.observer;`

and after all MC generations it does:
```bash
observer->loadStatisticalMeanTallies();
...
observer->writeVTK(forwardVtk, cfg.sourceDt);
```
Then SphericalObserver::writeVTK() writes its generic luminosity as
```bash
for (size_t i = 0; i < N; ++i)`
    file << observerEnergy_[i] * invDt << "\n";
```
with  `invDt = 1.0 / sourceDt;`, which physically is:
\[ L^{\rm forward}_i = \frac{E^{\rm MC}_i}{\Delta t}\]
where \(E_i^{\rm MC}\) is the energy accumulated from Monte Carlo packets assigned to observer at the point where they cross the observer sphere \(i\).
Because you have several final generations, observerEnergy_ is first replaced by the statistical mean across those final generations.

##### How is written
What SphericalObserver.cpp writes is:
`file << "SCALARS luminosity double 1\n";`
Likewise the group field is written as:
`file << "SCALARS group_" << g << "_luminosity double 1\n";`

So at this stage we only have something like:
luminosity
group_0_luminosity

The forward_ prefix is added later, when the forward and grey VTKs are merged into the final combined VTK in PostProcessResultWriter.cpp. The key function is:
```bash
std::string PrefixedField(std::string const& name, std::string const& pass)
{
    if (StartsWith(name, "photosphere_"))
        return name;
    if (StartsWith(name, "fld_"))
        return name;
    if (StartsWith(name, "log10_fld_"))
        return "fld_log10_" + name.substr(10);
    return pass + "_" + name;
}
```
This says: for an ordinary field, prepend the pass name. E.g: `PrefixedField("luminosity", "forward")` becomes `forward_luminosity`.

### forward_group_0_luminosity
Same but restricted to group 0

### fld_surface_luminosity
ForwardCalculation.cpp separately takes

```bash
std::vector<double> const& fldLuminosity =
    runtime.fldLuminosity;
```

and directly appends it:
```bash
vtkAppend << "SCALARS fld_surface_luminosity double 1\n"
          << "LOOKUP_TABLE default\n";

for (size_t p = 0; p < nObs; ++p)
    vtkAppend << fldLuminosity[p] << "\n";
```
Therefore there is no Monte Carlo propagation in this number and fld_surface_luminosity[i]=runtime.fldLuminosity[i], and runtime.fldLuminosity is built in PostProcessApplication.cpp by summing something of the form 
\[ L_i^{\rm FLD,surface} = \sum_{j\in i}
\max(0,F_{r,j}^{\rm FLD})\,\Delta A_j\]
where radialFlux is the projection on the radial direction of fldFlux[cell], which is build underd FLD approximation
It is therefore the snapshot's FLD radiative flux integrated over the relevant surface patches.
Which cell/patch are we looking at? Given our config.observer.radius, the code is evaluating the FLD field in cells intersecting the spherical surface (relative to observer.getCenter()).

##### How is written
PrefixedField() in PostProcessResultWriter handles it not adding nothing.

### grey_luminosity 
It's computed in source/3D/radiation/postprocess//GreyCalculation.cpp the same basic way as forward_luminosity: from Monte Carlo packets crossing the observer sphere, but in a grey mode

##### How is written
In PostProcessResultWriter.cpp
```bash
RenameScalar(
    greyLines[i], PrefixedField(name, "grey"))
```
so luminosity -> grey_luminosity

### fluxes
We define the physical area of that observer patch as
\[ A_i = \Delta\Omega_i R_{\rm obs}^2,\]
where the raius is config.observer.radius, and computes
\[F_i = \frac{L_i}{A_i}.\]
We do the same also for FLD, where the flux is not simply fldFlux[cell], but:  
\[
F_{{\rm FLD},i}^{\rm output}
\simeq
\frac{1}{A_i}
\int_{A_i}
\max(0,\mathbf F_{\rm FLD}\cdot\hat r)\,dA.
\]

### Radii
They are defined (source/3D/radiation/postprocess/PhotosphereCalculation.cpp) s.t. they are radii along each observer direction where the total optical depth (total for photosphere or effective for thermalization radius) reaches the chosen tauThreshold (2/3 for photosphere, 1 for thermalization).
The corresponding valid_* fields are flags indicating whether a valid crossing/radius was actually found along that ray.If non valid the radius is set to -1 (and the validity flag to 0).

##### How is computed
For every observer direction and every requested channel, the code launches a probe ray and lets it propagate through the simulation for the whole diameter of the observer sphere (i.e. 2*r_obs). 
Opacities are defined as:
\[ \tau_{\rm tot,new} =
\tau_{\rm tot,old} +\sigma_{\rm tot}\,ds, \quad
\tau_{\rm therm,new} = \tau_{\rm therm,old} + \sigma_{\rm therm}\,ds.
\]
where
\[\sigma_{\rm tot} = \sigma_{\rm abs}+\sigma_{\rm scat}, \quad 
\sigma_{\rm therm} = \sqrt{3\,\sigma_{\rm abs}\left(\sigma_{\rm abs}+\sigma_{\rm scat}\right)}
\]
`PhotosphereProbePhysics` checks where each threshold is crossed and return the radius (NB: it does not just use the cell center. It linearly interpolates inside the cell segment)

For photosphere_integrated_radius_tau_total, it first computes frequency-integrated multigroup absorption and scattering and then proceeds as the others.