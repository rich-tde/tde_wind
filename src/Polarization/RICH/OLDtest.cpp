#include "source/3D/radiation/postprocess/IMCPostProcess.hpp"

// This test runs a post-process IMC calculation on a single Snapshot3D, 
// without evolving the fluid 

namespace
{

PostProcessIMC::PostProcessScenario MakeTdeLuminosityScenario()
{
    PostProcessIMC::PostProcessScenario scenario =
        PostProcessIMC::MakeStaSnapshotScenario(
            "tde-gray-mg-luminosity-and-polarization");
    PostProcessIMC::PostProcessConfig& config = scenario.defaults;

    // Products: merged HDF5 statistics and a combined observer-sphere VTK map, 
    // which will be writtten with the following prefix:
    config.output.stem = "output/tde_gray_mg_polarization";
    config.output.writeVtk = true;

    // Observer sphere. Distances are in cm and times are in seconds.
    config.observer.radius = 7.5e14;
    config.observer.count = 512;
    config.transport.sourceDt = 100.0; // source time interval used for packet generation
    config.transport.duration = 1.0e6; // post-processing transport interval
    config.transport.photonsPerCell = 50;
    config.transport.useCellVelocities = true; // radiation transport with gas velocity????

    // Use two-sided MPI particle exchange because it has the smaller memory
    // footprint for this calculation. RDMA remains available through
    // --transport.communication rdma.
    config.transport.communication = PostProcessIMC::MonteCarloCommunication::TwoSided;

    // The same FLD-normalized thermalization surface feeds the multigroup and
    // gray STA calculations, so their luminosities are directly comparable.
    config.fluxSource.enabled = true; // the code constructs an internal surface from which radiation is injected
    config.fluxSource.thermalizationTau = 5.0; // searches for a thermalization surface around tau_eff = 5
    config.fluxSource.constructionRays = 4096; // rays/directions to determine the surface geometry.
    config.fluxSource.ddmcFaceOpticalDepth = 5.0; // where diffusion-based Monte-Carlo treatment can be used near the source surface
    // Place the CER  (common emission radius????) at the innermost per-group thermalization radius, so every
    // group is Planckian at that surface and the transport resolves each group's
    // own thermalization layer above it.
    config.fluxSource.surfaceMode = PostProcessIMC::FluxSourceSurfaceMode::MultigroupInnermost;
    // Every cell outside that surface emits thermally at its snapshot
    // temperature (Fleck factor 1); cells below 1e-10 of the total emission
    // get no packets. Burn-in generations subsample the emitting cells.
    config.volumeEmission.enabled = true;
    config.volumeEmission.cutoffFraction = 1.0e-10;
    // first a learning/burn-in stage: 
    // later generations preferentially spend Monte-Carlo packets on the important cells.
    config.volumeEmission.burninPacketsTarget = 2000000; 
    config.volumeEmission.learnedPhotonsPerCellBudget = 10;
    config.volumeEmission.learnedMinPhotons = 1;
    config.volumeEmission.learnedMaxPhotons = 5000;
    // Fleck factor 1 in every group everywhere: every cell outside the deep
    // surface emits its full spectrum, the burn-in explores each cell with the
    // same packet count. The radial
    // per-group gate is off: this snapshot is a thin disk whose photosphere
    // radiates vertically, so a radial tau_eff does not measure escape (gate
    // scan tau=3..30 never converged; gate off is converged against the deep
    // surface depth, tau 5 vs 100 agree to 2.5%).
    config.volumeEmission.gateGroups = false; 
    // Samples cells in a controlled/exact fashion. 
    // It helps establish the source importance reliably before adaptive sampling takes over.???
    config.volumeEmission.burninExact = true; 
    // exploration packets are split so none carries
    // more than 5% of the previous generation's escaping energy
    config.volumeEmission.explorationWeightFraction = 0.05;
    // Discrete Diffusion Monte Carlo above tau=5
    config.transport.ddmc = true;
    config.transport.ddmcMinCellOpticalDepth = 5.0;
    // Soft-group packets injected at the deep surface must cross layers whose
    // per-cell group depth is 1-15; at the default threshold of 15 they random
    // walk explicitly for 1e5+ steps each. DDMC is accurate down to a few.
    config.transport.randomWalk = false; // another possible acceleration technique. Switched-off because we use DDMC 
    config.transport.compton.enabled = false; // NO Comptonization
    config.opacityScaling.mode = PostProcessIMC::OpacityScaleMode::Planck; // use Planck-mean weighting

    // Track Stokes Q and U in both passes and retain all observer/photosphere
    // fields in the output bundle.
    config.polarization.enabled = true;
    config.photosphere.enabled = true;

    // Learn which source cells and energy groups reach each observer before
    // accumulating the independent generations used for reported errors.
    // Final generations use a Neyman allocation: packets per cell go as the
    // square root of the cell's deficit-weighted variance share, within
    // [learnedMinPhotons, learnedMaxPhotons], at an average of
    // learnedPhotonsPerCellBudget packets per learned cell per generation.
    config.adaptive.source.enabled = true;
    // in the end, a learned cell gets somewhere between 
    // learnedMinPhotons and learnedMaxPhotons packets per generation, 
    // with an average of learnedPhotonsPerCellBudget.
    config.adaptive.source.learnedPhotonsPerCellBudget = 100;
    config.adaptive.source.learnedMinPhotons = 10;
    config.adaptive.source.learnedMaxPhotons = 5000;
    // Neyman-style allocation, i.e. square-root weighting:
    // if a cell contributes variance V_i, receive N_i \propto sqrt(V_i) more photons.
    config.adaptive.source.scorePower = 0.5;
    // Steer toward an absolute polarization-degree uncertainty rather than an
    // SNR: weakly polarized observers can never reach SNR 10, and chasing it
    // saturates every deficit at the cap. Deficits are median-normalized so
    // only their ratios matter for the allocation.
    config.adaptive.observer.equity = true; // The adaptive algorithm tries not to optimize only the easiest or brightest observer directions, but it attempts to distribute computational effort
    config.adaptive.observer.targetPolarizationSigma = 1.0e-3; // target absolute uncertainty in polarization degree
    config.adaptive.observer.normalizeDeficits = true;
    // The adaptive algorithm can also learn which energy groups and which cells are important for each observer, 
    // and allocate more packets to those groups. 
    config.adaptive.group.quality = true;
    config.adaptive.group.sourceCells = true;
    config.adaptive.group.frequencySampling = true;
    config.loadBalance.measured = true; // redistributes work between MPI ranks since not all photons are equal in cost (eg: transparent regions are cheaper)

    return scenario;
}

} // namespace

int main(int argc, char* argv[])
{
    return PostProcessIMC::RunPostProcessMain(
        argc, argv, MakeTdeLuminosityScenario());
}
