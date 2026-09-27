#include "source/3D/radiation/postprocess/IMCPostProcess.hpp"

namespace
{

PostProcessIMC::PostProcessScenario MakeTdeLuminosityScenario()
{
    PostProcessIMC::PostProcessScenario scenario =
        PostProcessIMC::MakeStaSnapshotScenario(
            "tde-gray-mg-luminosity-and-polarization");
    PostProcessIMC::PostProcessConfig& config = scenario.defaults;

    // Products: merged HDF5 statistics and a combined observer-sphere VTK map.
    config.output.stem = "output/tde_gray_mg_polarization";
    config.output.writeVtk = true;

    // Observer sphere. Distances are in cm and times are in seconds.
    config.observer.radius = 7.5e14;
    config.observer.count = 512;
    config.transport.sourceDt = 100.0;
    config.transport.duration = 1.0e6;
    config.transport.photonsPerCell = 50;
    config.transport.useCellVelocities = true;

    // Use two-sided MPI particle exchange because it has the smaller memory
    // footprint for this calculation. RDMA remains available through
    // --transport.communication rdma.
    config.transport.communication = PostProcessIMC::MonteCarloCommunication::TwoSided;

    // The same FLD-normalized thermalization surface feeds the multigroup and
    // gray STA calculations, so their luminosities are directly comparable.
    config.fluxSource.enabled = true;
    config.fluxSource.thermalizationTau = 5.0;
    config.fluxSource.constructionRays = 4096;
    config.fluxSource.ddmcFaceOpticalDepth = 5.0;
    // Place the CER at the innermost per-group thermalization radius, so every
    // group is Planckian at the surface and the transport resolves each group's
    // own thermalization layer above it.
    config.fluxSource.surfaceMode = PostProcessIMC::FluxSourceSurfaceMode::MultigroupInnermost;
    // Every cell outside that surface emits thermally at its snapshot
    // temperature (Fleck factor 1); cells below 1e-10 of the total emission
    // get no packets. Burn-in generations subsample the emitting cells.
    config.volumeEmission.enabled = true;
    config.volumeEmission.cutoffFraction = 1.0e-10;
    config.volumeEmission.burninPacketsTarget = 2000000;
    config.volumeEmission.learnedPhotonsPerCellBudget = 10;
    config.volumeEmission.learnedMinPhotons = 1;
    config.volumeEmission.learnedMaxPhotons = 5000;
    // Fleck factor 1 in every group everywhere: every cell outside the deep
    // surface emits its full spectrum, the burn-in explores each cell with the
    // same packet count, and exploration packets are split so none carries
    // more than 5% of the previous generation's escaping energy. The radial
    // per-group gate is off: this snapshot is a thin disk whose photosphere
    // radiates vertically, so a radial tau_eff does not measure escape (gate
    // scan tau=3..30 never converged; gate off is converged against the deep
    // surface depth, tau 5 vs 100 agree to 2.5%).
    config.volumeEmission.gateGroups = false;
    config.volumeEmission.burninExact = true;
    config.volumeEmission.explorationWeightFraction = 0.05;
    config.transport.ddmc = true;
    // Soft-group packets injected at the deep surface must cross layers whose
    // per-cell group depth is 1-15; at the default threshold of 15 they random
    // walk explicitly for 1e5+ steps each. DDMC is accurate down to a few.
    config.transport.ddmcMinCellOpticalDepth = 5.0;
    config.transport.randomWalk = false;
    config.transport.compton.enabled = false;
    config.opacityScaling.mode = PostProcessIMC::OpacityScaleMode::Planck;

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
    config.adaptive.source.learnedPhotonsPerCellBudget = 100;
    config.adaptive.source.learnedMinPhotons = 10;
    config.adaptive.source.learnedMaxPhotons = 5000;
    config.adaptive.source.scorePower = 0.5;
    // Steer toward an absolute polarization-degree uncertainty rather than an
    // SNR: weakly polarized observers can never reach SNR 10, and chasing it
    // saturates every deficit at the cap. Deficits are median-normalized so
    // only their ratios matter for the allocation.
    config.adaptive.observer.equity = true;
    config.adaptive.observer.targetPolarizationSigma = 1.0e-3;
    config.adaptive.observer.normalizeDeficits = true;
    config.adaptive.group.quality = true;
    config.adaptive.group.sourceCells = true;
    config.adaptive.group.frequencySampling = true;
    config.loadBalance.measured = true;

    return scenario;
}

} // namespace

int main(int argc, char* argv[])
{
    return PostProcessIMC::RunPostProcessMain(
        argc, argv, MakeTdeLuminosityScenario());
}
