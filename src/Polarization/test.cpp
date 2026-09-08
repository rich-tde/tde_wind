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
    config.transport.duration = 750000.0;
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
    config.transport.ddmc = true;
    config.transport.randomWalk = false;
    config.transport.compton.enabled = false;
    config.opacityScaling.mode = PostProcessIMC::OpacityScaleMode::Planck;

    // Track Stokes Q and U in both passes and retain all observer/photosphere
    // fields in the output bundle.
    config.polarization.enabled = true;
    config.photosphere.enabled = true;

    // Learn which source cells and energy groups reach each observer before
    // accumulating the independent generations used for reported errors.
    config.adaptive.source.enabled = true;
    config.adaptive.observer.equity = true;
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
