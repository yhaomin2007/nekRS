#include "twoFluid.hpp"

#include "fluidSolver.hpp"
#include "geomSolver.hpp"
#include "linAlg.hpp"
#include "registerKernels.hpp"

twoFluid_t::twoFluid_t(fluidSolver_t *liquid_,
                       fluidSolver_t *gas_,
                       const std::unique_ptr<geomSolver_t> &geom_)
    : liquid(liquid_), gas(gas_), geom(geom_)
{
}

void twoFluid_t::setup()
{
  auto mesh = liquid->mesh;
  platform->options.getArgs("TWO FLUID GAS VOLUME FRACTION", alphaG);
  platform->options.getArgs("TWO FLUID BUBBLE DIAMETER", bubbleDiameter);
  platform->options.getArgs("TWO FLUID ALPHA FLOOR", alphaFloor);
  platform->options.getArgs("TWO FLUID GRAVITY X", gravity[0]);
  platform->options.getArgs("TWO FLUID GRAVITY Y", gravity[1]);
  platform->options.getArgs("TWO FLUID GRAVITY Z", gravity[2]);
  platform->options.getArgs("TWO FLUID DRAG MULTIPLIER", dragMultiplier);
  platform->options.getArgs("TWO FLUID COUPLING ITERATIONS", couplingIterations);
  platform->options.getArgs("TWO FLUID PRESSURE CORRECTORS", pressureCorrectors);
  projectionOnly = platform->options.compareArgs("TWO FLUID PROJECTION ONLY", "TRUE");

  nekrsCheck(alphaG <= 0 || alphaG >= 1,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "TWO FLUID gasVolumeFraction must lie strictly between zero and one: %g\n",
             alphaG);
  nekrsCheck(bubbleDiameter <= 0,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "TWO FLUID bubbleDiameter must be positive: %g\n",
             bubbleDiameter);
  nekrsCheck(dragMultiplier < 0,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "TWO FLUID dragMultiplier must be non-negative: %g\n",
             dragMultiplier);
  nekrsCheck(couplingIterations < 1 || pressureCorrectors < 1,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s\n",
             "TWO FLUID couplingIterations and pressureCorrectors must be positive.");

  alphaL = 1 - alphaG;
  const auto N = liquid->fieldOffset;
  o_alphaG = platform->device.malloc<dfloat>(N);
  o_alphaL = platform->device.malloc<dfloat>(N);
  o_drag = platform->device.malloc<dfloat>(N);
  o_implicitLiquid = platform->device.malloc<dfloat>(N);
  o_implicitGas = platform->device.malloc<dfloat>(N);
  o_pressureCoeff = platform->device.malloc<dfloat>(N);
  o_pressureResponseLiquid = platform->device.malloc<dfloat>(N);
  o_pressureResponseGas = platform->device.malloc<dfloat>(N);
  o_mixtureFlux = platform->device.malloc<dfloat>(mesh->dim * liquid->fieldOffset);
  o_baseExtLiquid = platform->device.malloc<dfloat>(liquid->fieldOffsetSum);
  o_baseExtGas = platform->device.malloc<dfloat>(gas->fieldOffsetSum);
  o_divergenceLiquid = platform->device.malloc<dfloat>(N);
  o_divergenceGas = platform->device.malloc<dfloat>(N);
  o_slipVelocity = platform->device.malloc<dfloat>(mesh->dim * liquid->fieldOffset);
  o_mixtureVelocity = platform->device.malloc<dfloat>(mesh->dim * liquid->fieldOffset);
  o_interphaseForce = platform->device.malloc<dfloat>(mesh->dim * liquid->fieldOffset);
  o_continuityResidual = platform->device.malloc<dfloat>(N);

  platform->linAlg->fill(mesh->Nlocal, alphaG, o_alphaG);
  platform->linAlg->fill(mesh->Nlocal, alphaL, o_alphaL);
  platform->linAlg->fill(N, 0.0, o_drag);
  platform->linAlg->fill(N, 0.0, o_implicitLiquid);
  platform->linAlg->fill(N, 0.0, o_implicitGas);
  platform->linAlg->fill(mesh->dim * N, 0.0, o_slipVelocity);
  platform->linAlg->fill(mesh->dim * N, 0.0, o_mixtureVelocity);
  platform->linAlg->fill(mesh->dim * N, 0.0, o_interphaseForce);
  platform->linAlg->fill(N, 0.0, o_continuityResidual);
  platform->linAlg->fill(N, 0.0, o_pressureResponseLiquid);
  platform->linAlg->fill(N, 0.0, o_pressureResponseGas);
  platform->linAlg->fill(N, 0.0, o_pressureCoeff);
  platform->linAlg->fill(N, 0.0, o_divergenceLiquid);
  platform->linAlg->fill(N, 0.0, o_divergenceGas);
  platform->linAlg->fill(N, 0.0, liquid->o_div);
  platform->linAlg->fill(N, 0.0, gas->o_div);

  // Both phases see the same pressure storage.  Only the primary phase owns
  // and destroys the pressure elliptic solver.
  gas->o_P = liquid->o_P;
  gas->o_Pe = liquid->o_Pe;
  liquid->twoFluid = this;

  liquid->userImplicitLinearTerm = [this](double) { return o_implicitLiquid; };
  gas->userImplicitLinearTerm = [this](double) { return o_implicitGas; };

  // UDF initial conditions have already been loaded at this point.  Report
  // them before the first pressure projection so initialization and
  // projection effects can be distinguished.
  reportContinuity("initial");
}

void twoFluid_t::updateAdvectionCoordinates()
{
  auto update = [&](fluidSolver_t *phase) {
    const int relative = 0; // moving mesh is intentionally excluded in v1
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      launchKernel("nrs-UrstCubatureHex3D",
                   phase->mesh->Nelements,
                   relative,
                   phase->mesh->o_cubvgeo,
                   phase->mesh->o_cubInterpT,
                   phase->fieldOffset,
                   0,
                   phase->cubatureOffset,
                   phase->o_U,
                   o_NULL,
                   phase->o_relUrst);
    } else {
      launchKernel("nrs-UrstHex3D",
                   phase->mesh->Nelements,
                   relative,
                   phase->mesh->o_vgeo,
                   phase->fieldOffset,
                   0,
                   phase->o_U,
                   o_NULL,
                   phase->o_relUrst);
    }
  };
  update(liquid);
  update(gas);
}

void twoFluid_t::makeExplicit(double time)
{
  auto mesh = liquid->mesh;
  o_baseExtLiquid.copyFrom(liquid->o_EXT, liquid->fieldOffsetSum);
  o_baseExtGas.copyFrom(gas->o_EXT, gas->fieldOffsetSum);
  launchKernel("twoFluid::dragSource",
               mesh->Nlocal,
               liquid->fieldOffset,
               alphaFloor,
               alphaL,
               alphaG,
               bubbleDiameter,
               dragMultiplier,
               gravity[0], gravity[1], gravity[2],
               liquid->o_rho,
               liquid->o_mue,
               gas->o_rho,
               liquid->o_U,
               gas->o_U,
               o_drag,
               o_implicitLiquid,
               o_implicitGas,
               liquid->o_EXT,
               gas->o_EXT);
}

void twoFluid_t::refreshCouplingForcing(double time, int tstep)
{
  liquid->o_EXT.copyFrom(o_baseExtLiquid, liquid->fieldOffsetSum);
  gas->o_EXT.copyFrom(o_baseExtGas, gas->fieldOffsetSum);

  if (platform->options.compareArgs("EQUATION TYPE", "NAVIERSTOKES")) {
    liquid->makeAdvection(time, tstep);
    gas->makeAdvection(time, tstep);
  }

  makeExplicit(time);
  liquid->makeForcing(false);
  gas->makeForcing(false);
}

void twoFluid_t::finalizeCouplingForcing()
{
  if (liquid->o_coeffEXT.size() > 1) {
    liquid->o_EXT.copyFrom(liquid->o_EXT,
                           liquid->fieldOffsetSum,
                           liquid->fieldOffsetSum,
                           0);
    gas->o_EXT.copyFrom(gas->o_EXT,
                        gas->fieldOffsetSum,
                        gas->fieldOffsetSum,
                        0);
  }
}

void twoFluid_t::phasePressureFlux(fluidSolver_t *phase, occa::memory o_flux)
{
  auto mesh = phase->mesh;

  auto o_curl = platform->deviceMemoryPool.reserve<dfloat>(phase->fieldOffsetSum);
  launchKernel("core-curlHex3D",
               mesh->Nelements,
               1,
               mesh->o_vgeo,
               mesh->o_D,
               phase->fieldOffset,
               phase->o_Ue,
               o_curl);
  oogs::startFinish(o_curl, mesh->dim, phase->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs3);
  platform->linAlg->axmyVector(mesh->Nlocal,
                              phase->fieldOffset,
                              0,
                              1.0,
                              mesh->o_invLMM,
                              o_curl);

  auto o_curlCurl = platform->deviceMemoryPool.reserve<dfloat>(phase->fieldOffsetSum);
  launchKernel("core-curlHex3D",
               mesh->Nelements,
               1,
               mesh->o_vgeo,
               mesh->o_D,
               phase->fieldOffset,
               o_curl,
               o_curlCurl);

  auto o_gradDiv = platform->deviceMemoryPool.reserve<dfloat>(phase->fieldOffsetSum);
  launchKernel("core-gradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               phase->fieldOffset,
               phase->o_div,
               o_gradDiv);

  auto o_inverseRho = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->adyz(mesh->Nlocal, 1.0, phase->o_rho, o_inverseRho);
  launchKernel("fluidSolver_t::pressureRhsHex3D",
               mesh->Nlocal,
               phase->fieldOffset,
               phase->o_mue,
               o_inverseRho,
               phase->o_JwF,
               o_curlCurl,
               o_gradDiv,
               o_flux);
  oogs::startFinish(o_flux, mesh->dim, phase->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs3);
  platform->linAlg->axmyVector(mesh->Nlocal,
                              phase->fieldOffset,
                              0,
                              1.0,
                              mesh->o_invLMM,
                              o_flux);
}

void twoFluid_t::updatePressureResponse(bool scaleForMomentumPredictor)
{
  const dfloat g0idt = *liquid->g0 / liquid->dt[0];
  const dfloat pressureScale = scaleForMomentumPredictor ? g0idt : 1.0;
  launchKernel("twoFluid::pressureResponse",
               liquid->mesh->Nlocal,
               alphaFloor,
               alphaL,
               alphaG,
               g0idt,
               pressureScale,
               liquid->o_rho,
               gas->o_rho,
               o_drag,
               o_pressureResponseLiquid,
               o_pressureResponseGas,
               o_pressureCoeff);
}

void twoFluid_t::solvePressure(double time, int stage)
{
  if (!liquid->ellipticSolverP) return;

  auto mesh = liquid->mesh;
  platform->timer.tic("mixture pressureSolve");

  auto o_fL = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffsetSum);
  auto o_fG = platform->deviceMemoryPool.reserve<dfloat>(gas->fieldOffsetSum);
  phasePressureFlux(liquid, o_fL);
  phasePressureFlux(gas, o_fG);

  launchKernel("twoFluid::mixtureFlux",
               mesh->Nlocal,
               liquid->fieldOffset,
               alphaL,
               alphaG,
               o_fL,
               o_fG,
               o_mixtureFlux);

  auto o_pRhs = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffset);
  launchKernel("core-wDivergenceVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               liquid->fieldOffset,
               o_mixtureFlux,
               o_pRhs);

  auto o_divMixture = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffset);
  platform->linAlg->axpbyz(mesh->Nlocal,
                           alphaL,
                           liquid->o_div,
                           alphaG,
                           gas->o_div,
                           o_divMixture);
  const dfloat g0idt = *liquid->g0 / liquid->dt[0];
  launchKernel("fluidSolver_t::pressureAddQtl",
               mesh->Nlocal,
               mesh->o_LMM,
               g0idt,
               o_divMixture,
               o_pRhs);

  launchKernel("twoFluid::mixtureVelocity",
               mesh->Nlocal,
               liquid->fieldOffset,
               alphaL,
               alphaG,
               liquid->o_U,
               gas->o_U,
               o_mixtureVelocity);
  launchKernel("fluidSolver_t::divergenceSurfaceHex3D",
               mesh->Nelements,
               mesh->o_sgeo,
               mesh->o_vmapM,
               liquid->o_EToB,
               g0idt,
               liquid->fieldOffset,
               o_mixtureFlux,
               o_mixtureVelocity,
               o_pRhs);

  updatePressureResponse(true);

  liquid->ellipticSolverP->solve(o_pressureCoeff,
                                 o_NULL,
                                 o_pRhs,
                                 liquid->o_P.slice(0, mesh->Nlocal));
  platform->timer.toc("mixture pressureSolve");
}

void twoFluid_t::correctMixtureContinuity(double time)
{
  if (!liquid->ellipticSolverP) return;

  auto mesh = liquid->mesh;
  updatePressureResponse(false);

  for (int corrector = 0; corrector < pressureCorrectors; ++corrector) {
    launchKernel("twoFluid::mixtureVelocity",
                 mesh->Nlocal,
                 liquid->fieldOffset,
                 alphaL,
                 alphaG,
                 liquid->o_U,
                 gas->o_U,
                 o_mixtureVelocity);

    auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffset);
    launchKernel("core-wDivergenceVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 liquid->fieldOffset,
                 o_mixtureVelocity,
                 o_rhs);

    auto o_phi = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffset);
    platform->linAlg->fill(liquid->fieldOffset, 0.0, o_phi);
    liquid->ellipticSolverP->solve(o_pressureCoeff, o_NULL, o_rhs, o_phi.slice(0, mesh->Nlocal));

    auto o_weakGradient = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffsetSum);
    launchKernel("core-wGradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 liquid->fieldOffset,
                 o_phi,
                 o_weakGradient);
    oogs::startFinish(o_weakGradient,
                      mesh->dim,
                      liquid->fieldOffset,
                      ogsDfloat,
                      ogsAdd,
                      mesh->oogs3);
    platform->linAlg->axmyVector(mesh->Nlocal,
                                liquid->fieldOffset,
                                0,
                                1.0,
                                mesh->o_invLMM,
                                o_weakGradient);

    launchKernel("twoFluid::correctPhaseVelocities",
                 mesh->Nlocal,
                 liquid->fieldOffset,
                 o_pressureResponseLiquid,
                 o_pressureResponseGas,
                 o_weakGradient,
                 liquid->o_U,
                 gas->o_U);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_phi, 1.0, liquid->o_P);
    liquid->applyDirichlet(time);
    gas->applyDirichlet(time);
  }

  updateDiagnostics();
}

void twoFluid_t::updateDiagnostics()
{
  auto mesh = liquid->mesh;
  launchKernel("twoFluid::mixtureVelocity",
               mesh->Nlocal,
               liquid->fieldOffset,
               alphaL,
               alphaG,
               liquid->o_U,
               gas->o_U,
               o_mixtureVelocity);
  launchKernel("twoFluid::diagnosticFields",
               mesh->Nlocal,
               liquid->fieldOffset,
               liquid->o_U,
               gas->o_U,
               o_drag,
               o_slipVelocity,
               o_interphaseForce);
  weakDivergence(liquid->o_U, o_divergenceLiquid);
  weakDivergence(gas->o_U, o_divergenceGas);
  weakDivergence(o_mixtureVelocity, o_continuityResidual);
}

void twoFluid_t::weakDivergence(const occa::memory &o_velocity,
                                occa::memory o_divergence)
{
  auto mesh = liquid->mesh;
  launchKernel("core-wDivergenceVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               liquid->fieldOffset,
               o_velocity,
               o_divergence);
  oogs::startFinish(o_divergence, 1, liquid->fieldOffset,
                    ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_divergence);
}

void twoFluid_t::reportContinuity(const char *label)
{
  auto mesh = liquid->mesh;
  updateDiagnostics();
  const auto comm = platform->comm.mpiComm();
  const auto maxLiquid = platform->linAlg->amax(mesh->Nlocal, o_divergenceLiquid, comm);
  const auto maxGas = platform->linAlg->amax(mesh->Nlocal, o_divergenceGas, comm);
  const auto maxMixture = platform->linAlg->amax(mesh->Nlocal, o_continuityResidual, comm);
  const auto volumeScale = 1.0 / sqrt(mesh->volume);
  const auto l2Liquid = volumeScale * platform->linAlg->weightedNorm2(
      mesh->Nlocal, mesh->o_LMM, o_divergenceLiquid, comm);
  const auto l2Gas = volumeScale * platform->linAlg->weightedNorm2(
      mesh->Nlocal, mesh->o_LMM, o_divergenceGas, comm);
  const auto l2Mixture = volumeScale * platform->linAlg->weightedNorm2(
      mesh->Nlocal, mesh->o_LMM, o_continuityResidual, comm);
  if (platform->comm.mpiRank() == 0) {
    printf("twoFluid divergence%s%s max|L|=%.8e max|G|=%.8e max|mix|=%.8e "
           "L2(L)=%.8e L2(G)=%.8e L2(mix)=%.8e\n",
           label ? " " : "",
           label ? label : "",
           maxLiquid,
           maxGas,
           maxMixture,
           l2Liquid,
           l2Gas,
           l2Mixture);
  }
}

void registerTwoFluidKernels()
{
  occa::properties props = platform->kernelInfo;
  const std::string dir = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/twoFluid/");
  for (const auto &name : {"diagnosticFields",
                           "correctPhaseVelocities",
                           "dragSource",
                           "mixtureFlux",
                           "mixtureVelocity",
                           "pressureCoefficient",
                           "pressureResponse"})
    platform->kernelRequests.add("twoFluid::" + std::string(name),
                                 dir + std::string(name) + ".okl",
                                 props);
}
