#include "twoFluid.hpp"

#include "fluidSolver.hpp"
#include "geomSolver.hpp"
#include "linAlg.hpp"
#include "linearSolverFactory.hpp"
#include "registerKernels.hpp"

#include <limits>

twoFluid_t::twoFluid_t(fluidSolver_t *liquid_,
                       fluidSolver_t *gas_,
                       const std::unique_ptr<geomSolver_t> &geom_)
    : liquid(liquid_), gas(gas_), geom(geom_)
{
}

twoFluid_t::~twoFluid_t()
{
  delete pressureCorrectionSolver;
}

void twoFluid_t::setup()
{
  auto mesh = liquid->mesh;
  platform->options.getArgs("TWO FLUID GAS VOLUME FRACTION", alphaG);
  platform->options.getArgs("TWO FLUID BUBBLE DIAMETER", bubbleDiameter);
  platform->options.getArgs("TWO FLUID ALPHA FLOOR", alphaFloor);
  platform->options.getArgs("TWO FLUID ALPHA DIFFUSIVITY", alphaDiffusivity);
  platform->options.getArgs("TWO FLUID GRAVITY X", gravity[0]);
  platform->options.getArgs("TWO FLUID GRAVITY Y", gravity[1]);
  platform->options.getArgs("TWO FLUID GRAVITY Z", gravity[2]);
  platform->options.getArgs("TWO FLUID DRAG MULTIPLIER", dragMultiplier);
  platform->options.getArgs("TWO FLUID MIXTURE CONTINUITY TOLERANCE",
                            mixtureContinuityTolerance);
  platform->options.getArgs("TWO FLUID COUPLING ITERATIONS", couplingIterations);
  platform->options.getArgs("TWO FLUID PRESSURE CORRECTORS", pressureCorrectors);
  projectionOnly = platform->options.compareArgs("TWO FLUID PROJECTION ONLY", "TRUE");
  int bdfOrder = 1;
  platform->options.getArgs("BDF ORDER", bdfOrder);

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
  nekrsCheck(alphaDiffusivity < 0,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "TWO FLUID alphaDiffusivity must be non-negative: %g\n",
             alphaDiffusivity);
  nekrsCheck(mixtureContinuityTolerance <= 0,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "TWO FLUID mixtureContinuityTolerance must be positive: %g\n",
             mixtureContinuityTolerance);
  nekrsCheck(couplingIterations < 1 || pressureCorrectors < 1,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s\n",
             "TWO FLUID couplingIterations and pressureCorrectors must be positive.");
  nekrsCheck(bdfOrder != 1,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "Conservative TWO FLUID volume-fraction transport currently requires tombo1/BDF1; got BDF order %d\n",
             bdfOrder);

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
  o_liquidVelocityTimePrevious =
      platform->device.malloc<dfloat>(liquid->fieldOffsetSum);
  o_gasVelocityTimePrevious =
      platform->device.malloc<dfloat>(gas->fieldOffsetSum);
  o_divergenceLiquid = platform->device.malloc<dfloat>(N);
  o_divergenceGas = platform->device.malloc<dfloat>(N);
  o_alphaGPrevious = platform->device.malloc<dfloat>(N);
  o_alphaGRaw = platform->device.malloc<dfloat>(N);
  o_boundCapacity = platform->device.malloc<dfloat>(N);
  o_alphaGradient = platform->device.malloc<dfloat>(mesh->dim * N);
  o_phaseFluxLiquid = platform->device.malloc<dfloat>(mesh->dim * N);
  o_phaseFluxGas = platform->device.malloc<dfloat>(mesh->dim * N);
  o_divergencePhaseFluxLiquid = platform->device.malloc<dfloat>(N);
  o_divergencePhaseFluxGas = platform->device.malloc<dfloat>(N);
  o_gasContinuityResidual = platform->device.malloc<dfloat>(N);
  o_liquidContinuityResidual = platform->device.malloc<dfloat>(N);
  o_slipVelocity = platform->device.malloc<dfloat>(mesh->dim * liquid->fieldOffset);
  o_mixtureVelocity = platform->device.malloc<dfloat>(mesh->dim * liquid->fieldOffset);
  o_interphaseForce = platform->device.malloc<dfloat>(mesh->dim * liquid->fieldOffset);
  o_continuityResidual = platform->device.malloc<dfloat>(N);

  o_alphaGCouplingPrevious = platform->device.malloc<dfloat>(N);
  o_liquidVelocityCouplingPrevious = platform->device.malloc<dfloat>(liquid->fieldOffsetSum);
  o_gasVelocityCouplingPrevious = platform->device.malloc<dfloat>(gas->fieldOffsetSum);
  o_pressureCouplingPrevious = platform->device.malloc<dfloat>(N);

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
  o_liquidVelocityTimePrevious.copyFrom(liquid->o_U, liquid->fieldOffsetSum);
  o_gasVelocityTimePrevious.copyFrom(gas->o_U, gas->fieldOffsetSum);
  o_alphaGPrevious.copyFrom(o_alphaG, N);
  o_alphaGRaw.copyFrom(o_alphaG, N);
  platform->linAlg->fill(N, 0.0, o_boundCapacity);
  platform->linAlg->fill(mesh->dim * N, 0.0, o_alphaGradient);
  platform->linAlg->fill(mesh->dim * N, 0.0, o_phaseFluxLiquid);
  platform->linAlg->fill(mesh->dim * N, 0.0, o_phaseFluxGas);
  platform->linAlg->fill(N, 0.0, o_divergencePhaseFluxLiquid);
  platform->linAlg->fill(N, 0.0, o_divergencePhaseFluxGas);
  platform->linAlg->fill(N, 0.0, o_gasContinuityResidual);
  platform->linAlg->fill(N, 0.0, o_liquidContinuityResidual);
  platform->linAlg->fill(N, 0.0, liquid->o_div);
  platform->linAlg->fill(N, 0.0, gas->o_div);

  gas->o_P = liquid->o_P;
  gas->o_Pe = liquid->o_Pe;
  liquid->twoFluid = this;

  liquid->userImplicitLinearTerm = [this](double) { return o_implicitLiquid; };
  gas->userImplicitLinearTerm = [this](double) { return o_implicitGas; };

  auto applyOperator = [this](const occa::memory &o_q, occa::memory &o_Aq) {
    pressureCorrectionOperator(o_q, o_Aq);
  };
  auto applyPreconditioner = [this](const occa::memory &o_r,
                                    occa::memory &o_z) {
    auto mesh = liquid->mesh;
    auto o_weakRhs = platform->deviceMemoryPool.reserve<dfloat>(
        liquid->fieldOffset);
    o_weakRhs.copyFrom(o_r, liquid->fieldOffset);
    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_weakRhs);

    auto &pressureOptions = liquid->ellipticSolverP->options();
    const auto initialGuess = pressureOptions.getArgs("INITIAL GUESS");
    pressureOptions.setArgs("INITIAL GUESS", "ZERO");
    platform->linAlg->fill(liquid->fieldOffset, 0.0, o_z);
    liquid->ellipticSolverP->solve(o_pressureCoeff,
                                   o_NULL,
                                   o_weakRhs,
                                   o_z.slice(0, mesh->Nlocal));
    pressureOptions.setArgs("INITIAL GUESS", initialGuess);
  };
  pressureCorrectionSolver = linearSolverFactory<dfloat>::create(
      "flexible gmres+nvector=20",
      "twoFluid exact pressure correction",
      mesh->Nlocal,
      1,
      liquid->fieldOffset,
      mesh->o_LMM,
      true,
      applyOperator,
      applyPreconditioner);

  reportContinuity("initial");
}

void twoFluid_t::beginTimeStep()
{
  o_alphaGPrevious.copyFrom(o_alphaG, liquid->fieldOffset);
  // BDF history is physical-time state, not nonlinear-iteration state.  Slot
  // zero in fluidSolver_t::o_U is overwritten by every coupling solve, so keep
  // an immutable copy for all momentum RHS assemblies in this time step.
  o_liquidVelocityTimePrevious.copyFrom(liquid->o_U,
                                        liquid->fieldOffsetSum);
  o_gasVelocityTimePrevious.copyFrom(gas->o_U, gas->fieldOffsetSum);
}

void twoFluid_t::updatePhaseFluxes()
{
  auto mesh = liquid->mesh;
  if (alphaDiffusivity > 0) {
    launchKernel("core-gradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 liquid->fieldOffset,
                 o_alphaG,
                 o_alphaGradient);
    oogs::startFinish(o_alphaGradient,
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
                                o_alphaGradient);
  }
  launchKernel("twoFluid::phaseFluxes",
               mesh->Nlocal,
               liquid->fieldOffset,
               alphaDiffusivity,
               o_alphaL,
               o_alphaG,
               o_alphaGradient,
               liquid->o_U,
               gas->o_U,
               o_phaseFluxLiquid,
               o_phaseFluxGas,
               o_mixtureVelocity);
}

void twoFluid_t::advanceVolumeFraction(int couplingIteration)
{
  auto mesh = liquid->mesh;
  const auto comm = platform->comm.mpiComm();

  currentCouplingIteration = couplingIteration;
  o_alphaGCouplingPrevious.copyFrom(o_alphaG, liquid->fieldOffset);
  o_liquidVelocityCouplingPrevious.copyFrom(liquid->o_U, liquid->fieldOffsetSum);
  o_gasVelocityCouplingPrevious.copyFrom(gas->o_U, gas->fieldOffsetSum);
  o_pressureCouplingPrevious.copyFrom(liquid->o_P, liquid->fieldOffset);

  auto o_alphaGIterationPrevious = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffset);
  o_alphaGIterationPrevious.copyFrom(o_alphaG, liquid->fieldOffset);

  updatePhaseFluxes();
  weakDivergence(o_phaseFluxGas, o_divergencePhaseFluxGas);
  launchKernel("twoFluid::advanceVolumeFraction",
               mesh->Nlocal,
               liquid->dt[0],
               alphaFloor,
               o_alphaGPrevious,
               o_divergencePhaseFluxGas,
               o_alphaGRaw,
               o_alphaG,
               o_alphaL);

  const dfloat rawIntegral = platform->linAlg->innerProd(mesh->Nlocal, mesh->o_LMM, o_alphaGRaw, comm);
  const dfloat boundedIntegral = platform->linAlg->innerProd(mesh->Nlocal, mesh->o_LMM, o_alphaG, comm);
  const dfloat correction = rawIntegral - boundedIntegral;
  const dfloat tolerance = 100 * std::numeric_limits<dfloat>::epsilon() * mesh->volume;

  if (std::abs(correction) > tolerance) {
    const dfloat direction = correction > 0 ? 1.0 : -1.0;
    launchKernel("twoFluid::boundCapacity",
                 mesh->Nlocal,
                 alphaFloor,
                 direction,
                 o_alphaG,
                 o_boundCapacity);
    const dfloat capacityIntegral = platform->linAlg->innerProd(mesh->Nlocal, mesh->o_LMM, o_boundCapacity, comm);
    nekrsCheck(capacityIntegral <= 0 || std::abs(correction) > capacityIntegral * (1 + 1e-10),
               comm,
               EXIT_FAILURE,
               "Cannot conservatively bound gas volume fraction: correction=%g capacity=%g\n",
               correction,
               capacityIntegral);
    launchKernel("twoFluid::correctBoundedAlpha",
                 mesh->Nlocal,
                 correction / capacityIntegral,
                 o_boundCapacity,
                 o_alphaG,
                 o_alphaL);
  }

  platform->linAlg->axpby(mesh->Nlocal, 1.0, o_alphaG, -1.0, o_alphaGIterationPrevious);
  const dfloat deltaL2 = platform->linAlg->weightedNorm2(mesh->Nlocal, mesh->o_LMM, o_alphaGIterationPrevious, comm);
  const dfloat alphaL2 = platform->linAlg->weightedNorm2(mesh->Nlocal, mesh->o_LMM, o_alphaG, comm);
  alphaCouplingRelativeL2 = deltaL2 / std::max(alphaL2, std::numeric_limits<dfloat>::epsilon());
}

void twoFluid_t::updateAdvectionCoordinates()
{
  auto update = [&](fluidSolver_t *phase) {
    const int relative = 0;
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
               o_alphaL,
               o_alphaG,
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
  liquid->makeForcing(o_liquidVelocityTimePrevious, false);
  gas->makeForcing(o_gasVelocityTimePrevious, false);
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
               o_alphaL,
               o_alphaG,
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
               o_alphaL,
               o_alphaG,
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
  platform->linAlg->fill(mesh->Nlocal, 0.0, o_divMixture);
  const dfloat g0idt = *liquid->g0 / liquid->dt[0];
  launchKernel("fluidSolver_t::pressureAddQtl",
               mesh->Nlocal,
               mesh->o_LMM,
               g0idt,
               o_divMixture,
               o_pRhs);

  updatePhaseFluxes();
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

void twoFluid_t::applyHomogeneousVelocityMask(
    fluidSolver_t *phase,
    occa::memory o_velocityCorrection)
{
  if (phase->ellipticSolver.size() == 1) {
    phase->ellipticSolver.at(0)->applyMask(o_velocityCorrection);
    return;
  }

  for (int component = 0; component < phase->mesh->dim; ++component) {
    auto o_component = o_velocityCorrection.slice(
        component * phase->fieldOffset, phase->mesh->Nlocal);
    phase->ellipticSolver.at(component)->applyMask(o_component);
  }
}

void twoFluid_t::pressureCorrectionFlux(const occa::memory &o_phi,
                                        occa::memory o_deltaLiquid,
                                        occa::memory o_deltaGas,
                                        occa::memory o_deltaMixture)
{
  auto mesh = liquid->mesh;
  auto o_weakGradient = platform->deviceMemoryPool.reserve<dfloat>(
      liquid->fieldOffsetSum);
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

  platform->linAlg->fill(liquid->fieldOffsetSum, 0.0, o_deltaLiquid);
  platform->linAlg->fill(gas->fieldOffsetSum, 0.0, o_deltaGas);
  launchKernel("twoFluid::correctPhaseVelocities",
               mesh->Nlocal,
               liquid->fieldOffset,
               o_pressureResponseLiquid,
               o_pressureResponseGas,
               o_weakGradient,
               o_deltaLiquid,
               o_deltaGas);
  applyHomogeneousVelocityMask(liquid, o_deltaLiquid);
  applyHomogeneousVelocityMask(gas, o_deltaGas);

  launchKernel("twoFluid::mixtureFlux",
               mesh->Nlocal,
               liquid->fieldOffset,
               o_alphaL,
               o_alphaG,
               o_deltaLiquid,
               o_deltaGas,
               o_deltaMixture);
}

void twoFluid_t::pressureCorrectionOperator(const occa::memory &o_phi,
                                            occa::memory o_Aphi)
{
  auto mesh = liquid->mesh;
  auto o_deltaLiquid = platform->deviceMemoryPool.reserve<dfloat>(
      liquid->fieldOffsetSum);
  auto o_deltaGas = platform->deviceMemoryPool.reserve<dfloat>(
      gas->fieldOffsetSum);
  auto o_deltaMixture = platform->deviceMemoryPool.reserve<dfloat>(
      liquid->fieldOffsetSum);
  pressureCorrectionFlux(o_phi,
                         o_deltaLiquid,
                         o_deltaGas,
                         o_deltaMixture);
  weakDivergence(o_deltaMixture, o_Aphi);
  // The phase update adds deltaU, so its divergence is subtracted from the
  // positive Schur operator used by the Krylov solve.
  platform->linAlg->scale(mesh->Nlocal, -1.0, o_Aphi);
}

void twoFluid_t::correctMixtureContinuity(double time, const char *stageLabel)
{
  if (!liquid->ellipticSolverP) return;

  auto mesh = liquid->mesh;
  const auto comm = platform->comm.mpiComm();
  const dfloat volumeScale = 1.0 / sqrt(mesh->volume);
  const dfloat eps = std::numeric_limits<dfloat>::epsilon();
  updatePressureResponse(false);

  for (int corrector = 0; corrector < pressureCorrectors; ++corrector) {
    updatePhaseFluxes();

    // Measure the exact mixture divergence used by the final continuity
    // diagnostic before applying this pressure correction.
    auto o_preRm = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffset);
    weakDivergence(o_mixtureVelocity, o_preRm);
    const dfloat preL2 = volumeScale * platform->linAlg->weightedNorm2(
        mesh->Nlocal, mesh->o_LMM, o_preRm, comm);
    const dfloat preMax = platform->linAlg->amax(mesh->Nlocal, o_preRm, comm);

    if (preL2 <= mixtureContinuityTolerance) {
      if (platform->comm.mpiRank() == 0 && stageLabel) {
        printf("twoFluid pressure %s corrector=%d skipped  L2(Rm)=%.8e "
               "target=%.8e\n",
               stageLabel,
               corrector + 1,
               preL2,
               mixtureContinuityTolerance);
      } else if (platform->comm.mpiRank() == 0) {
        printf("twoFluid pressure iter=%d corrector=%d skipped  L2(Rm)=%.8e "
               "target=%.8e\n",
               currentCouplingIteration + 1,
               corrector + 1,
               preL2,
               mixtureContinuityTolerance);
      }
      break;
    }

    auto o_phi = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffset);
    constexpr int maximumIterations = 50;
    pressureCorrectionSolver->solve(mixtureContinuityTolerance * sqrt(mesh->volume),
                                    maximumIterations,
                                    o_preRm,
                                    o_phi);

    auto o_deltaLiquid = platform->deviceMemoryPool.reserve<dfloat>(
        liquid->fieldOffsetSum);
    auto o_deltaGas = platform->deviceMemoryPool.reserve<dfloat>(
        gas->fieldOffsetSum);
    auto o_deltaMixture = platform->deviceMemoryPool.reserve<dfloat>(
        liquid->fieldOffsetSum);
    pressureCorrectionFlux(o_phi,
                           o_deltaLiquid,
                           o_deltaGas,
                           o_deltaMixture);
    platform->linAlg->axpbyMany(mesh->Nlocal,
                                mesh->dim,
                                liquid->fieldOffset,
                                1.0,
                                o_deltaLiquid,
                                1.0,
                                liquid->o_U);
    platform->linAlg->axpbyMany(mesh->Nlocal,
                                mesh->dim,
                                gas->fieldOffset,
                                1.0,
                                o_deltaGas,
                                1.0,
                                gas->o_U);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_phi, 1.0, liquid->o_P);

    auto o_Aphi = platform->deviceMemoryPool.reserve<dfloat>(
        liquid->fieldOffset);
    pressureCorrectionOperator(o_phi, o_Aphi);
    auto o_divCorrection = platform->deviceMemoryPool.reserve<dfloat>(
        liquid->fieldOffset);
    weakDivergence(o_deltaMixture, o_divCorrection);
    platform->linAlg->axpby(mesh->Nlocal,
                            1.0,
                            o_Aphi,
                            1.0,
                            o_divCorrection);
    const dfloat operatorNorm = platform->linAlg->weightedNorm2(
        mesh->Nlocal, mesh->o_LMM, o_Aphi, comm);
    const dfloat operatorConsistency = platform->linAlg->weightedNorm2(
        mesh->Nlocal, mesh->o_LMM, o_divCorrection, comm) /
        std::max(operatorNorm, eps);

    updatePhaseFluxes();
    auto o_postRm = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffset);
    weakDivergence(o_mixtureVelocity, o_postRm);
    dfloat postL2 = volumeScale * platform->linAlg->weightedNorm2(
        mesh->Nlocal, mesh->o_LMM, o_postRm, comm);
    dfloat postMax = platform->linAlg->amax(mesh->Nlocal, o_postRm, comm);
    const bool accepted = postL2 < preL2;
    if (!accepted) {
      // A Krylov solve that reaches its iteration limit can produce a trial
      // correction worse than the current state. Restore the phase velocities
      // and pressure exactly rather than allowing that error to accumulate.
      platform->linAlg->axpbyMany(mesh->Nlocal,
                                  mesh->dim,
                                  liquid->fieldOffset,
                                  -1.0,
                                  o_deltaLiquid,
                                  1.0,
                                  liquid->o_U);
      platform->linAlg->axpbyMany(mesh->Nlocal,
                                  mesh->dim,
                                  gas->fieldOffset,
                                  -1.0,
                                  o_deltaGas,
                                  1.0,
                                  gas->o_U);
      platform->linAlg->axpby(mesh->Nlocal, -1.0, o_phi, 1.0, liquid->o_P);
      updatePhaseFluxes();
      postL2 = preL2;
      postMax = preMax;
    }

    if (platform->comm.mpiRank() == 0 && stageLabel) {
      printf("twoFluid pressure %s corrector=%d  preL2(Rm)=%.8e preMax|Rm|=%.8e  "
             "postL2(Rm)=%.8e postMax|Rm|=%.8e  "
             "kspIters=%d kspResidual=%.8e operatorConsistency=%.8e accepted=%d\n",
             stageLabel,
             corrector + 1,
             preL2,
             preMax,
             postL2,
             postMax,
             pressureCorrectionSolver->nIter(),
             pressureCorrectionSolver->finalResidualNorm(),
             operatorConsistency,
             accepted);
    } else if (platform->comm.mpiRank() == 0) {
      printf("twoFluid pressure iter=%d corrector=%d  preL2(Rm)=%.8e preMax|Rm|=%.8e  "
             "postL2(Rm)=%.8e postMax|Rm|=%.8e  "
             "kspIters=%d kspResidual=%.8e operatorConsistency=%.8e accepted=%d\n",
             currentCouplingIteration + 1,
             corrector + 1,
             preL2,
             preMax,
             postL2,
             postMax,
             pressureCorrectionSolver->nIter(),
             pressureCorrectionSolver->finalResidualNorm(),
             operatorConsistency,
             accepted);
    }

    if (!accepted || postL2 <= mixtureContinuityTolerance) break;
  }

  updateDiagnostics();
  if (stageLabel) {
    reportContinuity(stageLabel);
  } else {
    reportCouplingIteration(currentCouplingIteration);
  }
}

void twoFluid_t::updateDiagnostics()
{
  auto mesh = liquid->mesh;
  updatePhaseFluxes();
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
  weakDivergence(o_phaseFluxLiquid, o_divergencePhaseFluxLiquid);
  weakDivergence(o_phaseFluxGas, o_divergencePhaseFluxGas);
  launchKernel("twoFluid::massResidual",
               mesh->Nlocal,
               1.0 / liquid->dt[0],
               o_alphaGPrevious,
               o_alphaG,
               o_divergencePhaseFluxLiquid,
               o_divergencePhaseFluxGas,
               o_liquidContinuityResidual,
               o_gasContinuityResidual,
               o_continuityResidual);
}

void twoFluid_t::weakDivergence(const occa::memory &o_velocity,
                                occa::memory o_divergence)
{
  // This is the assembled weak derivative D^T W followed by mass inversion.
  // With periodic or zero-normal-flux boundaries it represents
  // -div(o_velocity), not the strong positive divergence.  Its sign is
  // immaterial when enforcing a zero mixture residual, but transport updates
  // and physical phase-balance diagnostics must account for it explicitly.
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

void twoFluid_t::reportCouplingIteration(int couplingIteration)
{
  if (couplingIteration < 0) return;

  auto mesh = liquid->mesh;
  const auto comm = platform->comm.mpiComm();
  const dfloat eps = std::numeric_limits<dfloat>::epsilon();

  auto vectorRelativeL2 = [&](const occa::memory &o_current,
                              const occa::memory &o_previous,
                              dlong fieldOffsetSum) {
    auto o_delta = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
    o_delta.copyFrom(o_previous, fieldOffsetSum);
    platform->linAlg->axpbyMany(mesh->Nlocal,
                                mesh->dim,
                                liquid->fieldOffset,
                                1.0,
                                o_current,
                                -1.0,
                                o_delta);
    const dfloat deltaNorm = platform->linAlg->weightedNorm2Many(
        mesh->Nlocal,
        mesh->dim,
        liquid->fieldOffset,
        mesh->o_LMM,
        o_delta,
        comm);
    const dfloat currentNorm = platform->linAlg->weightedNorm2Many(
        mesh->Nlocal,
        mesh->dim,
        liquid->fieldOffset,
        mesh->o_LMM,
        o_current,
        comm);
    return deltaNorm / std::max(currentNorm, eps);
  };

  auto scalarRelativeL2 = [&](const occa::memory &o_current,
                              const occa::memory &o_previous) {
    auto o_delta = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffset);
    o_delta.copyFrom(o_previous, liquid->fieldOffset);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_current, -1.0, o_delta);
    const dfloat deltaNorm = platform->linAlg->weightedNorm2(
        mesh->Nlocal, mesh->o_LMM, o_delta, comm);
    const dfloat currentNorm = platform->linAlg->weightedNorm2(
        mesh->Nlocal, mesh->o_LMM, o_current, comm);
    return deltaNorm / std::max(currentNorm, eps);
  };

  const dfloat liquidRelativeL2 = vectorRelativeL2(
      liquid->o_U, o_liquidVelocityCouplingPrevious, liquid->fieldOffsetSum);
  const dfloat gasRelativeL2 = vectorRelativeL2(
      gas->o_U, o_gasVelocityCouplingPrevious, gas->fieldOffsetSum);
  const dfloat pressureRelativeL2 = scalarRelativeL2(
      liquid->o_P, o_pressureCouplingPrevious);

  const dfloat maxLiquid = platform->linAlg->amax(
      mesh->Nlocal, o_liquidContinuityResidual, comm);
  const dfloat maxGas = platform->linAlg->amax(
      mesh->Nlocal, o_gasContinuityResidual, comm);
  const dfloat maxMixture = platform->linAlg->amax(
      mesh->Nlocal, o_continuityResidual, comm);
  const dfloat volumeScale = 1.0 / sqrt(mesh->volume);
  const dfloat l2Liquid = volumeScale * platform->linAlg->weightedNorm2(
      mesh->Nlocal, mesh->o_LMM, o_liquidContinuityResidual, comm);
  const dfloat l2Gas = volumeScale * platform->linAlg->weightedNorm2(
      mesh->Nlocal, mesh->o_LMM, o_gasContinuityResidual, comm);
  const dfloat l2Mixture = volumeScale * platform->linAlg->weightedNorm2(
      mesh->Nlocal, mesh->o_LMM, o_continuityResidual, comm);

  if (platform->comm.mpiRank() == 0) {
    printf("twoFluid iter=%d  alphaG relL2=%.8e  UL relL2=%.8e  UG relL2=%.8e  "
           "p relL2=%.8e  L2(Rl)=%.8e  L2(Rg)=%.8e  L2(Rm)=%.8e  "
           "max|Rl|=%.8e  max|Rg|=%.8e  max|Rm|=%.8e\n",
           couplingIteration + 1,
           alphaCouplingRelativeL2,
           liquidRelativeL2,
           gasRelativeL2,
           pressureRelativeL2,
           l2Liquid,
           l2Gas,
           l2Mixture,
           maxLiquid,
           maxGas,
           maxMixture);
  }
}

void twoFluid_t::reportContinuity(const char *label)
{
  auto mesh = liquid->mesh;
  updateDiagnostics();
  const auto comm = platform->comm.mpiComm();
  const auto maxLiquid = platform->linAlg->amax(mesh->Nlocal, o_liquidContinuityResidual, comm);
  const auto maxGas = platform->linAlg->amax(mesh->Nlocal, o_gasContinuityResidual, comm);
  const auto maxMixture = platform->linAlg->amax(mesh->Nlocal, o_continuityResidual, comm);
  const auto volumeScale = 1.0 / sqrt(mesh->volume);
  const auto l2Liquid = volumeScale * platform->linAlg->weightedNorm2(
      mesh->Nlocal, mesh->o_LMM, o_liquidContinuityResidual, comm);
  const auto l2Gas = volumeScale * platform->linAlg->weightedNorm2(
      mesh->Nlocal, mesh->o_LMM, o_gasContinuityResidual, comm);
  const auto l2Mixture = volumeScale * platform->linAlg->weightedNorm2(
      mesh->Nlocal, mesh->o_LMM, o_continuityResidual, comm);
  const auto alphaMin = platform->linAlg->min(mesh->Nlocal, o_alphaG, comm);
  const auto alphaMax = platform->linAlg->max(mesh->Nlocal, o_alphaG, comm);
  if (platform->comm.mpiRank() == 0) {
    printf("twoFluid mass%s%s max|Rl|=%.8e max|Rg|=%.8e max|Rm|=%.8e "
           "L2(Rl)=%.8e L2(Rg)=%.8e L2(Rm)=%.8e alphaG=[%.8e,%.8e]\n",
           label ? " " : "",
           label ? label : "",
           maxLiquid,
           maxGas,
           maxMixture,
           l2Liquid,
           l2Gas,
           l2Mixture,
           alphaMin,
           alphaMax);
  }
}

void registerTwoFluidKernels()
{
  occa::properties props = platform->kernelInfo;
  const std::string dir = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/twoFluid/");
  for (const auto &name : {"diagnosticFields",
                           "advanceVolumeFraction",
                           "boundCapacity",
                           "correctPhaseVelocities",
                           "correctBoundedAlpha",
                           "dragSource",
                           "massResidual",
                           "mixtureFlux",
                           "phaseFluxes",
                           "pressureCoefficient",
                           "pressureResponse"})
    platform->kernelRequests.add("twoFluid::" + std::string(name),
                                 dir + std::string(name) + ".okl",
                                 props);
}
