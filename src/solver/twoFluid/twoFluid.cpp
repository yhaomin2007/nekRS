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

  nekrsCheck(alphaG <= 0 || alphaG >= 1,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "TWO FLUID gasVolumeFraction must lie strictly between zero and one: %g\n",
             alphaG);
  nekrsCheck(bubbleDiameter <= 0,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "TWO FLUID bubbleDiameter must be positive: %g\n",
             bubbleDiameter);

  alphaL = 1 - alphaG;
  const auto N = liquid->fieldOffset;
  o_alphaG = platform->device.malloc<dfloat>(N);
  o_alphaL = platform->device.malloc<dfloat>(N);
  o_drag = platform->device.malloc<dfloat>(N);
  o_implicitLiquid = platform->device.malloc<dfloat>(N);
  o_implicitGas = platform->device.malloc<dfloat>(N);
  o_pressureCoeff = platform->device.malloc<dfloat>(N);
  o_mixtureFlux = platform->device.malloc<dfloat>(mesh->dim * liquid->fieldOffset);
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

  // Both phases see the same pressure storage.  Only the primary phase owns
  // and destroys the pressure elliptic solver.
  gas->o_P = liquid->o_P;
  gas->o_Pe = liquid->o_Pe;
  liquid->twoFluid = this;

  liquid->userImplicitLinearTerm = [this](double) { return o_implicitLiquid; };
  gas->userImplicitLinearTerm = [this](double) { return o_implicitGas; };
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
  launchKernel("twoFluid::dragSource",
               mesh->Nlocal,
               liquid->fieldOffset,
               alphaFloor,
               alphaL,
               alphaG,
               bubbleDiameter,
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

void twoFluid_t::solvePressure(double time, int stage)
{
  if (!liquid->ellipticSolverP) return;

  auto mesh = liquid->mesh;
  platform->timer.tic("mixture pressureSolve");

  // Convert the assembled weak phase forcing to a continuous nodal field in
  // the same way as fluidSolver_t::solvePressure, then construct the mixture
  // volume flux alpha_l F_l + alpha_g F_g.
  auto o_fL = platform->deviceMemoryPool.reserve<dfloat>(liquid->fieldOffsetSum);
  auto o_fG = platform->deviceMemoryPool.reserve<dfloat>(gas->fieldOffsetSum);
  o_fL.copyFrom(liquid->o_JwF, liquid->fieldOffsetSum);
  o_fG.copyFrom(gas->o_JwF, gas->fieldOffsetSum);
  oogs::startFinish(o_fL, mesh->dim, liquid->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs3);
  oogs::startFinish(o_fG, mesh->dim, gas->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs3);
  platform->linAlg->axmyVector(mesh->Nlocal, liquid->fieldOffset, 0, 1.0, mesh->o_invLMM, o_fL);
  platform->linAlg->axmyVector(mesh->Nlocal, gas->fieldOffset, 0, 1.0, mesh->o_invLMM, o_fG);

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

  launchKernel("twoFluid::pressureCoefficient",
               mesh->Nlocal,
               alphaL,
               alphaG,
               liquid->o_rho,
               gas->o_rho,
               o_pressureCoeff);

  liquid->ellipticSolverP->solve(o_pressureCoeff,
                                 o_NULL,
                                 o_pRhs,
                                 liquid->o_P.slice(0, mesh->Nlocal));
  platform->timer.toc("mixture pressureSolve");
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
  launchKernel("core-divergenceVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               liquid->fieldOffset,
               o_mixtureVelocity,
               o_continuityResidual);
  oogs::startFinish(o_continuityResidual, 1, liquid->fieldOffset,
                    ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_continuityResidual);
}

void twoFluid_t::reportContinuity()
{
  auto mesh = liquid->mesh;
  updateDiagnostics();
  const auto norm = platform->linAlg->weightedNorm2(mesh->Nlocal,
                                                    mesh->o_LMM,
                                                    o_continuityResidual,
                                                    platform->comm.mpiComm());
  if (platform->comm.mpiRank() == 0)
    printf("twoFluid mixture-divergence norm: %.8e\n", norm);
}

void registerTwoFluidKernels()
{
  occa::properties props = platform->kernelInfo;
  const std::string dir = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/twoFluid/");
  for (const auto &name : {"diagnosticFields",
                           "dragSource",
                           "mixtureFlux",
                           "mixtureVelocity",
                           "pressureCoefficient"})
    platform->kernelRequests.add("twoFluid::" + std::string(name),
                                 dir + std::string(name) + ".okl",
                                 props);
}
