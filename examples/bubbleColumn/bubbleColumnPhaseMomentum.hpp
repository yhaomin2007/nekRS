#pragma once

namespace bubbleColumn::phaseMomentum {
static occa::kernel mixtureKernel, transportKernel, liquidRhsKernel, pressureResponseKernel;
static deviceMemory<dfloat> stress, convection, convectionEXT, divStress, liquidRhs;
static deviceMemory<dfloat> deltaUg, deltaUl;
static ExplicitHistory liquidConvectionHistory;
static dfloat predictorDivergenceRms;

inline bool reportContinuity()
{
  return p.stabilityMonitorEnabled != 0.0 && p.stabilityMonitorInterval > 0 &&
         (nrs->tstep == 1 || nrs->tstep % p.stabilityMonitorInterval == 0);
}

inline dfloat divergenceRms()
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  auto defect = divStress.slice(0, offset); // liquid stress is no longer needed
  opSEM::strongDivergence(mesh, offset, nrs->fluid->o_U, defect);
  platform->linAlg->axpby(mesh->Nlocal, -1.0, nrs->fluid->o_div, 1.0, defect);
  return platform->linAlg->weightedNorm2Many(mesh->Nlocal, 1, offset,
      mesh->o_LMM, defect, platform->comm.mpiComm()) / std::sqrt(mesh->volume);
}

inline void registerKernels(deviceKernelProperties &info)
{
  const std::string request = "bubbleColumn::phaseMomentum";
  if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    platform->kernelRequests.add(request, "bubbleColumnPhaseMomentum.okl", info);
  } else {
    mixtureKernel = platform->kernelRequests.load(request, "phaseMixtureVelocity");
    transportKernel = platform->kernelRequests.load(request, "phaseLiquidTransport");
    liquidRhsKernel = platform->kernelRequests.load(request, "phaseLiquidRhs");
    pressureResponseKernel = platform->kernelRequests.load(request, "phasePressureResponse");
  }
}

inline void allocate()
{
  const dlong offset = nrs->fieldOffset;
  stress.resize(9 * offset);
  convection.resize(3 * offset);
  convectionEXT.resize(3 * offset);
  divStress.resize(3 * offset);
  liquidRhs.resize(3 * offset);
  deltaUg.resize(3 * offset);
  deltaUl.resize(3 * offset);
}

inline void rebuildMixture()
{
  reconstructLiquidVelocity(); // refresh both phase vectors from the scalars
  mixtureKernel(nrs->meshV->Nlocal, nrs->fieldOffset, p.rhoLiquid, p.rhoGas,
      nrs->scalar->o_solution("alpha"), o_ug, o_ul, nrs->fluid->o_U);
}

inline void addLiquidSources()
{
  auto mesh = nrs->meshV;
  const dlong N = mesh->Nlocal, offset = nrs->fieldOffset;
  auto alpha = nrs->scalar->o_solution("alpha");
  const dfloat cVM = p.virtualMassEnabled * p.virtualMassCoefficient * p.rhoLiquid;
  transportKernel(N, offset, p.muLiquid, alpha, o_ul, o_gradUl, convection, stress);
  liquidConvectionHistory.apply(convection, convectionEXT, 3, true);
  for (int c = 0; c < 3; ++c) {
    auto row = stress.slice(3 * c * offset, 3 * offset);
    auto divergence = divStress.slice(c * offset, offset);
    opSEM::strongDivergence(mesh, offset, row, divergence);
  }
  liquidRhsKernel(N, offset, p.rhoLiquid, p.rhoGas + cVM, cVM, p.alphaFloor,
      p.gravity[0], p.gravity[1], p.gravity[2], alpha, o_ug, o_ul, o_dragLambda,
      o_ugSource, o_liquidAccelerationEXT, convectionEXT, divStress, liquidRhs);
  for (int c = 0; c < 3; ++c) {
    nrs->scalar->o_explicitTerms(liquidScalarNames[c]).copyFrom(liquidRhs, N, 0, c * offset);
  }
}

inline void applyPressure(double time)
{
  auto mesh = nrs->meshV;
  const dlong N = mesh->Nlocal, offset = nrs->fieldOffset;
  const dfloat cVM = p.virtualMassEnabled * p.virtualMassCoefficient * p.rhoLiquid;
  evaluateGasPressureGradient();
  pressureResponseKernel(N, offset, p.rhoLiquid, p.rhoGas + cVM, cVM, p.alphaFloor,
      nrs->dt[0] / nrs->g0, nrs->scalar->o_solution("alpha"), o_gradPStrong, deltaUg, deltaUl);
  const char *gasNames[] = {"ugx", "ugy", "ugz"};
  for (int c = 0; c < 3; ++c) {
    auto ug = nrs->scalar->o_solution(gasNames[c]);
    auto ul = nrs->scalar->o_solution(liquidScalarNames[c]);
    platform->linAlg->axpby(N, 1.0, deltaUg.slice(c * offset, N), 1.0, ug);
    platform->linAlg->axpby(N, 1.0, deltaUl.slice(c * offset, N), 1.0, ul);
  }
  nrs->scalar->applyDirichlet(time);
  rebuildMixture();
  if (reportContinuity()) {
    const dfloat corrected = divergenceRms();
    if (platform->comm.mpiRank() == 0) {
      printf("method1PhaseContinuity step=%d time=%.8e rms(before,after)=(%.8e,%.8e)\n",
          nrs->tstep, time, predictorDivergenceRms, corrected);
    }
  }
}
} // namespace bubbleColumn::phaseMomentum
