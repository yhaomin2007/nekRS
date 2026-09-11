#pragma once

#include <cmath>

#include "lowPassFilter.hpp"
#include "opSEM.hpp"

namespace bubbleColumn
{
struct Parameters {
  dfloat rhoLiquid;
  dfloat rhoGas;
  dfloat muLiquid;
  dfloat muGas;
  dfloat alphaInitial;
  dfloat alphaInlet;
  dfloat ugInlet;
  dfloat initialPlumeHeight;
  dfloat initialPlumeThickness;
  dfloat gravity[3];
  dfloat alphaFloor;
  dfloat alphaClippingEnabled;
  dfloat alphaMinimum;
  dfloat alphaMaximum;
  dfloat subtractAlphaDiffusion;
  dfloat subtractQgDiffusion[3];
  dfloat zRampBottom;
  dfloat zRampTop;
  dfloat outletDampingFactor;
  dfloat smoothGasVelocityMaskEnabled;
  dfloat gasVelocityClipEnabled;
  dfloat gasVelocityMaximum;
  dfloat gasMomentumCutoff;
  dfloat gasMomentumFullyActive;
  dfloat gasPressureEnabled;
  dfloat dragEnabled;
  dfloat bubbleDiameter;
  dfloat virtualMassEnabled;
  dfloat virtualMassCoefficient;
  dfloat divergenceFilterEnabled;
  int divergenceFilterModes;
  dfloat divergenceFilterStrength;
  dfloat divergenceExtrapolationEnabled;
  dfloat divergenceRampEnabled;
  int divergenceRampSteps;
  dfloat stabilityMonitorEnabled;
  int stabilityMonitorInterval;
};

static Parameters p;
static deviceMemory<dfloat> o_ug;
static deviceMemory<dfloat> o_ul;
static deviceMemory<dfloat> o_ulPrevious;
static deviceMemory<dfloat> o_ugPrevious;
static deviceMemory<dfloat> o_gradUl;
static deviceMemory<dfloat> o_virtualMassRelativeAcceleration;
static deviceMemory<dfloat> o_gradAlpha;
static deviceMemory<dfloat> o_gradAlphaAdvection;
static deviceMemory<dfloat> o_qg;
static deviceMemory<dfloat> o_gradQgAdvection;
static deviceMemory<dfloat> o_qgAdvectionFlux;
static deviceMemory<dfloat> o_divQgAdvectionFlux;
static deviceMemory<dfloat> o_gradUg;
static deviceMemory<dfloat> o_gradUgAdvection;
static deviceMemory<dfloat> o_gradP;
static deviceMemory<dfloat> o_rhoM;
static deviceMemory<dfloat> o_muM;
static deviceMemory<dfloat> o_divSource;
static deviceMemory<dfloat> o_divPrevious;
static deviceMemory<dfloat> o_divOlder;
static deviceMemory<dfloat> o_divExtrapolated;
static int divergenceHistoryCount = 0;
static deviceMemory<dfloat> o_divFilterWork;
static occa::memory o_divFilterMatrix;
static deviceMemory<dfloat> o_alphaDiffusionFlux;
static deviceMemory<dfloat> o_alphaDiffusionDivergence;
static deviceMemory<dfloat> o_qgDiffusionDivergence;
static deviceMemory<dfloat> o_scalarDiffusionBase;
static deviceMemory<dfloat> o_outletRampFactor;
static deviceMemory<dfloat> o_alphaExplicitBase;
static deviceMemory<dfloat> o_alphaRegularizationSource;
static deviceMemory<dfloat> o_driftStress;
static deviceMemory<dfloat> o_divDriftStress;
static deviceMemory<dfloat> o_gasStress;
static deviceMemory<dfloat> o_divGasStress;
static deviceMemory<dfloat> o_alphaSource;
static deviceMemory<dfloat> o_qgSource;
static deviceMemory<dfloat> o_dragLambda;
static deviceMemory<dfloat> o_mixtureForce;
static deviceMemory<dfloat> o_monitorMagnitude;
static deviceMemory<dfloat> o_gasPressureAccelerationActive;
static deviceMemory<dfloat> o_gasPressureAccelerationInactive;
static deviceMemory<dfloat> o_gasCfl;
static deviceMemory<dfloat> o_inverseGllSpacing;
static deviceMemory<int> o_inletBoundaryID;
static deviceMemory<dlong> o_scalarFieldOffsetScan;
static deviceMemory<dfloat> o_surfaceOne;
static deviceMemory<dfloat> o_surfaceScalar;
static occa::kernel reconstructGasVelocityKernel;
static occa::kernel postProcessGasFluxKernel;
static occa::kernel clipAlphaKernel;
static occa::kernel initializePlumeKernel;
static occa::kernel buildLiquidVelocityKernel;
static occa::kernel updateVirtualMassHistoryKernel;
static occa::kernel buildDivergenceFromAlphaRhsKernel;
static occa::kernel buildEquationTermsKernel;
static occa::kernel buildMixtureForceKernel;
static occa::kernel buildGasPressureAccelerationMonitorKernel;
static occa::kernel buildGasFluxConsistencyMonitorKernel;
static occa::kernel buildOutletRampFactorKernel;

inline void registerKernels(deviceKernelProperties &kernelInfo)
{
  const std::string request = "bubbleColumn::equations";
  // This is standalone OKL source.  The .okl suffix is required so OCCA
  // translates @kernel/@globalPtr before invoking the HIP compiler.
  const std::string fileName = "bubbleColumnEquations.okl";
  if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    platform->kernelRequests.add(request, fileName, kernelInfo);
  } else {
    reconstructGasVelocityKernel =
        platform->kernelRequests.load(request, "reconstructGasVelocity");
    postProcessGasFluxKernel =
        platform->kernelRequests.load(request, "postProcessGasFlux");
    clipAlphaKernel = platform->kernelRequests.load(request, "clipAlpha");
    initializePlumeKernel = platform->kernelRequests.load(request, "initializePlume");
    buildLiquidVelocityKernel = platform->kernelRequests.load(request, "buildLiquidVelocity");
    updateVirtualMassHistoryKernel =
        platform->kernelRequests.load(request, "updateVirtualMassHistory");
    buildDivergenceFromAlphaRhsKernel =
        platform->kernelRequests.load(request, "buildDivergenceFromAlphaRhs");
    buildEquationTermsKernel = platform->kernelRequests.load(request, "buildEquationTerms");
    buildMixtureForceKernel = platform->kernelRequests.load(request, "buildMixtureForce");
    buildGasPressureAccelerationMonitorKernel =
        platform->kernelRequests.load(request, "buildGasPressureAccelerationMonitor");
    buildGasFluxConsistencyMonitorKernel =
        platform->kernelRequests.load(request, "buildGasFluxConsistencyMonitor");
    buildOutletRampFactorKernel =
        platform->kernelRequests.load(request, "buildOutletRampFactor");
  }
}

inline void allocate()
{
  const dlong offset = nrs->fieldOffset;
  nekrsCheck(p.divergenceFilterEnabled != 0.0
                 && (p.divergenceFilterStrength < 0.0
                     || p.divergenceFilterStrength > 1.0),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "divergenceFilterStrength must be in [0,1], but is %g\n",
             p.divergenceFilterStrength);
  nekrsCheck(p.gasMomentumCutoff < 0.0
                 || p.gasMomentumFullyActive <= p.gasMomentumCutoff,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "gasMomentumCutoff must be nonnegative and "
             "gasMomentumFullyActive must exceed it (cutoff=%g, active=%g)\n",
             p.gasMomentumCutoff,
             p.gasMomentumFullyActive);
  nekrsCheck(p.gasVelocityClipEnabled != 0.0 && p.gasVelocityMaximum <= 0.0,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "gasVelocityMaximum must be positive when clipping is enabled, but is %g\n",
             p.gasVelocityMaximum);
  nekrsCheck(p.alphaClippingEnabled != 0.0
                 && (p.alphaMinimum < 0.0 || p.alphaMaximum > 1.0
                     || p.alphaMaximum <= p.alphaMinimum),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "alpha clipping bounds must satisfy 0 <= minimum < maximum <= 1 "
             "(minimum=%g, maximum=%g)\n",
             p.alphaMinimum,
             p.alphaMaximum);
  nekrsCheck(p.divergenceRampEnabled != 0.0 && p.divergenceRampSteps <= 0,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "divergenceRampSteps must be positive when the ramp is enabled, but is %d\n",
             p.divergenceRampSteps);
  nekrsCheck(p.zRampTop <= p.zRampBottom,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "zRampTop must exceed zRampBottom (bottom=%g, top=%g)\n",
             p.zRampBottom,
             p.zRampTop);
  nekrsCheck(p.outletDampingFactor < 1.0,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "outletDampingFactor must be at least 1, but is %g\n",
             p.outletDampingFactor);
  o_ug.resize(3 * offset);
  o_ul.resize(3 * offset);
  o_ulPrevious.resize(3 * offset);
  o_ugPrevious.resize(3 * offset);
  o_gradUl.resize(9 * offset);
  o_virtualMassRelativeAcceleration.resize(3 * offset);
  o_gradAlpha.resize(3 * offset);
  o_gradAlphaAdvection.resize(3 * offset);
  o_qg.resize(3 * offset);
  o_gradQgAdvection.resize(9 * offset);
  o_qgAdvectionFlux.resize(9 * offset);
  o_divQgAdvectionFlux.resize(3 * offset);
  o_gradUg.resize(9 * offset);
  o_gradUgAdvection.resize(9 * offset);
  o_gradP.resize(3 * offset);
  o_rhoM.resize(offset);
  o_muM.resize(offset);
  o_divSource.resize(offset);
  o_divPrevious.resize(offset);
  o_divOlder.resize(offset);
  o_divExtrapolated.resize(offset);
  platform->linAlg->fill(offset, 0.0, o_divPrevious);
  platform->linAlg->fill(offset, 0.0, o_divOlder);
  platform->linAlg->fill(offset, 0.0, o_divExtrapolated);
  o_divFilterWork.resize(3 * offset);
  o_alphaDiffusionFlux.resize(3 * offset);
  o_alphaDiffusionDivergence.resize(offset);
  o_qgDiffusionDivergence.resize(3 * offset);
  o_scalarDiffusionBase.resize(4 * offset);
  o_outletRampFactor.resize(offset);
  o_alphaExplicitBase.resize(offset);
  o_alphaRegularizationSource.resize(offset);
  o_driftStress.resize(9 * offset);
  o_divDriftStress.resize(3 * offset);
  o_gasStress.resize(9 * offset);
  o_divGasStress.resize(3 * offset);
  o_alphaSource.resize(offset);
  o_qgSource.resize(3 * offset);
  o_dragLambda.resize(offset);
  o_mixtureForce.resize(3 * offset);
  o_monitorMagnitude.resize(offset);
  o_gasPressureAccelerationActive.resize(offset);
  o_gasPressureAccelerationInactive.resize(offset);
  o_gasCfl.resize(nrs->meshV->Nelements);
  o_inverseGllSpacing.resize(nrs->meshV->N + 1);
  std::vector<dfloat> inverseGllSpacing(nrs->meshV->N + 1);
  for (int n = 0; n < nrs->meshV->N + 1; ++n) {
    dfloat spacing;
    if (n == 0) {
      spacing = nrs->meshV->gllz[n + 1] - nrs->meshV->gllz[n];
    } else if (n == nrs->meshV->N) {
      spacing = nrs->meshV->gllz[n] - nrs->meshV->gllz[n - 1];
    } else {
      spacing = 0.5 * (nrs->meshV->gllz[n + 1] - nrs->meshV->gllz[n - 1]);
    }
    inverseGllSpacing[n] = 1.0 / spacing;
  }
  o_inverseGllSpacing.copyFrom(inverseGllSpacing);
  o_inletBoundaryID.resize(1);
  o_inletBoundaryID.copyFrom(std::vector<int>{1});
  o_scalarFieldOffsetScan.resize(1);
  o_scalarFieldOffsetScan.copyFrom(std::vector<dlong>{0});
  const char *scalarNames[4] = {"alpha", "qgx", "qgy", "qgz"};
  for (int i = 0; i < 4; ++i) {
    o_scalarDiffusionBase.copyFrom(
        nrs->scalar->o_diffusionCoeff(scalarNames[i]),
        nrs->meshV->Nlocal,
        i * offset,
        0);
  }
  o_surfaceOne.resize(nrs->meshV->Nlocal);
  o_surfaceScalar.resize(nrs->meshV->Nlocal);
  platform->linAlg->fill(nrs->meshV->Nlocal, 1.0, o_surfaceOne);
  platform->linAlg->fill(offset, 0.0, o_alphaExplicitBase);
  platform->linAlg->fill(offset, 0.0, o_alphaRegularizationSource);

  if (p.divergenceFilterEnabled != 0.0) {
    o_divFilterMatrix = lowPassFilterSetup(nrs->meshV, p.divergenceFilterModes);
  }
}

inline void filterDivergence()
{
  if (p.divergenceFilterEnabled == 0.0) {
    return;
  }

  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;

  // vectorFilterRTHex3D computes
  //   output += -strength * (input - F(input)).
  // Initialize the first output component with q_raw so that it becomes
  //   q_filtered = q_raw - strength * (q_raw - F(q_raw)).
  // The two unused components remain zero.
  platform->linAlg->fill(3 * offset, 0.0, o_divFilterWork);
  o_divFilterWork.copyFrom(o_divSource, Nlocal, 0, 0);
  launchKernel("core-vectorFilterRTHex3D",
               nrs->meshV->Nelements,
               o_divFilterMatrix,
               p.divergenceFilterStrength,
               offset,
               o_divFilterWork,
               o_divFilterWork);
  o_divSource.copyFrom(o_divFilterWork, Nlocal, 0, 0);
}

inline void reconstructGasVelocity()
{
  reconstructGasVelocityKernel(nrs->meshV->Nlocal,
                               nrs->fieldOffset,
                               p.alphaFloor,
                               nrs->scalar->o_solution("alpha"),
                               nrs->scalar->o_solution("qgx"),
                               nrs->scalar->o_solution("qgy"),
                               nrs->scalar->o_solution("qgz"),
                               o_ug);
}

inline void reconstructLiquidVelocity()
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  o_qg.copyFrom(nrs->scalar->o_solution("qgx"), Nlocal, 0 * offset, 0);
  o_qg.copyFrom(nrs->scalar->o_solution("qgy"), Nlocal, 1 * offset, 0);
  o_qg.copyFrom(nrs->scalar->o_solution("qgz"), Nlocal, 2 * offset, 0);
  buildLiquidVelocityKernel(Nlocal,
                            offset,
                            p.rhoLiquid,
                            p.rhoGas,
                            p.alphaFloor,
                            nrs->scalar->o_solution("alpha"),
                            nrs->fluid->o_U,
                            o_qg,
                            o_ul);
}

inline void evaluatePointwiseTerms()
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  auto alpha = nrs->scalar->o_solution("alpha");

  o_qg.copyFrom(nrs->scalar->o_solution("qgx"), mesh->Nlocal, 0 * offset, 0);
  o_qg.copyFrom(nrs->scalar->o_solution("qgy"), mesh->Nlocal, 1 * offset, 0);
  o_qg.copyFrom(nrs->scalar->o_solution("qgz"), mesh->Nlocal, 2 * offset, 0);
  reconstructGasVelocity();
  buildLiquidVelocityKernel(mesh->Nlocal,
                            offset,
                            p.rhoLiquid,
                            p.rhoGas,
                            p.alphaFloor,
                            alpha,
                            nrs->fluid->o_U,
                            o_qg,
                            o_ul);
  // Keep averaged gradients for constitutive terms and diffusion, but use
  // element-local gradients for advection corrections. NekRS's native scalar
  // advection kernel differentiates each element locally; using the default
  // gather-scatter-averaged opSEM gradient here prevents the nominal
  // u_m.grad(scalar) cancellation from matching node-for-node.
  opSEM::strongGrad(mesh, offset, alpha, o_gradAlpha);
  opSEM::strongGrad(mesh, offset, alpha, o_gradAlphaAdvection, false);
  opSEM::strongGradVec(mesh, offset, o_qg, o_gradQgAdvection, false);
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg);
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUgAdvection, false);
  opSEM::strongGradVec(mesh, offset, o_ul, o_gradUl);
  opSEM::strongGrad(mesh, offset, nrs->fluid->o_P, o_gradP);

  // Form each row of q_g tensor-product u_g with the standard linAlg kernels.
  // Besides avoiding a case-specific CUDA kernel, this preserves the component
  // layout expected by strongDivergence: (qx*ugx, qx*ugy, qx*ugz), etc.
  for (int i = 0; i < 3; ++i) {
    const auto qi = o_qg.slice(i * offset, offset);
    for (int j = 0; j < 3; ++j) {
      const auto ugj = o_ug.slice(j * offset, offset);
      auto fluxComponent =
          o_qgAdvectionFlux.slice((3 * i + j) * offset, offset);
      fluxComponent.copyFrom(qi, offset);
      platform->linAlg->axmy(mesh->Nlocal, 1.0, ugj, fluxComponent);
    }
  }
  for (int i = 0; i < 3; ++i) {
    auto flux = o_qgAdvectionFlux.slice(3 * i * offset, 3 * offset);
    auto divergence = o_divQgAdvectionFlux.slice(i * offset, offset);
    opSEM::strongDivergence(mesh, offset, flux, divergence, false);
  }

  buildEquationTermsKernel(mesh->Nlocal,
                           offset,
                           p.rhoLiquid,
                           p.rhoGas,
                           p.muLiquid,
                           p.muGas,
                           p.alphaFloor,
                           p.gasPressureEnabled,
                           p.dragEnabled,
                           p.bubbleDiameter,
                           p.virtualMassEnabled,
                           p.virtualMassCoefficient,
                           p.gravity[0],
                           p.gravity[1],
                           p.gravity[2],
                           alpha,
                           nrs->fluid->o_U,
                           o_ug,
                           o_ul,
                           o_gradAlphaAdvection,
                           o_gradQgAdvection,
                           o_divQgAdvectionFlux,
                           o_gradUgAdvection,
                           o_gradUg,
                           o_virtualMassRelativeAcceleration,
                           o_gradP,
                           o_rhoM,
                           o_muM,
                           o_divSource,
                           o_driftStress,
                           o_gasStress,
                           o_alphaSource,
                           o_qgSource,
                           o_dragLambda);

  // Smoothly increase the implicit mixture viscosity and all implicit scalar
  // diffusion coefficients near the outlet. Do not scale the explicit gas
  // viscous-stress source, which would tighten its timestep restriction.
  // Restore scalar coefficients from their original .par values first so
  // repeated userProperties() calls cannot compound the ramp factor.
  buildOutletRampFactorKernel(mesh->Nlocal,
                              p.zRampBottom,
                              p.zRampTop,
                              p.outletDampingFactor,
                              mesh->o_z,
                              o_outletRampFactor);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, o_outletRampFactor, o_muM);
  const char *scalarNames[4] = {"alpha", "qgx", "qgy", "qgz"};
  for (int i = 0; i < 4; ++i) {
    auto diffusion = nrs->scalar->o_diffusionCoeff(scalarNames[i]);
    diffusion.copyFrom(o_scalarDiffusionBase, mesh->Nlocal, 0, i * offset);
    platform->linAlg->axmy(
        mesh->Nlocal, 1.0, o_outletRampFactor, diffusion);
  }

  for (int i = 0; i < 3; ++i) {
    auto stressRow = o_gasStress.slice(3 * i * offset, 3 * offset);
    auto stressDivergence = o_divGasStress.slice(i * offset, offset);
    opSEM::strongDivergence(mesh, offset, stressRow, stressDivergence);
    platform->linAlg->axpby(mesh->Nlocal,
                            1.0 / p.rhoGas,
                            stressDivergence,
                            1.0,
                            o_qgSource,
                            0,
                            i * offset);
  }
}

inline void initializeHistory()
{
  evaluatePointwiseTerms();
  const dlong offset = nrs->fieldOffset;
  o_ulPrevious.copyFrom(o_ul, 3 * offset);
  o_ugPrevious.copyFrom(o_ug, 3 * offset);
  platform->linAlg->fill(3 * offset, 0.0, o_virtualMassRelativeAcceleration);
}

inline void postProcessGasFlux()
{
  if (p.smoothGasVelocityMaskEnabled == 0.0 && p.gasVelocityClipEnabled == 0.0) {
    return;
  }

  postProcessGasFluxKernel(
      nrs->meshV->Nlocal,
      p.smoothGasVelocityMaskEnabled,
      p.gasVelocityClipEnabled,
      p.gasVelocityMaximum,
      p.alphaFloor,
      p.gasMomentumCutoff,
      p.gasMomentumFullyActive,
      nrs->scalar->o_solution("alpha"),
      nrs->scalar->o_solution("qgx"),
      nrs->scalar->o_solution("qgy"),
      nrs->scalar->o_solution("qgz"));
}

inline void clipAlpha(double, int)
{
  if (p.alphaClippingEnabled == 0.0) {
    return;
  }

  clipAlphaKernel(nrs->meshV->Nlocal,
                  p.alphaMinimum,
                  p.alphaMaximum,
                  nrs->scalar->o_solution("alpha"));
}

inline void updateVirtualMassHistory()
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  auto alpha = nrs->scalar->o_solution("alpha");

  o_qg.copyFrom(nrs->scalar->o_solution("qgx"), mesh->Nlocal, 0 * offset, 0);
  o_qg.copyFrom(nrs->scalar->o_solution("qgy"), mesh->Nlocal, 1 * offset, 0);
  o_qg.copyFrom(nrs->scalar->o_solution("qgz"), mesh->Nlocal, 2 * offset, 0);
  reconstructGasVelocityKernel(mesh->Nlocal,
                               offset,
                               p.alphaFloor,
                               alpha,
                               nrs->scalar->o_solution("qgx"),
                               nrs->scalar->o_solution("qgy"),
                               nrs->scalar->o_solution("qgz"),
                               o_ug);
  buildLiquidVelocityKernel(mesh->Nlocal,
                            offset,
                            p.rhoLiquid,
                            p.rhoGas,
                            p.alphaFloor,
                            alpha,
                            nrs->fluid->o_U,
                            o_qg,
                            o_ul);
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg);
  opSEM::strongGradVec(mesh, offset, o_ul, o_gradUl);
  updateVirtualMassHistoryKernel(mesh->Nlocal,
                                 offset,
                                 1.0 / nrs->dt[0],
                                 o_ug,
                                 o_ul,
                                 o_ulPrevious,
                                 o_ugPrevious,
                                 o_gradUl,
                                 o_gradUg,
                                 o_virtualMassRelativeAcceleration);
}

inline void evaluateMixtureForce()
{
  const dlong offset = nrs->fieldOffset;
  auto mesh = nrs->meshV;
  for (int i = 0; i < 3; ++i) {
    auto row = o_driftStress.slice(3 * i * offset, 3 * offset);
    auto divRow = o_divDriftStress.slice(i * offset, offset);
    opSEM::strongDivergence(mesh, offset, row, divRow);
  }
  buildMixtureForceKernel(mesh->Nlocal,
                          offset,
                          p.alphaFloor,
                          p.gravity[0],
                          p.gravity[1],
                          p.gravity[2],
                          nrs->scalar->o_solution("alpha"),
                          o_rhoM,
                          o_divDriftStress,
                          o_mixtureForce);
}

inline void subtractScalarDiffusion()
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;

  // Apply the same weak SEM stiffness used by the native scalar Helmholtz
  // operator. After gather-scatter, M^{-1} K s is a nodal source; sumMakef
  // multiplies it by M again. Thus the extrapolated explicit contribution is
  // +K s^lag, opposing the +K s^{n+1} implicit diffusion term exactly in the
  // unconstrained scalar equations (up to temporal lag/extrapolation).
  if (p.subtractAlphaDiffusion != 0.0) {
    auto diffusion = nrs->scalar->o_diffusionCoeff("alpha");
    launchKernel("core-weakLaplacianHex3D",
                 nrs->meshV->Nelements,
                 1,
                 o_scalarFieldOffsetScan,
                 nrs->meshV->o_ggeo,
                 nrs->meshV->o_D,
                 diffusion,
                 nrs->scalar->o_solution("alpha"),
                 o_alphaDiffusionDivergence);
    oogs::startFinish(o_alphaDiffusionDivergence,
                      1,
                      0,
                      ogsDfloat,
                      ogsAdd,
                      nrs->meshV->oogs);
    platform->linAlg->axmy(
        Nlocal, 1.0, nrs->meshV->o_invLMM, o_alphaDiffusionDivergence);
    platform->linAlg->axpby(Nlocal,
                            p.subtractAlphaDiffusion,
                            o_alphaDiffusionDivergence,
                            1.0,
                            o_alphaSource);
  }

  const char *qgNames[3] = {"qgx", "qgy", "qgz"};
  for (int i = 0; i < 3; ++i) {
    if (p.subtractQgDiffusion[i] == 0.0) {
      continue;
    }
    auto diffusion = nrs->scalar->o_diffusionCoeff(qgNames[i]);
    auto divergence = o_qgDiffusionDivergence.slice(i * offset, offset);
    launchKernel("core-weakLaplacianHex3D",
                 nrs->meshV->Nelements,
                 1,
                 o_scalarFieldOffsetScan,
                 nrs->meshV->o_ggeo,
                 nrs->meshV->o_D,
                 diffusion,
                 nrs->scalar->o_solution(qgNames[i]),
                 divergence);
    oogs::startFinish(
        divergence, 1, 0, ogsDfloat, ogsAdd, nrs->meshV->oogs);
    platform->linAlg->axmy(
        Nlocal, 1.0, nrs->meshV->o_invLMM, divergence);
    platform->linAlg->axpby(Nlocal,
                            p.subtractQgDiffusion[i],
                            divergence,
                            1.0,
                            o_qgSource,
                            0,
                            i * offset);
  }
}

inline void addExplicitSources(double)
{
  evaluatePointwiseTerms();
  subtractScalarDiffusion();
  evaluateMixtureForce();
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;

  // Copy only entries written by the pointwise kernels. Avoid whole-view
  // copies because scalar and fluid fields may have different padded extents.
  nrs->scalar->o_explicitTerms("alpha").copyFrom(o_alphaSource, Nlocal);
  // Keep the user-assembled part so updateProperties() can isolate any HPFRT
  // or GJP contribution subsequently added by scalar_t::makeExplicit().
  o_alphaExplicitBase.copyFrom(o_alphaSource, Nlocal);
  nrs->scalar->o_explicitTerms("qgx").copyFrom(o_qgSource, Nlocal, 0, 0 * offset);
  nrs->scalar->o_explicitTerms("qgy").copyFrom(o_qgSource, Nlocal, 0, 1 * offset);
  nrs->scalar->o_explicitTerms("qgz").copyFrom(o_qgSource, Nlocal, 0, 2 * offset);

  auto fluidTerms = nrs->fluid->o_explicitTerms();
  for (int i = 0; i < 3; ++i) {
    fluidTerms.copyFrom(o_mixtureForce, Nlocal, i * offset, i * offset);
  }
}

inline void buildDivergenceFromAlphaRhs()
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;

  // The native scalar equation is
  //   d(alpha)/dt + um.grad(alpha)
  //     = S_alpha + div(D_alpha grad(alpha))
  // for transportCoeff=1. Construct mixture divergence from this RHS rather
  // than differentiating alpha in time. The numerical diffusion configured in
  // [SCALAR ALPHA] is therefore included in mixture-density continuity.
  o_alphaDiffusionFlux.copyFrom(o_gradAlpha, 3 * offset);
  auto diffusion = nrs->scalar->o_diffusionCoeff("alpha");
  platform->linAlg->axmyVector(
      Nlocal, offset, 0, 1.0, diffusion, o_alphaDiffusionFlux);
  opSEM::strongDivergence(nrs->meshV,
                          offset,
                          o_alphaDiffusionFlux,
                          o_alphaDiffusionDivergence);

  // scalar_t::makeExplicit() adds HPFRT/GJP to the current explicit-term
  // buffer after userSource(). Recover that numerical RHS contribution so the
  // same alpha regularization is represented in mixture-density continuity.
  o_alphaRegularizationSource.copyFrom(
      nrs->scalar->o_explicitTerms("alpha"), Nlocal);
  platform->linAlg->axpby(Nlocal,
                          -1.0,
                          o_alphaExplicitBase,
                          1.0,
                          o_alphaRegularizationSource);

  buildDivergenceFromAlphaRhsKernel(Nlocal,
                                    p.rhoLiquid,
                                    p.rhoGas,
                                    nrs->scalar->o_solution("alpha"),
                                    o_alphaSource,
                                    o_alphaDiffusionDivergence,
                                    o_alphaRegularizationSource,
                                    o_divSource);
}

inline void updateProperties(double)
{
  // Called after all four scalars advance: refresh density, viscosity, and the
  // mixture divergence implied by the alpha-equation RHS.
  postProcessGasFlux();
  evaluatePointwiseTerms();
  buildDivergenceFromAlphaRhs();
  filterDivergence();
  nrs->fluid->o_prop.slice(0 * nrs->fieldOffset, nrs->fieldOffset).copyFrom(o_muM);
  nrs->fluid->o_prop.slice(1 * nrs->fieldOffset, nrs->fieldOffset).copyFrom(o_rhoM);
}

inline occa::memory implicitGasDrag(double, int scalarIndex)
{
  // Scalar ordering is ALPHA, QGX, QGY, QGZ.
  if (p.dragEnabled != 0.0 && scalarIndex >= 1 && scalarIndex <= 3) {
    return o_dragLambda;
  }
  return o_NULL;
}

inline void updateDivergence(double)
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  if (p.divergenceExtrapolationEnabled == 0.0 || divergenceHistoryCount == 0) {
    nrs->fluid->o_div.copyFrom(o_divSource, Nlocal);
  } else if (divergenceHistoryCount == 1) {
    // EXT1 startup: q^{n+1} = q^n.
    nrs->fluid->o_div.copyFrom(o_divPrevious, Nlocal);
  } else {
    // EXT2 from completed pressure steps: q^{n+1} = 2 q^n - q^{n-1}.
    o_divExtrapolated.copyFrom(o_divOlder, Nlocal);
    platform->linAlg->axpby(
        Nlocal, 2.0, o_divPrevious, -1.0, o_divExtrapolated);
    nrs->fluid->o_div.copyFrom(o_divExtrapolated, Nlocal);
  }

  const int tstep = platform->app->tstep;
  if (p.divergenceRampEnabled != 0.0 && tstep < p.divergenceRampSteps) {
    const dfloat phase = tstep > 0
        ? static_cast<dfloat>(tstep) / p.divergenceRampSteps
        : 0.0;
    const dfloat ramp = 0.5 * (1.0 - std::cos(3.14159265358979323846 * phase));
    platform->linAlg->scale(Nlocal, ramp, nrs->fluid->o_div);
  }
}

inline void updateDivergenceHistory()
{
  if (p.divergenceExtrapolationEnabled == 0.0) {
    return;
  }

  const dlong Nlocal = nrs->meshV->Nlocal;
  if (divergenceHistoryCount > 0) {
    o_divOlder.copyFrom(o_divPrevious, Nlocal);
  }
  o_divPrevious.copyFrom(o_divSource, Nlocal);
  divergenceHistoryCount = std::min(divergenceHistoryCount + 1, 2);
}

inline dfloat computeGasCfl()
{
  auto mesh = nrs->meshV;
  launchKernel("nrs-cflHex3D",
               mesh->Nelements,
               nrs->dt[0],
               mesh->o_vgeo,
               o_inverseGllSpacing,
               nrs->fieldOffset,
               o_ug,
               nrs->geom ? nrs->geom->o_U : o_NULL,
               o_gasCfl);
  return platform->linAlg->max(
      mesh->Nelements, o_gasCfl, platform->comm.mpiComm());
}

inline void printStabilityMonitors(double time, int tstep)
{
  if (p.stabilityMonitorEnabled == 0.0
      || p.stabilityMonitorInterval < 1
      || tstep % p.stabilityMonitorInterval != 0) {
    return;
  }

  // Refresh gas-dependent fields with the completed-step scalar, mixture
  // velocity, and pressure solutions. updateDivergenceHistory() must be called
  // first because evaluatePointwiseTerms() also refreshes its scratch source.
  evaluatePointwiseTerms();

  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  const MPI_Comm comm = platform->comm.mpiComm();

  const dfloat maxDiv =
      platform->linAlg->amax(Nlocal, nrs->fluid->o_div, comm);

  platform->linAlg->entrywiseMag(Nlocal, 3, offset, o_gradP, o_monitorMagnitude);
  buildGasPressureAccelerationMonitorKernel(Nlocal,
                                            p.rhoGas,
                                            p.gasMomentumCutoff,
                                            nrs->scalar->o_solution("alpha"),
                                            o_monitorMagnitude,
                                            o_gasPressureAccelerationActive,
                                            o_gasPressureAccelerationInactive);
  const dfloat maxGasPressureAccelerationActive = platform->linAlg->max(
      Nlocal, o_gasPressureAccelerationActive, comm);
  const dfloat maxGasPressureAccelerationInactive = platform->linAlg->max(
      Nlocal, o_gasPressureAccelerationInactive, comm);

  auto alpha = nrs->scalar->o_solution("alpha");
  const dfloat minAlpha = platform->linAlg->min(Nlocal, alpha, comm);
  const dfloat maxAlpha = platform->linAlg->max(Nlocal, alpha, comm);
  const dfloat meanAlpha = platform->linAlg->innerProd(
      Nlocal, nrs->meshV->o_LMM, alpha, comm) / nrs->meshV->volume;

  platform->linAlg->entrywiseMag(Nlocal, 3, offset, o_qg, o_monitorMagnitude);
  const dfloat maxQg = platform->linAlg->max(Nlocal, o_monitorMagnitude, comm);

  buildGasFluxConsistencyMonitorKernel(Nlocal,
                                       offset,
                                       alpha,
                                       o_qg,
                                       o_ug,
                                       o_monitorMagnitude);
  const dfloat maxQgConsistencyError =
      platform->linAlg->max(Nlocal, o_monitorMagnitude, comm);

  platform->linAlg->entrywiseMag(Nlocal, 3, offset, o_ug, o_monitorMagnitude);
  const dfloat maxUg = platform->linAlg->max(Nlocal, o_monitorMagnitude, comm);
  const dfloat gasCfl = computeGasCfl();

  platform->linAlg->entrywiseMag(
      Nlocal, 9, offset, o_driftStress, o_monitorMagnitude);
  const dfloat maxDriftStress =
      platform->linAlg->max(Nlocal, o_monitorMagnitude, comm);

  const dfloat maxDragLambda =
      platform->linAlg->max(Nlocal, o_dragLambda, comm);
  const dfloat maxDragStep = maxDragLambda * nrs->dt[0];

  // Boundary-integrated checks for the alpha inlet.  These distinguish an
  // incorrectly applied Dirichlet value from a layer that develops in the
  // first interior element.  qgFlux uses the outward mesh normal, so a gas
  // inflow through the z=0 face is normally negative.
  o_surfaceScalar.copyFrom(alpha, Nlocal);
  const dfloat inletArea = nrs->meshV->surfaceAreaMultiplyIntegrate(
      o_inletBoundaryID, o_surfaceOne);
  const dfloat inletAlphaIntegral = nrs->meshV->surfaceAreaMultiplyIntegrate(
      o_inletBoundaryID, o_surfaceScalar);
  const dfloat inletAlphaAverage =
      inletArea > 0.0 ? inletAlphaIntegral / inletArea : 0.0;
  const dfloat inletQgFlux = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_inletBoundaryID, o_qg);
  const dfloat prescribedQgFluxMagnitude =
      p.alphaInlet * std::abs(p.ugInlet) * inletArea;

  if (platform->comm.mpiRank() == 0) {
    printf("bubbleColumn stability step=%d time=%.8e max|divTarget|=%.8e "
           "min(alpha)=%.8e max(alpha)=%.8e mean(alpha)=%.8e "
           "max|qg|=%.8e max|qg-alpha*ug|=%.8e "
           "maxActive|gradP|/rhoG=%.8e maxInactive|gradP|/rhoG=%.8e "
           "max|ug|=%.8e CFL(ug)=%.8e max|tauDrift|=%.8e "
           "max(lambdaD*dt)=%.8e "
           "inletArea=%.8e inletMean(alpha)=%.8e "
           "inletIntegral(qg.n)=%.8e prescribed|inletFlux|=%.8e\n",
           tstep,
           time,
           maxDiv,
           minAlpha,
           maxAlpha,
           meanAlpha,
           maxQg,
           maxQgConsistencyError,
           maxGasPressureAccelerationActive,
           maxGasPressureAccelerationInactive,
           maxUg,
           gasCfl,
           maxDriftStress,
           maxDragStep,
           inletArea,
           inletAlphaAverage,
           inletQgFlux,
           prescribedQgFluxMagnitude);
  }
}
} // namespace bubbleColumn
