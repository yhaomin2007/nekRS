#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

#include "opSEM.hpp"

namespace bubbleColumn2
{
struct Parameters {
  dfloat rhoLiquid;
  dfloat rhoGas;
  dfloat muLiquid;
  dfloat muGas;
  dfloat alphaInitial;
  dfloat alphaInlet;
  dfloat ugInlet;
  dfloat ugInitial = 0.0;
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
  dfloat volumeVelocityClipEnabled;
  dfloat volumeVelocityMaximum;
  dfloat gasMomentumCutoff;
  dfloat gasMomentumFullyActive;
  dfloat gasPressureEnabled;
  dfloat dragEnabled;
  dfloat mixtureDragRampEnabled = 0.0;
  int mixtureDragRampStartStep = 0;
  int mixtureDragRampSteps = 10000;
  dfloat mixtureDragEnabled = 1.0;
  dfloat dragAlphaCutoff = 0.0;
  dfloat dragSlipLimitEnabled = 0.0;
  dfloat dragSlipMaximum = 1.0;
  dfloat bubbleDiameter;
  dfloat driftStressEnabled;
  dfloat mixtureViscousCorrectionEnabled = 0.0;
  dfloat virtualMassEnabled;
  dfloat virtualMassCoefficient;
  dfloat stabilityMonitorEnabled;
  int stabilityMonitorInterval;
  int validationOutputInterval;
};

static Parameters p;
static deviceMemory<dfloat> o_nativeTransportAdvection;
static deviceMemory<dfloat> o_gradUv;
static deviceMemory<dfloat> o_exactMixtureStress, o_baseNativeStress;
static deviceMemory<dfloat> o_divExactMixtureStress, o_divBaseNativeStress;
static occa::kernel correctAdvectionCancellationKernel;
static occa::kernel buildMixtureStressCorrectionKernel, addMixtureStressCorrectionKernel;
static deviceMemory<dfloat> o_ug;
static deviceMemory<dfloat> o_ul;
static deviceMemory<dfloat> o_ulPrevious;
static deviceMemory<dfloat> o_ugPrevious;
static deviceMemory<dfloat> o_gradUl;
static deviceMemory<dfloat> o_virtualMassRelativeAcceleration;
static deviceMemory<dfloat> o_gradAlpha;
static deviceMemory<dfloat> o_qg;
static deviceMemory<dfloat> o_gradQgAdvection;
static deviceMemory<dfloat> o_divQg;
static deviceMemory<dfloat> o_qgAdvectionFlux;
static deviceMemory<dfloat> o_divQgAdvectionFlux;
static deviceMemory<dfloat> o_gradUg;
static deviceMemory<dfloat> o_gradP;
static deviceMemory<dfloat> o_rhoM;
static deviceMemory<dfloat> o_rhoPressure;
static deviceMemory<dfloat> o_muM;
static deviceMemory<dfloat> o_alphaDiffusionDivergence;
static deviceMemory<dfloat> o_qgDiffusionDivergence;
static deviceMemory<dfloat> o_scalarDiffusionBase;
static deviceMemory<dfloat> o_outletRampFactor;
static deviceMemory<dfloat> o_driftStress;
static deviceMemory<dfloat> o_divDriftStress;
static deviceMemory<dfloat> o_gasStress;
static deviceMemory<dfloat> o_divGasStress;
static deviceMemory<dfloat> o_alphaSource;
static deviceMemory<dfloat> o_qgSource;
static deviceMemory<dfloat> o_dragLambda;
static deviceMemory<dfloat> o_gasDragSource, o_mixtureDragSource;
static deviceMemory<dfloat> o_previousGasDragSource, o_previousMixtureDragSource;
static deviceMemory<dfloat> o_mixtureInterphaseAcceleration;
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
static deviceMemory<int> o_outletBoundaryID;
static deviceMemory<int> o_allBoundaryIDs;
static deviceMemory<dfloat> o_validationGasDiffusiveFlux;
static deviceMemory<dfloat> o_validationMassFlux;
static deviceMemory<dfloat> o_validationDivQg;
static deviceMemory<dfloat> o_alphaUserAdvectionDiagnostic;
static deviceMemory<dfloat> o_alphaNativeAdvectionDiagnostic;
static int alphaAdvectionDiagnosticStep = -1;
static deviceMemory<dfloat> o_qgBeforeMask;
static deviceMemory<dfloat> o_qgMaskDelta;
static std::ofstream validationFile;
static std::vector<dfloat> gasVolumeHistory;
static std::vector<dfloat> totalMassHistory;
static dfloat validationInitialGasVolume = 0.0;
static dfloat validationInitialTotalMass = 0.0;
static dfloat validationPreviousTime = 0.0;
static dfloat validationPreviousGasBoundaryFlux = 0.0;
static dfloat validationPreviousMassBoundaryFlux = 0.0;
static dfloat cumulativeGasBoundaryIntegral = 0.0;
static dfloat cumulativeMassBoundaryIntegral = 0.0;
static dfloat alphaClipDeltaVolume = 0.0;
static dfloat cumulativeAlphaClipDeltaVolume = 0.0;
static dfloat qgMaskDeltaIntegral[3] = {0.0, 0.0, 0.0};
static dfloat qgMaskDeltaMagnitudeIntegral = 0.0;
static dfloat cumulativeQgMaskDeltaIntegral[3] = {0.0, 0.0, 0.0};
static dfloat cumulativeQgMaskDeltaMagnitudeIntegral = 0.0;
static bool validationInitialized = false;
static occa::kernel vectorMagnitudeKernel;
static occa::kernel reconstructGasVelocityKernel;
static occa::kernel postProcessGasFluxKernel;
static occa::kernel clipAlphaKernel;
static occa::kernel clipVectorMagnitudeKernel;
static occa::kernel initializeUniformKernel;
static occa::kernel buildLiquidVelocityKernel;
static occa::kernel updateVirtualMassHistoryKernel;
static occa::kernel buildEquationTermsKernel;
static occa::kernel buildMixtureForceKernel;
static occa::kernel buildGasPressureAccelerationMonitorKernel;
static occa::kernel buildGasFluxConsistencyMonitorKernel;
static occa::kernel buildOutletRampFactorKernel;

inline void registerKernels(deviceKernelProperties &kernelInfo)
{
  const std::string request = "bubbleColumn2::equations";
  // This is standalone OKL source.  The .okl suffix is required so OCCA
  // translates @kernel/@globalPtr before invoking the HIP compiler.
  const std::string fileName = "bubbleColumn2Equations.okl";
  if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    platform->kernelRequests.add(request, fileName, kernelInfo);
  } else {
    vectorMagnitudeKernel = platform->kernelRequests.load(request, "vectorMagnitude");
    correctAdvectionCancellationKernel = platform->kernelRequests.load(request, "correctAdvectionCancellation");
    buildMixtureStressCorrectionKernel = platform->kernelRequests.load(request, "buildMixtureStressCorrection");
    addMixtureStressCorrectionKernel = platform->kernelRequests.load(request, "addMixtureStressCorrection");
    reconstructGasVelocityKernel =
        platform->kernelRequests.load(request, "reconstructGasVelocity");
    postProcessGasFluxKernel =
        platform->kernelRequests.load(request, "postProcessGasFlux");
    clipAlphaKernel = platform->kernelRequests.load(request, "clipAlpha");
    clipVectorMagnitudeKernel =
        platform->kernelRequests.load(request, "clipVectorMagnitude");
    initializeUniformKernel = platform->kernelRequests.load(request, "initializeUniform");
    buildLiquidVelocityKernel = platform->kernelRequests.load(request, "buildLiquidVelocity");
    updateVirtualMassHistoryKernel =
        platform->kernelRequests.load(request, "updateVirtualMassHistory");
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
  nekrsCheck(p.volumeVelocityClipEnabled != 0.0
                 && p.volumeVelocityMaximum <= 0.0,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "volumeVelocityMaximum must be positive when clipping is enabled, "
             "but is %g\n",
             p.volumeVelocityMaximum);
  nekrsCheck(p.alphaClippingEnabled != 0.0
                 && (p.alphaMinimum < 0.0 || p.alphaMaximum > 1.0
                     || p.alphaMaximum <= p.alphaMinimum),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "alpha clipping bounds must satisfy 0 <= minimum < maximum <= 1 "
             "(minimum=%g, maximum=%g)\n",
             p.alphaMinimum,
             p.alphaMaximum);
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
  nekrsCheck(p.validationOutputInterval < 1,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "validationOutputInterval must be positive, but is %d\n",
             p.validationOutputInterval);
  nekrsCheck(nrs->scalar->Nsubsteps != 0, platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "bubbleColumn2 conservative source replacement requires scalar subcycling off.\n");
  o_nativeTransportAdvection.resize(nrs->scalar->fieldOffsetSum);
  o_gradUv.resize(9 * offset);
  o_exactMixtureStress.resize(9 * offset);
  o_baseNativeStress.resize(9 * offset);
  o_divExactMixtureStress.resize(3 * offset);
  o_divBaseNativeStress.resize(3 * offset);
  o_ug.resize(3 * offset);
  o_ul.resize(3 * offset);
  o_ulPrevious.resize(3 * offset);
  o_ugPrevious.resize(3 * offset);
  o_gradUl.resize(9 * offset);
  o_virtualMassRelativeAcceleration.resize(3 * offset);
  o_gradAlpha.resize(3 * offset);
  o_qg.resize(3 * offset);
  o_gradQgAdvection.resize(9 * offset);
  o_divQg.resize(offset);
  o_qgAdvectionFlux.resize(9 * offset);
  o_divQgAdvectionFlux.resize(3 * offset);
  o_gradUg.resize(9 * offset);
  o_gradP.resize(3 * offset);
  o_rhoM.resize(offset);
  o_rhoPressure.resize(offset);
  o_muM.resize(offset);
  o_alphaDiffusionDivergence.resize(offset);
  o_qgDiffusionDivergence.resize(3 * offset);
  o_scalarDiffusionBase.resize(4 * offset);
  o_outletRampFactor.resize(offset);
  o_driftStress.resize(9 * offset);
  o_divDriftStress.resize(3 * offset);
  o_gasStress.resize(9 * offset);
  o_divGasStress.resize(3 * offset);
  o_alphaSource.resize(offset);
  o_qgSource.resize(3 * offset);
  o_dragLambda.resize(offset);
  o_gasDragSource.resize(3 * offset);
  o_mixtureDragSource.resize(3 * offset);
  o_previousGasDragSource.resize(3 * offset);
  o_previousMixtureDragSource.resize(3 * offset);
  platform->linAlg->fill(3 * offset, 0.0, o_previousGasDragSource);
  platform->linAlg->fill(3 * offset, 0.0, o_previousMixtureDragSource);
  o_mixtureInterphaseAcceleration.resize(3 * offset);
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
  o_outletBoundaryID.resize(1);
  o_outletBoundaryID.copyFrom(std::vector<int>{2});
  o_allBoundaryIDs.resize(3);
  o_allBoundaryIDs.copyFrom(std::vector<int>{1, 2, 3});
  o_validationGasDiffusiveFlux.resize(3 * offset);
  o_validationMassFlux.resize(3 * offset);
  o_validationDivQg.resize(offset);
  o_alphaUserAdvectionDiagnostic.resize(offset);
  o_alphaNativeAdvectionDiagnostic.resize(nrs->scalar->fieldOffsetSum);
  o_qgBeforeMask.resize(3 * offset);
  o_qgMaskDelta.resize(3 * offset);
  platform->linAlg->fill(nrs->meshV->Nlocal, 1.0, o_surfaceOne);

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
                            p.alphaFloor,
                            alpha,
                            nrs->fluid->o_U,
                            o_qg,
                            o_ul);
  // Use normalized gather-scatter assembly for every user-side SEM gradient
  // and divergence. This keeps the reconstructed cancellation and conservative
  // flux terms single-valued at shared CG nodes.
  opSEM::strongGrad(mesh, offset, alpha, o_gradAlpha);
  opSEM::strongGradVec(mesh, offset, o_qg, o_gradQgAdvection);
  opSEM::strongDivergence(mesh, offset, o_qg, o_divQg);
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg);
  opSEM::strongGradVec(mesh, offset, o_ul, o_gradUl);
  opSEM::strongGradVec(mesh, offset, nrs->fluid->o_U, o_gradUv);
  opSEM::strongGrad(mesh, offset, nrs->fluid->o_P, o_gradP);

  // Form F_ij = q_g,i * u_g,j and take an assembled strong divergence of
  // each tensor row for the conservative QG transport correction.
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
    opSEM::strongDivergence(mesh, offset, flux, divergence);
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
                           p.mixtureDragEnabled,
                           p.dragAlphaCutoff,
                           p.dragSlipLimitEnabled,
                           p.dragSlipMaximum,
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
                           o_gradAlpha,
                           o_gradQgAdvection,
                           o_divQg,
                           o_divQgAdvectionFlux,
                           o_gradUg,
                           o_virtualMassRelativeAcceleration,
                           o_gradP,
                           o_rhoM,
                           o_rhoPressure,
                           o_muM,
                           o_driftStress,
                           o_gasStress,
                           o_alphaSource,
                           o_qgSource,
                           o_dragLambda,
                           o_mixtureInterphaseAcceleration,
                           o_gasDragSource, o_mixtureDragSource);

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

inline void evaluateNativeScalarAdvection(int scalarIndex,
                                          deviceMemory<dfloat> &o_advection,
                                          bool averageSharedNodes)
{
  auto mesh = nrs->meshV;
  const int alphaIndex = scalarIndex;

  // Reuse the exact NekRS scalar-advection operator and the contravariant
  // predicted velocity already prepared for the current timestep.
  if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
    launchKernel("core-strongAdvectionCubatureVolumeScalarHex3D",
                 mesh->Nelements,
                 1,
                 0,
                 0,
                 mesh->o_vgeo,
                 mesh->o_cubDiffInterpT,
                 mesh->o_cubInterpT,
                 mesh->o_cubProjectT,
                 nrs->scalar->o_compute + alphaIndex,
                 nrs->scalar->o_fieldOffsetScan + alphaIndex,
                 nrs->scalar->vFieldOffset,
                 nrs->scalar->vCubatureOffset,
                 nrs->scalar->o_S,
                 nrs->scalar->o_relUrst,
                 nrs->scalar->o_rho,
                 o_advection);
  } else {
    launchKernel("core-strongAdvectionVolumeScalarHex3D",
                 mesh->Nelements,
                 1,
                 0,
                 mesh->o_vgeo,
                 mesh->o_D,
                 nrs->scalar->o_compute + alphaIndex,
                 nrs->scalar->o_fieldOffsetScan + alphaIndex,
                 nrs->scalar->vFieldOffset,
                 nrs->scalar->o_S,
                 nrs->scalar->o_relUrst,
                 nrs->scalar->o_rho,
                 o_advection);
  }

  if (averageSharedNodes) {
    const dlong alphaOffset = nrs->scalar->fieldOffsetScan[alphaIndex];
    auto o_alphaAdvection =
        o_advection.slice(alphaOffset, nrs->fieldOffset);
    oogs::startFinish(o_alphaAdvection,
                      1,
                      0,
                      ogsDfloat,
                      ogsAdd,
                      mesh->oogs);
    platform->linAlg->axmy(
        mesh->Nlocal, 1.0, mesh->o_invAJw, o_alphaAdvection);
  }
}

inline void evaluateNativeAlphaAdvection(deviceMemory<dfloat> &o_advection,
                                         bool averageSharedNodes)
{
  evaluateNativeScalarAdvection(nrs->scalar->nameToIndex.at("alpha"),
                                o_advection, averageSharedNodes);
}

inline void correctConservativeAdvection()
{
  const dlong offset = nrs->fieldOffset;
  const char *names[4] = {"alpha", "qgx", "qgy", "qgz"};
  for (int i = 0; i < 4; ++i) {
    const int index = nrs->scalar->nameToIndex.at(names[i]);
    evaluateNativeScalarAdvection(index, o_nativeTransportAdvection, false);
    auto native = o_nativeTransportAdvection.slice(nrs->scalar->fieldOffsetScan[index], offset);
    auto grad = i == 0 ? o_gradAlpha : o_gradQgAdvection.slice(3 * (i - 1) * offset, 3 * offset);
    auto source = i == 0 ? o_alphaSource : o_qgSource.slice((i - 1) * offset, offset);
    correctAdvectionCancellationKernel(nrs->meshV->Nlocal, offset,
                                       nrs->fluid->o_U, grad, native, source);
  }
}

inline void captureAlphaAdvectionDiagnostics()
{
  if (nrs->scalar->Nsubsteps != 0
      || nrs->tstep <= 0
      || nrs->tstep % p.validationOutputInterval != 0) {
    return;
  }

  auto mesh = nrs->meshV;
  const dlong Nlocal = mesh->Nlocal;

  // alphaSource = A_m,user(alpha) - D(q_g). Recover and retain the user-side
  // mixture-advection term before later callbacks refresh the scratch fields.
  o_alphaUserAdvectionDiagnostic.copyFrom(o_alphaSource, Nlocal);
  platform->linAlg->axpby(
      Nlocal, 1.0, o_divQg, 1.0, o_alphaUserAdvectionDiagnostic);

  // Re-evaluate exactly the same native scalar-advection kernel that NekRS
  // calls immediately after userSource(). This isolates the spatial
  // cancellation defect A_m,native(alpha) - A_m,user(alpha).
  evaluateNativeAlphaAdvection(o_alphaNativeAdvectionDiagnostic, false);
  alphaAdvectionDiagnosticStep = nrs->tstep;
}

inline void initializeHistory()
{
  const dlong offset = nrs->fieldOffset;
  platform->linAlg->fill(3 * offset, 0.0, o_virtualMassRelativeAcceleration);
  evaluatePointwiseTerms();
  o_ulPrevious.copyFrom(o_ul, 3 * offset);
  o_ugPrevious.copyFrom(o_ug, 3 * offset);
  platform->linAlg->fill(3 * offset, 0.0, o_virtualMassRelativeAcceleration);
}

inline void postProcessGasFlux()
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  if (p.smoothGasVelocityMaskEnabled == 0.0 && p.gasVelocityClipEnabled == 0.0) {
    for (int i = 0; i < 3; ++i) {
      qgMaskDeltaIntegral[i] = 0.0;
    }
    qgMaskDeltaMagnitudeIntegral = 0.0;
    return;
  }

  o_qgBeforeMask.copyFrom(nrs->scalar->o_solution("qgx"), Nlocal, 0 * offset, 0);
  o_qgBeforeMask.copyFrom(nrs->scalar->o_solution("qgy"), Nlocal, 1 * offset, 0);
  o_qgBeforeMask.copyFrom(nrs->scalar->o_solution("qgz"), Nlocal, 2 * offset, 0);

  postProcessGasFluxKernel(
      Nlocal,
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

  const char *qgNames[3] = {"qgx", "qgy", "qgz"};
  const MPI_Comm comm = platform->comm.mpiComm();
  for (int i = 0; i < 3; ++i) {
    auto delta = o_qgMaskDelta.slice(i * offset, offset);
    delta.copyFrom(nrs->scalar->o_solution(qgNames[i]), Nlocal);
    platform->linAlg->axpby(
        Nlocal, -1.0, o_qgBeforeMask, 1.0, delta, i * offset, 0);
    qgMaskDeltaIntegral[i] = platform->linAlg->innerProd(
        Nlocal, nrs->meshV->o_LMM, delta, comm);
    cumulativeQgMaskDeltaIntegral[i] += qgMaskDeltaIntegral[i];
  }
  vectorMagnitudeKernel(
      Nlocal, 3, offset, o_qgMaskDelta, o_monitorMagnitude);
  qgMaskDeltaMagnitudeIntegral = platform->linAlg->innerProd(
      Nlocal, nrs->meshV->o_LMM, o_monitorMagnitude, comm);
  cumulativeQgMaskDeltaMagnitudeIntegral += qgMaskDeltaMagnitudeIntegral;
}

inline void clipAlpha(double, int)
{
  alphaClipDeltaVolume = 0.0;
  if (p.alphaClippingEnabled == 0.0) {
    return;
  }

  const dlong Nlocal = nrs->meshV->Nlocal;
  const MPI_Comm comm = platform->comm.mpiComm();
  auto alpha = nrs->scalar->o_solution("alpha");
  const dfloat volumeBefore = platform->linAlg->innerProd(
      Nlocal, nrs->meshV->o_LMM, alpha, comm);

  clipAlphaKernel(Nlocal,
                  p.alphaMinimum,
                  p.alphaMaximum,
                  alpha);
  const dfloat volumeAfter = platform->linAlg->innerProd(
      Nlocal, nrs->meshV->o_LMM, alpha, comm);
  alphaClipDeltaVolume = volumeAfter - volumeBefore;
  cumulativeAlphaClipDeltaVolume += alphaClipDeltaVolume;
}

inline void clipVolumeVelocity()
{
  if (p.volumeVelocityClipEnabled == 0.0) {
    return;
  }

  clipVectorMagnitudeKernel(nrs->meshV->Nlocal,
                            nrs->fieldOffset,
                            p.volumeVelocityMaximum,
                            nrs->fluid->o_U);
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
  if (p.driftStressEnabled != 0.0) {
    for (int i = 0; i < 3; ++i) {
      auto row = o_driftStress.slice(3 * i * offset, 3 * offset);
      auto divRow = o_divDriftStress.slice(i * offset, offset);
      opSEM::strongDivergence(mesh, offset, row, divRow);
    }
  }
  buildMixtureForceKernel(mesh->Nlocal,
                          offset,
                          p.alphaFloor,
                          p.driftStressEnabled,
                          p.gravity[0],
                          p.gravity[1],
                          p.gravity[2],
                          nrs->scalar->o_solution("alpha"),
                          o_divDriftStress,
                          o_mixtureInterphaseAcceleration,
                          o_mixtureForce);
  buildMixtureStressCorrectionKernel(mesh->Nlocal, offset,
      p.rhoLiquid, p.rhoGas, p.muLiquid, p.muGas,
      nrs->scalar->o_solution("alpha"), o_rhoPressure, o_gradUl, o_gradUg,
      o_gradUv, o_exactMixtureStress, o_baseNativeStress);
  for (int i = 0; i < 3; ++i) {
    auto row = o_exactMixtureStress.slice(3 * i * offset, 3 * offset);
    auto div = o_divExactMixtureStress.slice(i * offset, offset);
    opSEM::strongDivergence(mesh, offset, row, div);
    row = o_baseNativeStress.slice(3 * i * offset, 3 * offset);
    div = o_divBaseNativeStress.slice(i * offset, offset);
    opSEM::strongDivergence(mesh, offset, row, div);
  }
  addMixtureStressCorrectionKernel(mesh->Nlocal, offset,
      p.mixtureViscousCorrectionEnabled, o_rhoPressure,
      o_divExactMixtureStress, o_divBaseNativeStress, o_mixtureForce);
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

// Scale only mixture drag using the native timestep counter.
inline dfloat mixtureDragRampFactor(int step)
{
  if (p.mixtureDragRampEnabled == 0.0) return 1.0;
  return std::max(0.0, std::min(1.0,
      (static_cast<dfloat>(step) - p.mixtureDragRampStartStep) / p.mixtureDragRampSteps));
}

inline void rampMixtureDrag(int step)
{
  if (p.mixtureDragRampEnabled == 0.0) return;
  const dfloat factor = mixtureDragRampFactor(step);
  const dlong N = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  for (int i = 0; i < 3; ++i) {
    auto drag = o_mixtureDragSource.slice(i * offset, offset);
    auto force = o_mixtureForce.slice(i * offset, offset);
    platform->linAlg->axpby(N, factor - 1.0, drag, 1.0, force);
    platform->linAlg->axpby(N, 0.0, drag, factor, drag);
  }
}

// All EXT history slots must contain the SAME drag contribution for this step.
// Native makeForcing shifts the modified histories afterward, so every old
// slot contains o_previous* at the next source call. Non-drag terms are untouched.
inline void freezeDragHistory()
{
  if (nrs->tstep == 0) return;
  const dlong N = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  for (int stage = 1; stage < nrs->scalar->o_coeffEXT.size(); ++stage) {
    for (int i = 0; i < 3; ++i) {
      const dlong dst = stage * nrs->scalar->fieldOffsetSum
          + nrs->scalar->fieldOffsetScan[i + 1];
      platform->linAlg->axpby(N, -1.0, o_previousGasDragSource, 1.0,
          nrs->scalar->o_EXT, i * offset, dst);
      platform->linAlg->axpby(N, 1.0, o_gasDragSource, 1.0,
          nrs->scalar->o_EXT, i * offset, dst);
    }
  }
  for (int stage = 1; stage < nrs->fluid->o_coeffEXT.size(); ++stage) {
    for (int i = 0; i < 3; ++i) {
      const dlong dst = stage * nrs->fluid->fieldOffsetSum + i * offset;
      platform->linAlg->axpby(N, -1.0, o_previousMixtureDragSource, 1.0,
          nrs->fluid->o_EXT, i * offset, dst);
      platform->linAlg->axpby(N, 1.0, o_mixtureDragSource, 1.0,
          nrs->fluid->o_EXT, i * offset, dst);
    }
  }
  o_previousGasDragSource.copyFrom(o_gasDragSource);
  o_previousMixtureDragSource.copyFrom(o_mixtureDragSource);
}

inline void addExplicitSources(double)
{
  evaluatePointwiseTerms();
  correctConservativeAdvection();
  captureAlphaAdvectionDiagnostics();
  subtractScalarDiffusion();
  evaluateMixtureForce();
  rampMixtureDrag(nrs->tstep);
  if (p.mixtureDragRampEnabled != 0.0 && p.stabilityMonitorEnabled != 0.0
      && p.stabilityMonitorInterval > 0
      && nrs->tstep % p.stabilityMonitorInterval == 0
      && platform->comm.mpiRank() == 0)
    printf("bubbleColumn2 mixtureDragRamp step=%d factor=%.8e\n",
           nrs->tstep, mixtureDragRampFactor(nrs->tstep));
  freezeDragHistory();
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;

  // Copy only entries written by the pointwise kernels. Avoid whole-view
  // copies because scalar and fluid fields may have different padded extents.
  nrs->scalar->o_explicitTerms("alpha").copyFrom(o_alphaSource, Nlocal);
  nrs->scalar->o_explicitTerms("qgx").copyFrom(o_qgSource, Nlocal, 0, 0 * offset);
  nrs->scalar->o_explicitTerms("qgy").copyFrom(o_qgSource, Nlocal, 0, 1 * offset);
  nrs->scalar->o_explicitTerms("qgz").copyFrom(o_qgSource, Nlocal, 0, 2 * offset);

  auto fluidTerms = nrs->fluid->o_explicitTerms();
  for (int i = 0; i < 3; ++i) {
    fluidTerms.copyFrom(o_mixtureForce, Nlocal, i * offset, i * offset);
  }
}

inline void updateProperties(double)
{
  // Called after all four scalars advance. Refresh the effective viscosity and
  // pressure mobility; the projected volume velocity remains divergence free.
  postProcessGasFlux();
  evaluatePointwiseTerms();
  nrs->fluid->o_prop.slice(0 * nrs->fieldOffset, nrs->fieldOffset).copyFrom(o_muM);
  nrs->fluid->o_prop.slice(1 * nrs->fieldOffset, nrs->fieldOffset)
      .copyFrom(o_rhoPressure);
}

inline void enforceZeroDivergence(double)
{
  // evaluateDivergence() zero-fills fluid->o_div before this callback.
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

struct ValidationState {
  dfloat gasVolume;
  dfloat totalMass;
  dfloat gasAdvectiveInlet;
  dfloat gasAdvectiveOutlet;
  dfloat gasAdvectiveTotal;
  dfloat gasDiffusiveInlet;
  dfloat gasDiffusiveOutlet;
  dfloat gasDiffusiveTotal;
  dfloat alphaDivQgVolumeIntegral;
  dfloat alphaDivergenceTheoremDefect;
  dfloat alphaDivergenceTheoremRelative;
  dfloat alphaNativeMixtureAdvectionVolumeIntegral;
  dfloat alphaUserMixtureAdvectionVolumeIntegral;
  dfloat alphaAdvectionCancellationDefect;
  dfloat alphaAdvectionCancellationRelative;
  dfloat totalMassFluxInlet;
  dfloat totalMassFluxOutlet;
  dfloat totalMassFluxTotal;
};

inline ValidationState computeValidationState(int tstep)
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  const MPI_Comm comm = platform->comm.mpiComm();
  auto alpha = nrs->scalar->o_solution("alpha");

  ValidationState state{};
  state.gasVolume = platform->linAlg->innerProd(
      Nlocal, nrs->meshV->o_LMM, alpha, comm);
  state.totalMass = platform->linAlg->innerProd(
      Nlocal, nrs->meshV->o_LMM, o_rhoM, comm);

  o_qg.copyFrom(nrs->scalar->o_solution("qgx"), Nlocal, 0 * offset, 0);
  o_qg.copyFrom(nrs->scalar->o_solution("qgy"), Nlocal, 1 * offset, 0);
  o_qg.copyFrom(nrs->scalar->o_solution("qgz"), Nlocal, 2 * offset, 0);
  state.gasAdvectiveInlet = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_inletBoundaryID, o_qg);
  state.gasAdvectiveOutlet = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_outletBoundaryID, o_qg);
  state.gasAdvectiveTotal = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_allBoundaryIDs, o_qg);

  // Test the discrete divergence theorem for the exact assembled operator used
  // in the alpha source: integral_Omega D_avg(q_g) dV = integral_boundary q_g.n dA.
  // This extra SEM operation is needed only on rows written to the CSV.
  if (tstep % p.validationOutputInterval == 0) {
    opSEM::strongDivergence(nrs->meshV, offset, o_qg, o_validationDivQg);
    state.alphaDivQgVolumeIntegral = platform->linAlg->innerProd(
        Nlocal, nrs->meshV->o_LMM, o_validationDivQg, comm);
    state.alphaDivergenceTheoremDefect =
        state.alphaDivQgVolumeIntegral - state.gasAdvectiveTotal;
    dfloat divergenceScale = std::max(
        std::abs(state.alphaDivQgVolumeIntegral),
        std::abs(state.gasAdvectiveTotal));
    state.alphaDivergenceTheoremRelative =
        std::abs(state.alphaDivergenceTheoremDefect)
        / std::max(divergenceScale, 1.0e-30);
  } else {
    state.alphaDivQgVolumeIntegral = NAN;
    state.alphaDivergenceTheoremDefect = NAN;
    state.alphaDivergenceTheoremRelative = NAN;
  }

  // The native kernel and reconstructed user term were sampled together in
  // userSource(). A NaN explicitly marks rows for which no matching sample is
  // available instead of silently comparing fields from different steps.
  if (alphaAdvectionDiagnosticStep == tstep) {
    const int alphaIndex = nrs->scalar->nameToIndex.at("alpha");
    const dlong alphaOffset = nrs->scalar->fieldOffsetScan[alphaIndex];
    auto nativeAlphaAdvection =
        o_alphaNativeAdvectionDiagnostic.slice(alphaOffset, Nlocal);
    state.alphaNativeMixtureAdvectionVolumeIntegral =
        platform->linAlg->innerProd(
            Nlocal, nrs->meshV->o_LMM, nativeAlphaAdvection, comm);
    state.alphaUserMixtureAdvectionVolumeIntegral =
        platform->linAlg->innerProd(
            Nlocal,
            nrs->meshV->o_LMM,
            o_alphaUserAdvectionDiagnostic,
            comm);
    state.alphaAdvectionCancellationDefect =
        state.alphaNativeMixtureAdvectionVolumeIntegral
        - state.alphaUserMixtureAdvectionVolumeIntegral;
    dfloat cancellationScale = std::max(
        std::abs(state.alphaNativeMixtureAdvectionVolumeIntegral),
        std::abs(state.alphaUserMixtureAdvectionVolumeIntegral));
    state.alphaAdvectionCancellationRelative =
        std::abs(state.alphaAdvectionCancellationDefect)
        / std::max(cancellationScale, 1.0e-30);
  } else {
    state.alphaNativeMixtureAdvectionVolumeIntegral = NAN;
    state.alphaUserMixtureAdvectionVolumeIntegral = NAN;
    state.alphaAdvectionCancellationDefect = NAN;
    state.alphaAdvectionCancellationRelative = NAN;
  }

  // Build the diagnostic flux from the current completed-step alpha field.
  // Do not rely on o_gradAlpha, whose last refresh depends on the solver and
  // monitor callback ordering.
  opSEM::strongGrad(
      nrs->meshV, offset, alpha, o_validationGasDiffusiveFlux);
  auto alphaDiffusion = nrs->scalar->o_diffusionCoeff("alpha");
  platform->linAlg->axmyVector(Nlocal,
                               offset,
                               0,
                               -1.0,
                               alphaDiffusion,
                               o_validationGasDiffusiveFlux);
  state.gasDiffusiveInlet = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_inletBoundaryID, o_validationGasDiffusiveFlux);
  state.gasDiffusiveOutlet = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_outletBoundaryID, o_validationGasDiffusiveFlux);
  state.gasDiffusiveTotal = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_allBoundaryIDs, o_validationGasDiffusiveFlux);

  // rho_l*(1-alpha)*u_l + rho_g*alpha*u_g
  // = rho_l*u_v + (rho_g-rho_l)*q_g.
  for (int i = 0; i < 3; ++i) {
    auto massFlux = o_validationMassFlux.slice(i * offset, offset);
    massFlux.copyFrom(nrs->fluid->o_U, Nlocal, 0, i * offset);
    platform->linAlg->scale(Nlocal, p.rhoLiquid, massFlux);
    platform->linAlg->axpby(Nlocal,
                            p.rhoGas - p.rhoLiquid,
                            o_qg,
                            1.0,
                            massFlux,
                            i * offset,
                            0);
  }
  state.totalMassFluxInlet = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_inletBoundaryID, o_validationMassFlux);
  state.totalMassFluxOutlet = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_outletBoundaryID, o_validationMassFlux);
  state.totalMassFluxTotal = nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      offset, o_allBoundaryIDs, o_validationMassFlux);
  return state;
}

inline dfloat validationBdfDerivative(dfloat current,
                                      const std::vector<dfloat>& history)
{
  std::vector<dfloat> coeff(nrs->o_coeffBDF.size());
  nrs->o_coeffBDF.copyTo(coeff.data());
  for (int i = history.size(); i < coeff.size(); ++i) {
    if (std::abs(coeff[i]) > 1.0e-14) {
      return NAN;
    }
  }
  dfloat derivative = nrs->g0 * current;
  const int nHistory = std::min(history.size(), coeff.size());
  for (int i = 0; i < nHistory; ++i) {
    derivative -= coeff[i] * history[i];
  }
  return derivative / nrs->dt[0];
}

inline void writeValidationChecks(double time, int tstep)
{
  const ValidationState state = computeValidationState(tstep);
  const dfloat gasBoundaryFlux =
      state.gasAdvectiveTotal + state.gasDiffusiveTotal;
  const dfloat massBoundaryFlux = state.totalMassFluxTotal;

  if (!validationInitialized) {
    validationInitialGasVolume = state.gasVolume;
    validationInitialTotalMass = state.totalMass;
    validationPreviousTime = time;
    validationPreviousGasBoundaryFlux = gasBoundaryFlux;
    validationPreviousMassBoundaryFlux = massBoundaryFlux;
    // The initial inventories already include clipping and QG masking from
    // this completed step, so cumulative postprocessing corrections begin
    // after the same baseline state.
    cumulativeAlphaClipDeltaVolume = 0.0;
    for (int i = 0; i < 3; ++i) {
      cumulativeQgMaskDeltaIntegral[i] = 0.0;
    }
    cumulativeQgMaskDeltaMagnitudeIntegral = 0.0;
    gasVolumeHistory.insert(gasVolumeHistory.begin(), state.gasVolume);
    totalMassHistory.insert(totalMassHistory.begin(), state.totalMass);
    if (platform->comm.mpiRank() == 0) {
      validationFile.open("bubbleColumn2_conservation.csv", std::ios::out);
      validationFile
          << "step,time,dt,gas_volume,dgas_volume_dt,"
          << "total_alpha_time_derivative,"
          << "gas_advective_flux_inlet_outward,gas_advective_flux_outlet_outward,"
          << "gas_advective_flux_all_boundaries,gas_diffusive_flux_inlet_outward,"
          << "gas_diffusive_flux_outlet_outward,gas_diffusive_flux_all_boundaries,"
          << "alpha_div_qg_volume_integral,alpha_divergence_theorem_defect,"
          << "alpha_divergence_theorem_relative,"
          << "alpha_native_mixture_advection_volume_integral,"
          << "alpha_user_mixture_advection_volume_integral,"
          << "alpha_advection_cancellation_defect,"
          << "alpha_advection_cancellation_relative,"
          << "gas_balance_residual,gas_balance_relative,"
          << "total_mass,dtotal_mass_dt,total_mass_flux_inlet_outward,"
          << "total_mass_flux_outlet_outward,total_mass_flux_all_boundaries,"
          << "total_mass_balance_residual,total_mass_balance_relative,"
          << "alpha_clip_delta_volume,alpha_clip_cumulative_volume,"
          << "qg_mask_delta_integral_x,qg_mask_delta_integral_y,"
          << "qg_mask_delta_integral_z,qg_mask_delta_magnitude_integral,"
          << "qg_mask_cumulative_integral_x,qg_mask_cumulative_integral_y,"
          << "qg_mask_cumulative_integral_z,qg_mask_cumulative_magnitude_integral,"
          << "gas_boundary_flux_cumulative,gas_cumulative_error_raw,"
          << "gas_cumulative_error_clip_corrected,mass_boundary_flux_cumulative,"
          << "mass_cumulative_error_raw,mass_cumulative_error_clip_corrected\n";
    }
    validationInitialized = true;
    return;
  }

  const dfloat elapsed = time - validationPreviousTime;
  cumulativeGasBoundaryIntegral +=
      0.5 * elapsed * (validationPreviousGasBoundaryFlux + gasBoundaryFlux);
  cumulativeMassBoundaryIntegral +=
      0.5 * elapsed * (validationPreviousMassBoundaryFlux + massBoundaryFlux);

  const dfloat dGasVolumeDt =
      validationBdfDerivative(state.gasVolume, gasVolumeHistory);
  const dfloat dTotalMassDt =
      validationBdfDerivative(state.totalMass, totalMassHistory);
  const dfloat gasResidual = dGasVolumeDt + gasBoundaryFlux;
  const dfloat massResidual = dTotalMassDt + massBoundaryFlux;
  dfloat gasScale = std::max(std::abs(dGasVolumeDt),
                             std::abs(state.gasAdvectiveInlet));
  gasScale = std::max(gasScale, std::abs(state.gasAdvectiveOutlet));
  gasScale = std::max(gasScale, std::abs(state.gasDiffusiveInlet));
  gasScale = std::max(gasScale, std::abs(state.gasDiffusiveOutlet));
  const dfloat gasRelative = std::abs(gasResidual) / std::max(gasScale, 1.0e-30);
  dfloat massScale = std::max(std::abs(dTotalMassDt),
                              std::abs(state.totalMassFluxInlet));
  massScale = std::max(massScale, std::abs(state.totalMassFluxOutlet));
  const dfloat massRelative =
      std::abs(massResidual) / std::max(massScale, 1.0e-30);
  const dfloat gasCumulativeError = state.gasVolume
                                    - validationInitialGasVolume
                                    + cumulativeGasBoundaryIntegral;
  const dfloat gasCumulativeErrorCorrected =
      gasCumulativeError - cumulativeAlphaClipDeltaVolume;
  const dfloat massCumulativeError = state.totalMass
                                     - validationInitialTotalMass
                                     + cumulativeMassBoundaryIntegral;
  const dfloat massClipCorrection =
      (p.rhoGas - p.rhoLiquid) * cumulativeAlphaClipDeltaVolume;
  const dfloat massCumulativeErrorCorrected =
      massCumulativeError - massClipCorrection;

  if (tstep % p.validationOutputInterval == 0
      && platform->comm.mpiRank() == 0) {
    validationFile << std::scientific << std::setprecision(16)
                   << tstep << ',' << time << ',' << nrs->dt[0] << ','
                   << state.gasVolume << ',' << dGasVolumeDt << ','
                   << dGasVolumeDt << ','
                   << state.gasAdvectiveInlet << ','
                   << state.gasAdvectiveOutlet << ','
                   << state.gasAdvectiveTotal << ','
                   << state.gasDiffusiveInlet << ','
                   << state.gasDiffusiveOutlet << ','
                   << state.gasDiffusiveTotal << ','
                   << state.alphaDivQgVolumeIntegral << ','
                   << state.alphaDivergenceTheoremDefect << ','
                   << state.alphaDivergenceTheoremRelative << ','
                   << state.alphaNativeMixtureAdvectionVolumeIntegral << ','
                   << state.alphaUserMixtureAdvectionVolumeIntegral << ','
                   << state.alphaAdvectionCancellationDefect << ','
                   << state.alphaAdvectionCancellationRelative << ','
                   << gasResidual << ',' << gasRelative << ','
                   << state.totalMass << ',' << dTotalMassDt << ','
                   << state.totalMassFluxInlet << ','
                   << state.totalMassFluxOutlet << ','
                   << state.totalMassFluxTotal << ','
                   << massResidual << ',' << massRelative << ','
                   << alphaClipDeltaVolume << ','
                   << cumulativeAlphaClipDeltaVolume << ','
                   << qgMaskDeltaIntegral[0] << ','
                   << qgMaskDeltaIntegral[1] << ','
                   << qgMaskDeltaIntegral[2] << ','
                   << qgMaskDeltaMagnitudeIntegral << ','
                   << cumulativeQgMaskDeltaIntegral[0] << ','
                   << cumulativeQgMaskDeltaIntegral[1] << ','
                   << cumulativeQgMaskDeltaIntegral[2] << ','
                   << cumulativeQgMaskDeltaMagnitudeIntegral << ','
                   << cumulativeGasBoundaryIntegral << ','
                   << gasCumulativeError << ','
                   << gasCumulativeErrorCorrected << ','
                   << cumulativeMassBoundaryIntegral << ','
                   << massCumulativeError << ','
                   << massCumulativeErrorCorrected << '\n';
    validationFile.flush();
  }

  validationPreviousTime = time;
  validationPreviousGasBoundaryFlux = gasBoundaryFlux;
  validationPreviousMassBoundaryFlux = massBoundaryFlux;
  gasVolumeHistory.insert(gasVolumeHistory.begin(), state.gasVolume);
  totalMassHistory.insert(totalMassHistory.begin(), state.totalMass);
  const int maxHistory = nrs->o_coeffBDF.size();
  if (gasVolumeHistory.size() > maxHistory) {
    gasVolumeHistory.resize(maxHistory);
    totalMassHistory.resize(maxHistory);
  }
}

// Host copies are made only at the requested monitor interval. Report both
// the largest gas drag relaxation rate and the largest physical mixture drag force.
inline void printDragLocations(double time, int tstep)
{
  const dlong N = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  const int rank = platform->comm.mpiRank();
  const MPI_Comm comm = platform->comm.mpiComm();
  std::vector<dfloat> a(N), rate(N), q(3*N), ug(3*N), ul(3*N), um(3*N);
  nrs->scalar->o_solution("alpha").copyTo(a, N);
  o_dragLambda.copyTo(rate, N);
  for (int c = 0; c < 3; ++c) {
    o_qg.copyTo(q, N, c*offset, c*N);
    o_ug.copyTo(ug, N, c*offset, c*N);
    o_ul.copyTo(ul, N, c*offset, c*N);
    nrs->fluid->o_U.slice(c*offset, N).copyTo(um.data()+c*N);
  }
  dlong indices[2] = {0, 0};
  double maxima[2] = {-1.0, -1.0};
  for (dlong n = 0; n < N; ++n) {
    const double ac = std::max(0.0, std::min(1.0, double(a[n])));
    const double beta = std::max(1.0-ac, double(p.alphaFloor));
    double force2 = 0.0;
    for (int c = 0; c < 3; ++c) {
      const double f = (p.mixtureDragEnabled != 0.0 ? 1.0 : 0.0)*ac*(1.0-p.rhoGas/p.rhoLiquid)*rate[n]*beta
                       *(ul[n+c*N]-ug[n+c*N]);
      force2 += f*f;
    }
    const double values[2] = {double(rate[n]), std::sqrt(force2)};
    for (int k = 0; k < 2; ++k) {
      if (values[k] > maxima[k]) { maxima[k] = values[k]; indices[k] = n; }
    }
  }
  for (int k = 0; k < 2; ++k) {
    struct { double value; int rank; } local{maxima[k], rank}, global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
    double data[27] = {};
    if (rank == global.rank && N > 0) {
      const dlong n = indices[k];
      dfloat xyz[3];
      nrs->meshV->o_x.slice(n, 1).copyTo(&xyz[0]);
      nrs->meshV->o_y.slice(n, 1).copyTo(&xyz[1]);
      nrs->meshV->o_z.slice(n, 1).copyTo(&xyz[2]);
      data[0] = double(n); data[1] = xyz[0]; data[2] = xyz[1]; data[3] = xyz[2];
      data[4] = a[n]; data[5] = rate[n]; data[6] = rate[n]*nrs->dt[0];
      double slip2 = 0.0;
      for (int c = 0; c < 3; ++c) {
        data[7+c] = q[n+c*N]; data[10+c] = ug[n+c*N];
        data[13+c] = ul[n+c*N]; data[16+c] = um[n+c*N];
        const double slip = ug[n+c*N]-ul[n+c*N]; slip2 += slip*slip;
      }
      data[19] = std::sqrt(slip2);
      data[20] = p.rhoLiquid*p.bubbleDiameter*data[19]/p.muLiquid;
      const double cdRe = data[20] < 1000.0
          ? 24.0*(1.0+0.15*std::pow(data[20],0.687)) : 0.44*data[20];
      data[21] = 0.75*cdRe*p.muLiquid/(p.bubbleDiameter*p.bubbleDiameter);
      const double ac = std::max(0.0, std::min(1.0, double(a[n])));
      const double beta = std::max(1.0-ac, double(p.alphaFloor));
      data[22] = (p.mixtureDragEnabled != 0.0 ? 1.0 : 0.0)*ac*(1.0-p.rhoGas/p.rhoLiquid)*rate[n];
      data[23] = data[22]*beta*data[19];
      double delta2 = 0.0;
      for (int c = 0; c < 3; ++c) {
        const double delta = ug[n+c*N]-um[n+c*N]; delta2 += delta*delta;
      }
      data[24] = std::sqrt(delta2);
      data[25] = p.dragSlipLimitEnabled != 0.0 && data[24] > p.dragSlipMaximum
          ? p.dragSlipMaximum/data[24] : 1.0;
      data[26] = data[20]*data[25];
    }
    MPI_Bcast(data, 27, MPI_DOUBLE, global.rank, comm);
    if (rank == 0) {
      printf("bubbleColumn2 dragLocation step=%d time=%.8e criterion=%s maximum=%.8e "
             "rank=%d localNode=%.0f xyz=(%.8e,%.8e,%.8e) alpha=%.8e "
             "lambdaD=%.8e lambdaDdt=%.8e qg=(%.8e,%.8e,%.8e) "
             "ug=(%.8e,%.8e,%.8e) ul=(%.8e,%.8e,%.8e) "
             "um=(%.8e,%.8e,%.8e) slip=%.8e Re=%.8e KiRaw=%.8e "
             "mixtureRate=%.8e |FD|=%.8e |ug-um|=%.8e dragSlipFactor=%.8e ReUsed=%.8e\n",
             tstep, time, k == 0 ? "lambdaD" : "mixtureDragMagnitude", global.value,
             global.rank, data[0], data[1], data[2], data[3], data[4], data[5], data[6],
             data[7], data[8], data[9], data[10], data[11], data[12],
             data[13], data[14], data[15], data[16], data[17], data[18], data[19],
             data[20], data[21], data[22], data[23], data[24], data[25], data[26]);
    }
  }
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

  vectorMagnitudeKernel(Nlocal, 3, offset, o_gradP, o_monitorMagnitude);
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

  vectorMagnitudeKernel(Nlocal, 3, offset, o_qg, o_monitorMagnitude);
  const dfloat maxQg = platform->linAlg->max(Nlocal, o_monitorMagnitude, comm);

  buildGasFluxConsistencyMonitorKernel(Nlocal,
                                       offset,
                                       alpha,
                                       o_qg,
                                       o_ug,
                                       o_monitorMagnitude);
  const dfloat maxQgConsistencyError =
      platform->linAlg->max(Nlocal, o_monitorMagnitude, comm);

  vectorMagnitudeKernel(Nlocal, 3, offset, o_ug, o_monitorMagnitude);
  const dfloat maxUg = platform->linAlg->max(Nlocal, o_monitorMagnitude, comm);
  const dfloat gasCfl = computeGasCfl();

  vectorMagnitudeKernel(
      Nlocal, 9, offset, o_driftStress, o_monitorMagnitude);
  const dfloat maxDriftStress =
      platform->linAlg->max(Nlocal, o_monitorMagnitude, comm);

  const dfloat maxDragLambda =
      platform->linAlg->max(Nlocal, o_dragLambda, comm);
  const dfloat maxDragStep = maxDragLambda * nrs->dt[0];
  printDragLocations(time, tstep);

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
    printf("bubbleColumn2 stability step=%d time=%.8e max|divTarget|=%.8e "
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
} // namespace bubbleColumn2
