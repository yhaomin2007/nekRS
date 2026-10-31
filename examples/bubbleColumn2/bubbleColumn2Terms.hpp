#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

#include "opSEM.hpp"
#include "explicitFilter.hpp"

namespace bubbleColumn2
{
struct Parameters {
  dfloat rhoLiquid;
  dfloat rhoGas;
  dfloat muLiquid;
  dfloat muGas;
  dfloat alphaInlet;
  dfloat ugInlet;
  dfloat ugInitial = 0.0;
  dfloat initialPlumeHeight = 0.05;
  dfloat initialPlumeThickness = 0.01;
  dfloat gravity[3];
  dfloat alphaFloor;
  dfloat alphaClippingEnabled;
  int alphaExplicitFilterModes = 1;
  dfloat alphaExplicitFilterStrength = 0.0;
  dfloat alphaMinimum;
  dfloat alphaMaximum;
  dfloat subtractAlphaDiffusion;
  dfloat subtractUgDiffusion[3];
  dfloat zRampBottom;
  dfloat zRampTop;
  dfloat outletDampingFactor;
  dfloat smoothGasVelocityMaskEnabled;
  int gasVelocityLowAlphaCutoffEnabled = 0;
  dfloat gasVelocityClipEnabled;
  dfloat gasVelocityMaximum;
  dfloat volumeVelocityClipEnabled;
  dfloat volumeVelocityMaximum;
  dfloat gasMomentumCutoff;
  dfloat gasMomentumFullyActive;
  dfloat gasPressureEnabled;
  int gasPressureInletMaskEnabled = 0;
  int gasPressureInletRegionEnabled = 0;
  dfloat gasPressureInletZeroEndZ = 0.01;
  dfloat gasPressureInletRampEndZ = 0.02;
  int gasPressureGradientMethod = 0;
  dfloat gasPressureGradientFilterEnabled = 0.0;
  dfloat gasPressureGradientFilterWeight = 0.2;
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
  int virtualMassTimeDerivativeOrder = 2;
  int virtualMassStartStep = 0;
  int virtualMassRampStep = 0;
  dfloat stabilityMonitorEnabled;
  int stabilityMonitorInterval;
  int validationOutputInterval;
};

static Parameters p;
static deviceMemory<dfloat> o_gasTransportFields;
static deviceMemory<dfloat> o_gasCubatureFields;
static deviceMemory<dfloat> o_gasCubatureProducts;
static deviceMemory<dfloat> o_gasTransport;
static occa::kernel gasInterpolateKernel, gasCubatureProductsKernel;
static occa::kernel gasProjectKernel, alphaConservativeCubatureKernel;
static deviceMemory<dfloat> o_gasAdvector;
static deviceMemory<dfloat> o_scalarCoeffEXT;

static deviceMemory<dfloat> o_nativeTransportAdvection;
static deviceMemory<dfloat> o_gradUv;
static deviceMemory<dfloat> o_exactMixtureStress, o_baseNativeStress;
static deviceMemory<dfloat> o_divExactMixtureStress, o_divBaseNativeStress;
static occa::kernel buildMixtureStressCorrectionKernel, addMixtureStressCorrectionKernel;
static deviceMemory<dfloat> o_ug;
static deviceMemory<dfloat> o_ul;
static deviceMemory<dfloat> o_ulPrevious;
static deviceMemory<dfloat> o_ugPrevious;
static deviceMemory<dfloat> o_ulPrevious2, o_ugPrevious2;
static int virtualMassHistorySamples = 0;
static dfloat virtualMassPreviousDt = 0.0;
static deviceMemory<dfloat> o_gradUl;
static deviceMemory<dfloat> o_virtualMassRelativeAcceleration, o_liquidAcceleration;
static deviceMemory<dfloat> o_qg;
static deviceMemory<dfloat> o_gradUg;
static deviceMemory<dfloat> o_gradP;
static deviceMemory<dfloat> o_gradPFilterHistory, o_gasPressureRegionWeight;
static deviceMemory<dfloat> o_gradPStrong, o_gradPWeak;
static deviceMemory<dfloat> o_gasPressureInletMask, o_inletVelocityWeight;
static bool pressureGradientFilterInitialized = false;
static int pressureGradientFilterStep = -1;
static deviceMemory<dfloat> o_rhoM;
static deviceMemory<dfloat> o_rhoPressure;
static deviceMemory<dfloat> o_muM;
static deviceMemory<dfloat> o_alphaDiffusionDivergence;
static deviceMemory<dfloat> o_ugDiffusionDivergence;
static deviceMemory<dfloat> o_scalarDiffusionBase;
static deviceMemory<dfloat> o_outletRampFactor;
static deviceMemory<dfloat> o_driftStress;
static deviceMemory<dfloat> o_divDriftStress;
static deviceMemory<dfloat> o_gasStress;
static deviceMemory<dfloat> o_divGasStress;
static deviceMemory<dfloat> o_alphaSource;
static deviceMemory<dfloat> o_ugSource;
static deviceMemory<dfloat> o_dragLambda;
static deviceMemory<dfloat> o_mixtureDragSource;
static deviceMemory<dfloat> o_previousMixtureDragSource;
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
static dfloat alphaFilterDeltaVolume = 0.0;
static dfloat cumulativeAlphaFilterDeltaVolume = 0.0;
static dfloat alphaClipDeltaVolume = 0.0;
static dfloat cumulativeAlphaClipDeltaVolume = 0.0;
static dfloat qgMaskDeltaIntegral[3] = {0.0, 0.0, 0.0};
static dfloat qgMaskDeltaMagnitudeIntegral = 0.0;
static dfloat cumulativeQgMaskDeltaIntegral[3] = {0.0, 0.0, 0.0};
static dfloat cumulativeQgMaskDeltaMagnitudeIntegral = 0.0;
static bool validationInitialized = false;
static occa::kernel vectorMagnitudeKernel;
static occa::kernel reconstructGasVelocityKernel;
static occa::kernel postProcessGasVelocityKernel;
static occa::kernel clipAlphaKernel;
static occa::kernel clipVectorMagnitudeKernel;
static occa::kernel initializePlumeKernel, buildGasFluxKernel, addGasStressKernel;
static occa::kernel buildLiquidVelocityKernel;
static occa::kernel updateVirtualMassHistoryKernel;
static occa::kernel buildEquationTermsKernel;
static occa::kernel buildMixtureForceKernel;
static occa::kernel buildGasPressureAccelerationMonitorKernel;
static occa::kernel buildGasFluxConsistencyMonitorKernel;
static occa::kernel buildOutletRampFactorKernel;

inline void registerKernels(deviceKernelProperties &kernelInfo)
{
  userExplicitFilter::registerKernels(kernelInfo);
  {
    auto cubInfo = kernelInfo;
    int cubN = 0;
    platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
    cubInfo.define("p_cubNq") = cubN + 1;
    cubInfo.define("p_cubNp") = (cubN + 1) * (cubN + 1) * (cubN + 1);
    const std::string req = "bubbleColumn2::transport";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(req, "bubbleColumn2Transport.okl", cubInfo);
    } else {
      gasInterpolateKernel = platform->kernelRequests.load(req, "gasInterpolate");
      gasCubatureProductsKernel = platform->kernelRequests.load(req, "gasCubatureProducts");
      gasProjectKernel = platform->kernelRequests.load(req, "gasProject");
      alphaConservativeCubatureKernel = platform->kernelRequests.load(req, "alphaConservativeCubature");
    }
  }

  const std::string request = "bubbleColumn2::equations";
  // This is standalone OKL source.  The .okl suffix is required so OCCA
  // translates @kernel/@globalPtr before invoking the HIP compiler.
  const std::string fileName = "bubbleColumn2Equations.okl";
  if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    platform->kernelRequests.add(request, fileName, kernelInfo);
  } else {
    vectorMagnitudeKernel = platform->kernelRequests.load(request, "vectorMagnitude");
    buildMixtureStressCorrectionKernel = platform->kernelRequests.load(request, "buildMixtureStressCorrection");
    addMixtureStressCorrectionKernel = platform->kernelRequests.load(request, "addMixtureStressCorrection");
    reconstructGasVelocityKernel =
        platform->kernelRequests.load(request, "reconstructGasVelocity");
    postProcessGasVelocityKernel =
        platform->kernelRequests.load(request, "postProcessGasVelocity");
    clipAlphaKernel = platform->kernelRequests.load(request, "clipAlpha");
    clipVectorMagnitudeKernel =
        platform->kernelRequests.load(request, "clipVectorMagnitude");
    buildGasFluxKernel = platform->kernelRequests.load(request, "buildGasFlux");
    addGasStressKernel = platform->kernelRequests.load(request, "addGasStress");
    initializePlumeKernel = platform->kernelRequests.load(request, "initializePlume");
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

// Fixed-mesh UG pressure-source weight. z coordinates are in mesh units.
inline dfloat gasPressureInletRampWeight(dfloat z)
{
  if (!p.gasPressureInletRegionEnabled) return 1.0;
  dfloat t = (z - p.gasPressureInletZeroEndZ)
      / (p.gasPressureInletRampEndZ - p.gasPressureInletZeroEndZ);
  t = std::max(dfloat(0), std::min(dfloat(1), t));
  return t * t * (3.0 - 2.0 * t);
}

inline void allocate()
{
  const dlong offset = nrs->fieldOffset;
  nekrsCheck(nrs->meshV->cubNq < 2 * nrs->meshV->N + 1
                 || !platform->options.compareArgs("ADVECTION TYPE", "CUBATURE"),
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Conservative gas transport requires cubaturePolynomialOrder >= 2*polynomialOrder and cubature advection.\n");
  nekrsCheck(platform->options.compareArgs("MOVING MESH", "TRUE"),
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Gas transport requires a fixed mesh.\n");
  nekrsCheck(p.gasPressureInletMaskEnabled < 0 || p.gasPressureInletMaskEnabled > 1
                 || p.gasPressureGradientMethod < 0 || p.gasPressureGradientMethod > 1,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Gas pressure mask/gradient method must be 0 or 1.\n");
  nekrsCheck(p.gasPressureInletRegionEnabled < 0 || p.gasPressureInletRegionEnabled > 1
                 || (p.gasPressureInletRegionEnabled
                     && (!std::isfinite(p.gasPressureInletZeroEndZ)
                         || !std::isfinite(p.gasPressureInletRampEndZ)
                         || p.gasPressureInletRampEndZ <= p.gasPressureInletZeroEndZ)),
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Gas pressure inlet region requires enabled=0/1 and finite rampEndZ > zeroEndZ.\n");
  nekrsCheck(!std::isfinite(p.gasPressureGradientFilterWeight)
                 || p.gasPressureGradientFilterWeight <= 0 || p.gasPressureGradientFilterWeight > 1,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Gas pressure gradient filter weight must be in (0,1].\n");
  nekrsCheck(p.alphaExplicitFilterModes < 1 || p.alphaExplicitFilterModes > nrs->meshV->N - 1
                 || !std::isfinite(p.alphaExplicitFilterStrength)
                 || p.alphaExplicitFilterStrength < 0 || p.alphaExplicitFilterStrength > 1,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Alpha explicit filter requires modes in [1,N-1] and strength in [0,1].\n");
  const dlong cubOffset = nrs->meshV->Nelements * nrs->meshV->cubNp;
  o_gasTransportFields.resize(4 * offset);
  o_gasTransport.resize(4 * offset);
  o_gasCubatureFields.resize(4 * cubOffset);
  o_gasCubatureProducts.resize(4 * cubOffset);
  o_gasAdvector.resize(3 * offset);

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
  o_ulPrevious2.resize(3 * offset);
  o_ugPrevious2.resize(3 * offset);
  o_gradUl.resize(9 * offset);
  o_virtualMassRelativeAcceleration.resize(3 * offset);
  o_liquidAcceleration.resize(3 * offset);
  o_qg.resize(3 * offset);
  o_gradUg.resize(9 * offset);
  o_gradP.resize(3 * offset);
  o_gradPFilterHistory.resize(3 * offset);
  {
    auto mesh = nrs->meshV;
    std::vector<dfloat> weight(offset, 1.0);
    for (dlong e = 0; e < mesh->Nelements; ++e) {
      for (int f = 0; f < mesh->Nfaces; ++f) {
        if (mesh->EToB[e * mesh->Nfaces + f] != 3) continue;
        for (int n = 0; n < mesh->Nfp; ++n) {
          const dlong faceNode = (e * mesh->Nfaces + f) * mesh->Nfp + n;
          weight[mesh->vmapM[faceNode]] = 0.0;
        }
      }
    }
    o_inletVelocityWeight.resize(offset);
    o_inletVelocityWeight.copyFrom(weight.data(), offset);
    // Wall precedence at shared inlet/wall nodes, including remote copies.
    oogs::startFinish(o_inletVelocityWeight, 1, 0,
                      ogsDfloat, ogsMin, mesh->oogs);
    // This case reserves boundary usrwrk for the nodal inlet velocity weight.
    platform->app->bc->o_usrwrk = o_inletVelocityWeight;
  }
  if (p.gasPressureInletMaskEnabled) {
    auto mesh = nrs->meshV;
    std::vector<dfloat> keep(offset, 1.0);
    for (dlong e = 0; e < mesh->Nelements; ++e) {
      for (int f = 0; f < mesh->Nfaces; ++f) {
        if (mesh->EToB[e * mesh->Nfaces + f] != 1) continue;
        for (int n = 0; n < mesh->Nfp; ++n) {
          const dlong faceNode = (e * mesh->Nfaces + f) * mesh->Nfp + n;
          keep[mesh->vmapM[faceNode]] = 0.0;
        }
      }
    }
    o_gasPressureInletMask.resize(offset);
    o_gasPressureInletMask.copyFrom(keep.data(), offset);
    // Propagate inlet membership to every CG copy, including wall corners
    // and copies on other MPI ranks. A zero on any copy masks them all.
    oogs::startFinish(o_gasPressureInletMask, 1, 0,
                      ogsDfloat, ogsMin, mesh->oogs);
  }
  o_gradPStrong.resize(3 * offset);
  o_gradPWeak.resize(3 * offset);
  if (p.gasPressureInletRegionEnabled) {
    const dlong Nlocal = nrs->meshV->Nlocal;
    std::vector<dfloat> z(Nlocal), weight(offset, 1.0);
    nrs->meshV->o_z.copyTo(z.data(), Nlocal);
    for (dlong n = 0; n < Nlocal; ++n) weight[n] = gasPressureInletRampWeight(z[n]);
    o_gasPressureRegionWeight.resize(offset);
    o_gasPressureRegionWeight.copyFrom(weight.data(), offset);
  }
  o_rhoM.resize(offset);
  o_rhoPressure.resize(offset);
  o_muM.resize(offset);
  o_alphaDiffusionDivergence.resize(offset);
  o_ugDiffusionDivergence.resize(3 * offset);
  o_scalarDiffusionBase.resize(4 * offset);
  o_outletRampFactor.resize(offset);
  o_driftStress.resize(9 * offset);
  o_divDriftStress.resize(3 * offset);
  o_gasStress.resize(9 * offset);
  o_divGasStress.resize(3 * offset);
  o_alphaSource.resize(offset);
  o_ugSource.resize(3 * offset);
  o_dragLambda.resize(offset);
  o_mixtureDragSource.resize(3 * offset);
  o_previousMixtureDragSource.resize(3 * offset);
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
  const char *scalarNames[4] = {"alpha", "ugx", "ugy", "ugz"};
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
                               p.gasVelocityLowAlphaCutoffEnabled,
                               p.alphaFloor,
                               nrs->scalar->o_solution("alpha"),
                               nrs->scalar->o_solution("ugx"),
                               nrs->scalar->o_solution("ugy"),
                               nrs->scalar->o_solution("ugz"),
                               o_ug);
  buildGasFluxKernel(nrs->meshV->Nlocal, nrs->fieldOffset,
                     nrs->scalar->o_solution("alpha"), o_ug, o_qg);
}

inline void reconstructLiquidVelocity()
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  reconstructGasVelocity();
  buildLiquidVelocityKernel(Nlocal,
                            offset,
                            p.alphaFloor,
                            nrs->scalar->o_solution("alpha"),
                            nrs->fluid->o_U,
                            o_qg,
                            o_ul);
}

inline dfloat virtualMassRampFactor(int step)
{
  if (step <= p.virtualMassStartStep) return 0.0;
  if (p.virtualMassRampStep == 0) return 1.0;
  return std::min<dfloat>(1.0,
      (static_cast<dfloat>(step) - p.virtualMassStartStep) / p.virtualMassRampStep);
}

inline void evaluateGasCubatureTransport(bool conservativeAlpha = false)
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  const dlong cubOffset = mesh->Nelements * mesh->cubNp;
  o_gasTransportFields.copyFrom(nrs->scalar->o_solution("alpha"), mesh->Nlocal);
  for (int i = 0; i < 3; ++i) {
    o_gasTransportFields.copyFrom(o_gasAdvector, mesh->Nlocal, (i + 1) * offset, i * offset);
  }
  gasInterpolateKernel(mesh->Nelements, offset, cubOffset,
                       mesh->o_cubInterpT, o_gasTransportFields, o_gasCubatureFields);
  if (conservativeAlpha) {
    alphaConservativeCubatureKernel(mesh->Nelements, cubOffset, mesh->o_cubD,
        mesh->o_cubvgeo, o_gasCubatureFields, o_gasCubatureProducts);
  } else {
    gasCubatureProductsKernel(mesh->Nelements, cubOffset, mesh->o_cubD,
        mesh->o_cubvgeo, o_gasCubatureFields, o_gasCubatureProducts);
  }
  gasProjectKernel(mesh->Nelements, offset, cubOffset, mesh->o_cubInterpT,
                   mesh->o_vgeo, o_gasCubatureProducts, o_gasTransport);
  // Mass-weighted CG assembly of normalized nodal loads.
  platform->linAlg->axmyMany(mesh->Nlocal, 4, offset, 0, 1.0, mesh->o_LMM, o_gasTransport);
  oogs::startFinish(o_gasTransport, 4, offset, ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmyMany(mesh->Nlocal, 4, offset, 0, 1.0, mesh->o_invLMM, o_gasTransport);
  // Component zero is full div(alpha*UG) in conservative mode, compression otherwise.
}

inline void prepareGasAdvection()
{
  const dlong offset = nrs->fieldOffset;
  // Evaluate nonlinear convection at one time level. Native sumMakef applies
  // EXT to its history; predicting the advector here would extrapolate twice.
  reconstructGasVelocity();
  o_gasAdvector.copyFrom(o_ug, 3 * offset);
}

// Form the UG-only pressure source from the selected lagged gradient.
// Spatial attenuation never changes raw gradients or temporal filter history.
inline void prepareGasPressureForcing()
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  const auto &o_selectedGradP =
      (p.gasPressureGradientMethod == 1) ? o_gradPWeak : o_gradPStrong;

  if (p.gasPressureGradientFilterEnabled == 0.0) {
    o_gradP.copyFrom(o_selectedGradP, 3 * offset);
  } else {
    // At source assembly, o_P contains the pressure from the most recently
    // completed physical step. Advance the exponential history only once per
    // timestep because this routine can be called more than once in that step.
    const int tstep = nrs->tstep;
    if (!pressureGradientFilterInitialized) {
      o_gradPFilterHistory.copyFrom(o_selectedGradP, 3 * offset);
      pressureGradientFilterInitialized = true;
      pressureGradientFilterStep = tstep;
    } else if (tstep != pressureGradientFilterStep) {
      for (int i = 0; i < 3; ++i) {
        const auto laggedGradient =
            o_selectedGradP.slice(i * offset, offset);
        auto filteredGradient = o_gradPFilterHistory.slice(i * offset, offset);
        platform->linAlg->axpby(mesh->Nlocal,
                                p.gasPressureGradientFilterWeight,
                                laggedGradient,
                                1.0 - p.gasPressureGradientFilterWeight,
                                filteredGradient);
      }
      pressureGradientFilterStep = tstep;
    }
    o_gradP.copyFrom(o_gradPFilterHistory, 3 * offset);
  }

  if (p.gasPressureInletMaskEnabled) {
    platform->linAlg->axmyMany(mesh->Nlocal, 3, offset, 0, 1.0,
                               o_gasPressureInletMask, o_gradP);
  }
  if (p.gasPressureInletRegionEnabled) {
    platform->linAlg->axmyMany(mesh->Nlocal, 3, offset, 0, 1.0,
                               o_gasPressureRegionWeight, o_gradP);
  }
}

inline void evaluatePointwiseTerms()
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  auto alpha = nrs->scalar->o_solution("alpha");

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
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg);
  opSEM::strongGradVec(mesh, offset, o_ul, o_gradUl);
  opSEM::strongGradVec(mesh, offset, nrs->fluid->o_U, o_gradUv);
  // Compute both pressure-gradient discretizations from the same lagged
  // pressure field so they can be compared directly.
  //
  // Strong: assembled pointwise grad(p), as used previously in this case.
  opSEM::strongGrad(mesh, offset, nrs->fluid->o_P, o_gradPStrong);
  //
  // Weak: reproduce the volume weak-gradient operator used by the native
  // NekRS velocity RHS. core-wGradientVolumeHex3D returns the element weak
  // load G_w p. Gather it, divide by the assembled lumped mass, and reverse
  // sign so o_gradPWeak has the same physical sign convention as +grad(p).
  // When this field is subsequently multiplied by the scalar mass matrix in
  // scalar_t::sumMakef, -o_gradPWeak/rhoEff reproduces the weak volume
  // pressure force (up to the phase coefficient and scalar boundary terms).
  launchKernel("core-wGradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               offset,
               nrs->fluid->o_P,
               o_gradPWeak);
  oogs::startFinish(
      o_gradPWeak, mesh->dim, offset, ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmyMany(
      mesh->Nlocal, mesh->dim, offset, 0, -1.0, mesh->o_invLMM, o_gradPWeak);

  prepareGasPressureForcing();

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
                           p.virtualMassEnabled * virtualMassRampFactor(nrs->tstep),
                           p.virtualMassCoefficient,
                           p.gravity[0],
                           p.gravity[1],
                           p.gravity[2],
                           alpha,
                           nrs->fluid->o_U,
                           o_ug,
                           o_ul,
                           o_gradUg,
                           o_virtualMassRelativeAcceleration,
                           o_liquidAcceleration,
                           o_gradP,
                           o_rhoM,
                           o_rhoPressure,
                           o_muM,
                           o_driftStress,
                           o_gasStress,
                           o_alphaSource,
                           o_ugSource,
                           o_dragLambda,
                           o_mixtureInterphaseAcceleration,
                           o_mixtureDragSource);

  evaluateGasCubatureTransport(true);
  platform->linAlg->axpby(mesh->Nlocal, -1.0, o_gasTransport, 0.0, o_alphaSource);

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
  const char *scalarNames[4] = {"alpha", "ugx", "ugy", "ugz"};
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
    addGasStressKernel(mesh->Nlocal, p.alphaFloor,
        p.rhoGas + p.virtualMassEnabled * virtualMassRampFactor(nrs->tstep)
                   * p.virtualMassCoefficient * p.rhoLiquid,
        alpha, stressDivergence, o_ugSource.slice(i * offset, offset));
  }
}

inline void evaluateNativeAlphaAdvection(deviceMemory<dfloat> &o_advection,
                                         bool averageSharedNodes, const char *name = "alpha", bool mixture = true)
{
  auto mesh = nrs->meshV;
  const int alphaIndex = nrs->scalar->nameToIndex.at(name);

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
                 mixture ? nrs->fluid->o_relUrst : nrs->scalar->o_relUrst,
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
                 mixture ? nrs->fluid->o_relUrst : nrs->scalar->o_relUrst,
                 nrs->scalar->o_rho,
                 o_advection);
  }

  if (averageSharedNodes) {
    const dlong alphaOffset = nrs->scalar->fieldOffsetScan[alphaIndex];
    auto o_alphaAdvection =
        o_advection.slice(alphaOffset, nrs->fieldOffset);
    // Native weighted=0 advection is already normalized. Assemble its
    // nodal values with the mass matrix, unlike JW-weighted opSEM gradients.
    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_alphaAdvection);
    oogs::startFinish(o_alphaAdvection,
                      1,
                      0,
                      ogsDfloat,
                      ogsAdd,
                      mesh->oogs);
    platform->linAlg->axmy(
        mesh->Nlocal, 1.0, mesh->o_invLMM, o_alphaAdvection);
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

  evaluateNativeAlphaAdvection(o_alphaNativeAdvectionDiagnostic, true);
  const int index = nrs->scalar->nameToIndex.at("alpha");
  o_alphaUserAdvectionDiagnostic.copyFrom(o_alphaNativeAdvectionDiagnostic,
      Nlocal, 0, nrs->scalar->fieldOffsetScan[index]);
  // Preserve element-local native values for downstream mass-assembly diagnostics.
  evaluateNativeAlphaAdvection(o_alphaNativeAdvectionDiagnostic, false);
  alphaAdvectionDiagnosticStep = nrs->tstep;
}

inline void initializeHistory()
{
  nrs->scalar->o_U = nrs->fluid->o_U;
  nrs->scalar->o_Ue = nrs->fluid->o_Ue;
  nrs->scalar->o_relUrst = nrs->fluid->o_relUrst;
  const auto nEXT = nrs->scalar->o_coeffEXT.size();
  o_scalarCoeffEXT.resize(nEXT);
  std::vector<dfloat> coefficients(nEXT, 0.0);
  coefficients[0] = 1.0;
  o_scalarCoeffEXT.copyFrom(coefficients.data(), nEXT);
  nrs->scalar->o_coeffEXT = o_scalarCoeffEXT;
  prepareGasAdvection();

  const dlong offset = nrs->fieldOffset;
  platform->linAlg->fill(3 * offset, 0.0, o_virtualMassRelativeAcceleration);
  platform->linAlg->fill(3 * offset, 0.0, o_liquidAcceleration);
  evaluatePointwiseTerms();
  o_ulPrevious.copyFrom(o_ul, 3 * offset);
  o_ugPrevious.copyFrom(o_ug, 3 * offset);
  o_ulPrevious2.copyFrom(o_ul, 3 * offset);
  o_ugPrevious2.copyFrom(o_ug, 3 * offset);
  virtualMassHistorySamples = 0;
  virtualMassPreviousDt = 0.0;
  platform->linAlg->fill(3 * offset, 0.0, o_virtualMassRelativeAcceleration);
  platform->linAlg->fill(3 * offset, 0.0, o_liquidAcceleration);
}

inline void postProcessGasVelocity()
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  if ((!p.gasVelocityLowAlphaCutoffEnabled || p.smoothGasVelocityMaskEnabled == 0.0)
      && p.gasVelocityClipEnabled == 0.0) {
    for (int i = 0; i < 3; ++i) {
      qgMaskDeltaIntegral[i] = 0.0;
    }
    qgMaskDeltaMagnitudeIntegral = 0.0;
    return;
  }

  reconstructGasVelocity();
  o_qgBeforeMask.copyFrom(o_qg);

  postProcessGasVelocityKernel(
      Nlocal,
      p.gasVelocityLowAlphaCutoffEnabled,
      p.smoothGasVelocityMaskEnabled,
      p.gasVelocityClipEnabled,
      p.gasVelocityMaximum,
      p.alphaFloor,
      p.gasMomentumCutoff,
      p.gasMomentumFullyActive,
      nrs->scalar->o_solution("alpha"),
      nrs->scalar->o_solution("ugx"),
      nrs->scalar->o_solution("ugy"),
      nrs->scalar->o_solution("ugz"));

  reconstructGasVelocity();
  const MPI_Comm comm = platform->comm.mpiComm();
  for (int i = 0; i < 3; ++i) {
    auto delta = o_qgMaskDelta.slice(i * offset, offset);
    delta.copyFrom(o_qg, Nlocal, 0, i * offset);
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

inline void assembleNodalField(deviceMemory<dfloat>& o_divergence)
{
  auto mesh = nrs->meshV;
  const dlong Nlocal = mesh->Nlocal;

  // The completed divergence target may contain element-local contributions,
  // particularly after modal filtering or when recovering scalar
  // regularization. Convert it to one mass-weighted CG value at every shared
  // node before supplying it to the pressure projection:
  //   q_CG = M_L^{-1} Q^T M_e q_e.
  platform->linAlg->axmy(Nlocal, 1.0, mesh->o_LMM, o_divergence);
  oogs::startFinish(
      o_divergence, 1, 0, ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmy(Nlocal, 1.0, mesh->o_invLMM, o_divergence);
}

inline void postProcessAlpha(double time, int)
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  const MPI_Comm comm = platform->comm.mpiComm();
  auto alpha = nrs->scalar->o_solution("alpha");
  alphaFilterDeltaVolume = 0.0;
  alphaClipDeltaVolume = 0.0;
  if (p.alphaExplicitFilterStrength == 0.0 && p.alphaClippingEnabled == 0.0) return;
  dfloat volumeBefore = platform->linAlg->innerProd(
      Nlocal, nrs->meshV->o_LMM, alpha, comm);
  if (p.alphaExplicitFilterStrength > 0.0) {
    userExplicitFilter::explicitFilterApply("alpha", nrs->meshV,
        p.alphaExplicitFilterModes, p.alphaExplicitFilterStrength, alpha);
    assembleNodalField(alpha);
    // Reimpose prescribed scalar boundary values after direct field filtering.
    nrs->scalar->applyDirichlet(time);
    const dfloat volumeAfter = platform->linAlg->innerProd(
        Nlocal, nrs->meshV->o_LMM, alpha, comm);
    alphaFilterDeltaVolume = volumeAfter - volumeBefore;
    cumulativeAlphaFilterDeltaVolume += alphaFilterDeltaVolume;
    volumeBefore = volumeAfter;
  }
  if (p.alphaClippingEnabled != 0.0) {
    clipAlphaKernel(Nlocal, p.alphaMinimum, p.alphaMaximum, alpha);
    const dfloat volumeAfter = platform->linAlg->innerProd(
        Nlocal, nrs->meshV->o_LMM, alpha, comm);
    alphaClipDeltaVolume = volumeAfter - volumeBefore;
    cumulativeAlphaClipDeltaVolume += alphaClipDeltaVolume;
  }
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

inline void virtualMassTimeCoefficients(dfloat dt, dfloat previousDt,
                                       bool secondOrder, dfloat &c0, dfloat &c1, dfloat &c2)
{
  c0 = 1.0 / dt;
  c1 = -c0;
  c2 = 0.0;
  if (secondOrder) {
    c0 = (2.0 * dt + previousDt) / (dt * (dt + previousDt));
    c1 = -(dt + previousDt) / (dt * previousDt);
    c2 = dt / (previousDt * (dt + previousDt));
  }
}

inline void updateVirtualMassHistory()
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  auto alpha = nrs->scalar->o_solution("alpha");

  reconstructGasVelocity();
  buildLiquidVelocityKernel(mesh->Nlocal,
                            offset,
                            p.alphaFloor,
                            alpha,
                            nrs->fluid->o_U,
                            o_qg,
                            o_ul);
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg);
  opSEM::strongGradVec(mesh, offset, o_ul, o_gradUl);
  dfloat c0, c1, c2;
  virtualMassTimeCoefficients(nrs->dt[0], virtualMassPreviousDt,
      p.virtualMassTimeDerivativeOrder == 2 && virtualMassHistorySamples >= 1,
      c0, c1, c2);
  updateVirtualMassHistoryKernel(mesh->Nlocal,
                                 offset,
                                 c0, c1, c2,
                                 o_ug,
                                 o_ul,
                                 o_ulPrevious,
                                 o_ugPrevious,
                                 o_ulPrevious2,
                                 o_ugPrevious2,
                                 o_gradUl,
                                 o_gradUg,
                                 o_virtualMassRelativeAcceleration,
                                 o_liquidAcceleration);
  virtualMassPreviousDt = nrs->dt[0];
  ++virtualMassHistorySamples;
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

  const char *ugNames[3] = {"ugx", "ugy", "ugz"};
  for (int i = 0; i < 3; ++i) {
    if (p.subtractUgDiffusion[i] == 0.0) {
      continue;
    }
    auto diffusion = nrs->scalar->o_diffusionCoeff(ugNames[i]);
    auto divergence = o_ugDiffusionDivergence.slice(i * offset, offset);
    launchKernel("core-weakLaplacianHex3D",
                 nrs->meshV->Nelements,
                 1,
                 o_scalarFieldOffsetScan,
                 nrs->meshV->o_ggeo,
                 nrs->meshV->o_D,
                 diffusion,
                 nrs->scalar->o_solution(ugNames[i]),
                 divergence);
    oogs::startFinish(
        divergence, 1, 0, ogsDfloat, ogsAdd, nrs->meshV->oogs);
    platform->linAlg->axmy(
        Nlocal, 1.0, nrs->meshV->o_invLMM, divergence);
    platform->linAlg->axpby(Nlocal,
                            p.subtractUgDiffusion[i],
                            divergence,
                            1.0,
                            o_ugSource,
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
  for (int stage = 1; stage < nrs->fluid->o_coeffEXT.size(); ++stage) {
    for (int i = 0; i < 3; ++i) {
      const dlong dst = stage * nrs->fluid->fieldOffsetSum + i * offset;
      platform->linAlg->axpby(N, -1.0, o_previousMixtureDragSource, 1.0,
          nrs->fluid->o_EXT, i * offset, dst);
      platform->linAlg->axpby(N, 1.0, o_mixtureDragSource, 1.0,
          nrs->fluid->o_EXT, i * offset, dst);
    }
  }
  o_previousMixtureDragSource.copyFrom(o_mixtureDragSource);
}

inline void addExplicitSources(double)
{
  prepareGasAdvection();
  evaluatePointwiseTerms();
  subtractScalarDiffusion();
  evaluateMixtureForce();
  rampMixtureDrag(nrs->tstep);
  freezeDragHistory();
  captureAlphaAdvectionDiagnostics();
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;

  // Copy only entries written by the pointwise kernels. Avoid whole-view
  // copies because scalar and fluid fields may have different padded extents.
  nrs->scalar->o_explicitTerms("alpha").copyFrom(o_alphaSource, Nlocal);
  // Gas momentum remains material-velocity form: subtract UG.grad(UG),
  // not div(UG tensor UG). Alpha has already received -div(alpha*UG).
  evaluateGasCubatureTransport(false);
  const char *names[4] = {"alpha", "ugx", "ugy", "ugz"};
  for (int i = 0; i < 3; ++i) {
    auto source = o_ugSource.slice(i * offset, offset);
    platform->linAlg->axpby(Nlocal, -1.0,
        o_gasTransport.slice((i + 1) * offset, offset), 1.0, source);
    nrs->scalar->o_explicitTerms(names[i + 1]).copyFrom(source, Nlocal);
  }
  for (int i = 0; i < 4; ++i) {
    evaluateNativeAlphaAdvection(o_nativeTransportAdvection, true, names[i], true);
    const int index = nrs->scalar->nameToIndex.at(names[i]);
    platform->linAlg->axpby(Nlocal, 1.0,
        o_nativeTransportAdvection.slice(nrs->scalar->fieldOffsetScan[index], offset),
        1.0, nrs->scalar->o_explicitTerms(names[i]));
  }

  auto fluidTerms = nrs->fluid->o_explicitTerms();
  for (int i = 0; i < 3; ++i) {
    fluidTerms.copyFrom(o_mixtureForce, Nlocal, i * offset, i * offset);
  }
}

inline occa::memory implicitGasDrag(double, int scalarIndex)
{
  if (scalarIndex >= 1 && scalarIndex <= 3) return o_dragLambda;
  return o_NULL;
}

inline void updateProperties(double)
{
  // Called after all four scalars advance. Refresh the effective viscosity and
  // pressure mobility; the projected volume velocity remains divergence free.
  postProcessGasVelocity();
  prepareGasAdvection();
  evaluatePointwiseTerms();
  nrs->fluid->o_prop.slice(0 * nrs->fieldOffset, nrs->fieldOffset).copyFrom(o_muM);
  nrs->fluid->o_prop.slice(1 * nrs->fieldOffset, nrs->fieldOffset)
      .copyFrom(o_rhoPressure);
  // userProperties bypasses nrs_t's automatic scalar AVM invocation.
  // Apply after baseline diffusion/outlet ramp updates, once per callback.
  // Native AVM selects only scalars configured with regularization = avm.
  nrs->scalar->applyAVM();
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

  reconstructGasVelocity();
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
    cumulativeAlphaFilterDeltaVolume = 0.0;
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
          << "alpha_filter_delta_volume,alpha_filter_cumulative_volume,"
          << "qg_mask_delta_integral_x,qg_mask_delta_integral_y,"
          << "qg_mask_delta_integral_z,qg_mask_delta_magnitude_integral,"
          << "qg_mask_cumulative_integral_x,qg_mask_cumulative_integral_y,"
          << "qg_mask_cumulative_integral_z,qg_mask_cumulative_magnitude_integral,"
          << "gas_boundary_flux_cumulative,gas_cumulative_error_raw,"
          << "gas_cumulative_error_postprocess_corrected,mass_boundary_flux_cumulative,"
          << "mass_cumulative_error_raw,mass_cumulative_error_postprocess_corrected\n";
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
      gasCumulativeError - cumulativeAlphaClipDeltaVolume - cumulativeAlphaFilterDeltaVolume;
  const dfloat massCumulativeError = state.totalMass
                                     - validationInitialTotalMass
                                     + cumulativeMassBoundaryIntegral;
  const dfloat massClipCorrection =
      (p.rhoGas - p.rhoLiquid) * (cumulativeAlphaClipDeltaVolume + cumulativeAlphaFilterDeltaVolume);
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
                   << alphaFilterDeltaVolume << ','
                   << cumulativeAlphaFilterDeltaVolume << ','
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
  const double inertiaRatio = (p.rhoGas + p.virtualMassEnabled * virtualMassRampFactor(nrs->tstep)
      * p.virtualMassCoefficient * p.rhoLiquid) / p.rhoGas;
  dlong indices[2] = {0, 0};
  double maxima[2] = {-1.0, -1.0};
  for (dlong n = 0; n < N; ++n) {
    const double ac = std::max(0.0, std::min(1.0, double(a[n])));
    double force2 = 0.0;
    for (int c = 0; c < 3; ++c) {
      const double f = (p.mixtureDragEnabled != 0.0 ? 1.0 : 0.0)*ac*(1.0-p.rhoGas/p.rhoLiquid)*rate[n]*inertiaRatio
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
      data[22] = (p.mixtureDragEnabled != 0.0 ? 1.0 : 0.0)*ac*(1.0-p.rhoGas/p.rhoLiquid)*rate[n]*inertiaRatio;
      data[23] = data[22]*data[19];
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

  // Refresh gas-dependent fields with the completed-step solutions.
  prepareGasAdvection();
  evaluatePointwiseTerms();

  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;
  const MPI_Comm comm = platform->comm.mpiComm();

  const dfloat maxDiv =
      platform->linAlg->amax(Nlocal, nrs->fluid->o_div, comm);

  vectorMagnitudeKernel(Nlocal, 3, offset, o_gradP, o_monitorMagnitude);
  buildGasPressureAccelerationMonitorKernel(Nlocal,
                                            p.rhoGas + p.virtualMassEnabled * virtualMassRampFactor(nrs->tstep)
                                                       * p.virtualMassCoefficient * p.rhoLiquid,
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
           "maxActive|gradP|/rhoEff=%.8e maxInactive|gradP|/rhoEff=%.8e "
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
