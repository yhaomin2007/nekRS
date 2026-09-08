#pragma once

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
  dfloat initialPlumeHeight;
  dfloat initialPlumeThickness;
  dfloat gravity[3];
  dfloat alphaFloor;
  dfloat dragEnabled;
  dfloat bubbleDiameter;
  dfloat virtualMassEnabled;
  dfloat virtualMassCoefficient;
  dfloat subtractAlphaDiffusion;
  dfloat subtractUgDiffusion[3];
};

static Parameters p;
static deviceMemory<dfloat> o_ug;
static deviceMemory<dfloat> o_ul;
static deviceMemory<dfloat> o_ulPrevious;
static deviceMemory<dfloat> o_ugPrevious;
static deviceMemory<dfloat> o_gradUl;
static deviceMemory<dfloat> o_gradUv;
static deviceMemory<dfloat> o_virtualMassRelativeAcceleration;
static deviceMemory<dfloat> o_gradAlpha;
static deviceMemory<dfloat> o_gradUg;
static deviceMemory<dfloat> o_gradP;
static deviceMemory<dfloat> o_rhoM;
static deviceMemory<dfloat> o_muM;
static deviceMemory<dfloat> o_driftStress;
static deviceMemory<dfloat> o_divDriftStress;
static deviceMemory<dfloat> o_exactKinematicStress;
static deviceMemory<dfloat> o_nativeDynamicStress;
static deviceMemory<dfloat> o_divExactKinematicStress;
static deviceMemory<dfloat> o_divNativeDynamicStress;
static deviceMemory<dfloat> o_alphaDiffusionFlux;
static deviceMemory<dfloat> o_ugDiffusionFlux;
static deviceMemory<dfloat> o_alphaDiffusionDivergence;
static deviceMemory<dfloat> o_ugDiffusionDivergence;
static deviceMemory<dfloat> o_mixtureInterphaseAcceleration;
static deviceMemory<dfloat> o_alphaSource;
static deviceMemory<dfloat> o_ugSource;
static deviceMemory<dfloat> o_dragLambda;
static deviceMemory<dfloat> o_mixtureForce;
static occa::kernel packGasVelocityKernel;
static occa::kernel initializePlumeKernel;
static occa::kernel buildLiquidVelocityKernel;
static occa::kernel updateVirtualMassHistoryKernel;
static occa::kernel buildEquationTermsKernel;
static occa::kernel buildMixtureForceKernel;

inline void registerKernels(deviceKernelProperties &kernelInfo)
{
  const std::string request = "bubbleColumn2::equations";
  // This is standalone OKL source.  The .okl suffix is required so OCCA
  // translates @kernel/@globalPtr before invoking the HIP compiler.
  const std::string fileName = "bubbleColumn2Equations.okl";
  if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    platform->kernelRequests.add(request, fileName, kernelInfo);
  } else {
    packGasVelocityKernel = platform->kernelRequests.load(request, "packGasVelocity");
    initializePlumeKernel = platform->kernelRequests.load(request, "initializePlume");
    buildLiquidVelocityKernel = platform->kernelRequests.load(request, "buildLiquidVelocity");
    updateVirtualMassHistoryKernel =
        platform->kernelRequests.load(request, "updateVirtualMassHistory");
    buildEquationTermsKernel = platform->kernelRequests.load(request, "buildEquationTerms");
    buildMixtureForceKernel = platform->kernelRequests.load(request, "buildMixtureForce");
  }
}

inline void allocate()
{
  const dlong offset = nrs->fieldOffset;
  o_ug.resize(3 * offset);
  o_ul.resize(3 * offset);
  o_ulPrevious.resize(3 * offset);
  o_ugPrevious.resize(3 * offset);
  o_gradUl.resize(9 * offset);
  o_gradUv.resize(9 * offset);
  o_virtualMassRelativeAcceleration.resize(3 * offset);
  o_gradAlpha.resize(3 * offset);
  o_gradUg.resize(9 * offset);
  o_gradP.resize(3 * offset);
  o_rhoM.resize(offset);
  o_muM.resize(offset);
  o_driftStress.resize(9 * offset);
  o_divDriftStress.resize(3 * offset);
  o_exactKinematicStress.resize(9 * offset);
  o_nativeDynamicStress.resize(9 * offset);
  o_divExactKinematicStress.resize(3 * offset);
  o_divNativeDynamicStress.resize(3 * offset);
  o_alphaDiffusionFlux.resize(3 * offset);
  o_ugDiffusionFlux.resize(9 * offset);
  o_alphaDiffusionDivergence.resize(offset);
  o_ugDiffusionDivergence.resize(3 * offset);
  o_mixtureInterphaseAcceleration.resize(3 * offset);
  o_alphaSource.resize(offset);
  o_ugSource.resize(3 * offset);
  o_dragLambda.resize(offset);
  o_mixtureForce.resize(3 * offset);
}

inline void evaluatePointwiseTerms()
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  auto alpha = nrs->scalar->o_solution("alpha");

  packGasVelocityKernel(mesh->Nlocal,
                        offset,
                        nrs->scalar->o_solution("ugx"),
                        nrs->scalar->o_solution("ugy"),
                        nrs->scalar->o_solution("ugz"),
                        o_ug);
  buildLiquidVelocityKernel(mesh->Nlocal,
                            offset,
                            p.alphaFloor,
                            alpha,
                            nrs->fluid->o_U,
                            o_ug,
                            o_ul);
  opSEM::strongGrad(mesh, offset, alpha, o_gradAlpha);
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg);
  opSEM::strongGradVec(mesh, offset, o_ul, o_gradUl);
  opSEM::strongGradVec(mesh, offset, nrs->fluid->o_U, o_gradUv);
  opSEM::strongGrad(mesh, offset, nrs->fluid->o_P, o_gradP);

  buildEquationTermsKernel(mesh->Nlocal,
                           offset,
                           p.rhoLiquid,
                           p.rhoGas,
                           p.muLiquid,
                           p.muGas,
                           p.alphaFloor,
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
                           o_gradAlpha,
                           o_gradUg,
                           o_gradUl,
                           o_gradUv,
                           o_virtualMassRelativeAcceleration,
                           o_gradP,
                           o_rhoM,
                           o_muM,
                           o_driftStress,
                           o_alphaSource,
                           o_ugSource,
                           o_dragLambda,
                           o_mixtureInterphaseAcceleration,
                           o_exactKinematicStress,
                           o_nativeDynamicStress);
}

inline void initializeHistory()
{
  evaluatePointwiseTerms();
  const dlong offset = nrs->fieldOffset;
  o_ulPrevious.copyFrom(o_ul, 3 * offset);
  o_ugPrevious.copyFrom(o_ug, 3 * offset);
  platform->linAlg->fill(3 * offset, 0.0, o_virtualMassRelativeAcceleration);
}

inline void updateVirtualMassHistory()
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  auto alpha = nrs->scalar->o_solution("alpha");

  packGasVelocityKernel(mesh->Nlocal,
                        offset,
                        nrs->scalar->o_solution("ugx"),
                        nrs->scalar->o_solution("ugy"),
                        nrs->scalar->o_solution("ugz"),
                        o_ug);
  buildLiquidVelocityKernel(mesh->Nlocal,
                            offset,
                            p.alphaFloor,
                            alpha,
                            nrs->fluid->o_U,
                            o_ug,
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

    auto exactRow = o_exactKinematicStress.slice(3 * i * offset, 3 * offset);
    auto divExactRow = o_divExactKinematicStress.slice(i * offset, offset);
    opSEM::strongDivergence(mesh, offset, exactRow, divExactRow);

    auto nativeRow = o_nativeDynamicStress.slice(3 * i * offset, 3 * offset);
    auto divNativeRow = o_divNativeDynamicStress.slice(i * offset, offset);
    opSEM::strongDivergence(mesh, offset, nativeRow, divNativeRow);
  }
  buildMixtureForceKernel(mesh->Nlocal,
                          offset,
                          p.gravity[0],
                          p.gravity[1],
                          p.gravity[2],
                          o_rhoM,
                          o_divDriftStress,
                          o_divExactKinematicStress,
                          o_divNativeDynamicStress,
                          o_mixtureInterphaseAcceleration,
                          o_mixtureForce);
}

inline void subtractScalarDiffusion()
{
  const dlong Nlocal = nrs->meshV->Nlocal;
  const dlong offset = nrs->fieldOffset;

  if (p.subtractAlphaDiffusion != 0.0) {
    o_alphaDiffusionFlux.copyFrom(o_gradAlpha, 3 * offset);
    auto diffusion = nrs->scalar->o_diffusionCoeff("alpha");
    platform->linAlg->axmyVector(
        Nlocal, offset, 0, 1.0, diffusion, o_alphaDiffusionFlux);
    opSEM::strongDivergence(
        nrs->meshV, offset, o_alphaDiffusionFlux, o_alphaDiffusionDivergence);
    platform->linAlg->axpby(Nlocal,
                            -p.subtractAlphaDiffusion,
                            o_alphaDiffusionDivergence,
                            1.0,
                            o_alphaSource);
  }

  const char *gasNames[3] = {"ugx", "ugy", "ugz"};
  for (int i = 0; i < 3; ++i) {
    if (p.subtractUgDiffusion[i] == 0.0) {
      continue;
    }
    auto grad = o_gradUg.slice(3 * i * offset, 3 * offset);
    auto flux = o_ugDiffusionFlux.slice(3 * i * offset, 3 * offset);
    flux.copyFrom(grad, 3 * offset);
    auto diffusion = nrs->scalar->o_diffusionCoeff(gasNames[i]);
    platform->linAlg->axmyVector(Nlocal, offset, 0, 1.0, diffusion, flux);
    auto divergence = o_ugDiffusionDivergence.slice(i * offset, offset);
    opSEM::strongDivergence(nrs->meshV, offset, flux, divergence);
    platform->linAlg->axpby(Nlocal,
                            -p.subtractUgDiffusion[i],
                            divergence,
                            1.0,
                            o_ugSource,
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
  nrs->scalar->o_explicitTerms("ugx").copyFrom(o_ugSource, Nlocal, 0, 0 * offset);
  nrs->scalar->o_explicitTerms("ugy").copyFrom(o_ugSource, Nlocal, 0, 1 * offset);
  nrs->scalar->o_explicitTerms("ugz").copyFrom(o_ugSource, Nlocal, 0, 2 * offset);

  auto fluidTerms = nrs->fluid->o_explicitTerms();
  for (int i = 0; i < 3; ++i) {
    fluidTerms.copyFrom(o_mixtureForce, Nlocal, i * offset, i * offset);
  }
}

inline void updateProperties(double)
{
  // Called after all four scalars advance: refresh pressure mobility and sources.
  evaluatePointwiseTerms();
  nrs->fluid->o_prop.slice(0 * nrs->fieldOffset, nrs->fieldOffset).copyFrom(o_muM);
  nrs->fluid->o_prop.slice(1 * nrs->fieldOffset, nrs->fieldOffset).copyFrom(o_rhoM);
}

inline occa::memory implicitGasDrag(double, int scalarIndex)
{
  // Scalar ordering is ALPHA, UGX, UGY, UGZ.
  if (p.dragEnabled != 0.0 && scalarIndex >= 1 && scalarIndex <= 3) {
    return o_dragLambda;
  }
  return o_NULL;
}

inline void enforceZeroDivergence(double)
{
  // nrs_t::evaluateDivergence() zero-fills fluid->o_div before invoking this
  // callback. Leaving it unchanged enforces div(uv)=0 exactly.
}

} // namespace bubbleColumn2
