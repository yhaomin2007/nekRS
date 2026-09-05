#pragma once

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
  dfloat gravity[3];
  dfloat alphaFloor;
  dfloat dragEnabled;
  dfloat bubbleDiameter;
  dfloat virtualMassEnabled;
  dfloat virtualMassCoefficient;
};

static Parameters p;
static deviceMemory<dfloat> o_ug;
static deviceMemory<dfloat> o_ul;
static deviceMemory<dfloat> o_ulPrevious;
static deviceMemory<dfloat> o_ugPrevious;
static deviceMemory<dfloat> o_gradUl;
static deviceMemory<dfloat> o_virtualMassRelativeAcceleration;
static deviceMemory<dfloat> o_gradAlpha;
static deviceMemory<dfloat> o_gradUg;
static deviceMemory<dfloat> o_gradP;
static deviceMemory<dfloat> o_rhoM;
static deviceMemory<dfloat> o_muM;
static deviceMemory<dfloat> o_divSource;
static deviceMemory<dfloat> o_driftStress;
static deviceMemory<dfloat> o_divDriftStress;
static deviceMemory<dfloat> o_alphaSource;
static deviceMemory<dfloat> o_ugSource;
static deviceMemory<dfloat> o_mixtureForce;
static occa::kernel packGasVelocityKernel;
static occa::kernel buildLiquidVelocityKernel;
static occa::kernel updateVirtualMassHistoryKernel;
static occa::kernel buildEquationTermsKernel;
static occa::kernel buildMixtureForceKernel;

inline void registerKernels(deviceKernelProperties &kernelInfo)
{
  const std::string request = "bubbleColumn::equations";
  // This is standalone OKL source.  The .okl suffix is required so OCCA
  // translates @kernel/@globalPtr before invoking the HIP compiler.
  const std::string fileName = "bubbleColumnEquations.okl";
  if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    platform->kernelRequests.add(request, fileName, kernelInfo);
  } else {
    packGasVelocityKernel = platform->kernelRequests.load(request, "packGasVelocity");
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
  o_virtualMassRelativeAcceleration.resize(3 * offset);
  o_gradAlpha.resize(3 * offset);
  o_gradUg.resize(9 * offset);
  o_gradP.resize(3 * offset);
  o_rhoM.resize(offset);
  o_muM.resize(offset);
  o_divSource.resize(offset);
  o_driftStress.resize(9 * offset);
  o_divDriftStress.resize(3 * offset);
  o_alphaSource.resize(offset);
  o_ugSource.resize(3 * offset);
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
                            p.rhoLiquid,
                            p.rhoGas,
                            p.alphaFloor,
                            alpha,
                            nrs->fluid->o_U,
                            o_ug,
                            o_ul);
  opSEM::strongGrad(mesh, offset, alpha, o_gradAlpha);
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg);
  opSEM::strongGradVec(mesh, offset, o_ul, o_gradUl);
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
                           o_virtualMassRelativeAcceleration,
                           o_gradP,
                           o_rhoM,
                           o_muM,
                           o_divSource,
                           o_driftStress,
                           o_alphaSource,
                           o_ugSource);
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
                            p.rhoLiquid,
                            p.rhoGas,
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
  }
  buildMixtureForceKernel(mesh->Nlocal,
                          offset,
                          p.gravity[0],
                          p.gravity[1],
                          p.gravity[2],
                          o_rhoM,
                          o_divDriftStress,
                          o_mixtureForce);
}

inline void addExplicitSources(double)
{
  evaluatePointwiseTerms();
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
  // Called after all four scalars advance: refresh Eqs. (15) and (20).
  evaluatePointwiseTerms();
  nrs->fluid->o_prop.slice(0 * nrs->fieldOffset, nrs->fieldOffset).copyFrom(o_muM);
  nrs->fluid->o_prop.slice(1 * nrs->fieldOffset, nrs->fieldOffset).copyFrom(o_rhoM);
  auto o_diffusion = nrs->scalar->o_diffusionCoeff();
  auto o_transport = nrs->scalar->o_transportCoeff();
  platform->linAlg->fill(nrs->scalar->fieldOffsetSum, 1e-12, o_diffusion);
  platform->linAlg->fill(nrs->scalar->fieldOffsetSum, 1.0, o_transport);
}

inline void updateDivergence(double)
{
  nrs->fluid->o_div.copyFrom(o_divSource);
}
} // namespace bubbleColumn
