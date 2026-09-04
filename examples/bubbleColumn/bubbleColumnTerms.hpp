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
  dfloat dragCoefficient;
};

static Parameters p;
static deviceMemory<dfloat> o_ug;
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

inline void allocate()
{
  const dlong offset = nrs->fieldOffset;
  o_ug.resize(3 * offset);
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

  packGasVelocity(mesh->Nlocal,
                  offset,
                  nrs->scalar->o_solution("ugx"),
                  nrs->scalar->o_solution("ugy"),
                  nrs->scalar->o_solution("ugz"),
                  o_ug);
  opSEM::strongGrad(mesh, offset, alpha, o_gradAlpha);
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg);
  opSEM::strongGrad(mesh, offset, nrs->fluid->o_P, o_gradP);

  buildEquationTerms(mesh->Nlocal,
                     offset,
                     p.rhoLiquid,
                     p.rhoGas,
                     p.muLiquid,
                     p.muGas,
                     p.alphaFloor,
                     p.dragCoefficient,
                     p.gravity[0],
                     p.gravity[1],
                     p.gravity[2],
                     alpha,
                     nrs->fluid->o_U,
                     o_ug,
                     o_gradAlpha,
                     o_gradUg,
                     o_gradP,
                     o_rhoM,
                     o_muM,
                     o_divSource,
                     o_driftStress,
                     o_alphaSource,
                     o_ugSource);
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
  buildMixtureForce(mesh->Nlocal,
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
  nrs->scalar->o_explicitTerms("alpha").copyFrom(o_alphaSource);
  nrs->scalar->o_explicitTerms("ugx").copyFrom(
      o_ugSource.slice(0 * nrs->fieldOffset, nrs->fieldOffset));
  nrs->scalar->o_explicitTerms("ugy").copyFrom(
      o_ugSource.slice(1 * nrs->fieldOffset, nrs->fieldOffset));
  nrs->scalar->o_explicitTerms("ugz").copyFrom(
      o_ugSource.slice(2 * nrs->fieldOffset, nrs->fieldOffset));
  nrs->fluid->o_explicitTerms().copyFrom(o_mixtureForce);
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
