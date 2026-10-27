#pragma once

// Included inside frozenInner after the snapshot and norm helpers.
// Use native SEM/cubature operators, with UDF-owned outputs. Never write ADV.
static int updateConvection = 0;
static dfloat picardTolerance = 1e-8;
static deviceMemory<dfloat> advU, advS, advector, extrapAdvU, extrapAdvS;
static deviceMemory<dfloat> baseU, baseS, previousU, previousP, previousS;

inline void convection(const occa::memory &u, const occa::memory &s)
{
  auto mesh = nrs->meshV;
  auto &f = nrs->fluid;
  auto &sc = nrs->scalar;
  advU.resize(f->fieldOffsetSum);
  advS.resize(sc->fieldOffsetSum);
  advector.resize(f->o_relUrst.size());
  if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
    launchKernel("nrs-UrstCubatureHex3D", mesh->Nelements, 0,
                 mesh->o_cubvgeo, mesh->o_cubInterpT, f->fieldOffset, 0,
                 f->cubatureOffset, u, u, advector);
    launchKernel("core-strongAdvectionCubatureVolumeHex3D", mesh->Nelements,
                 mesh->o_vgeo, mesh->o_cubDiffInterpT, mesh->o_cubInterpT,
                 mesh->o_cubProjectT, f->fieldOffset, f->cubatureOffset,
                 u, advector, advU);
    launchKernel("core-strongAdvectionCubatureVolumeScalarHex3D", mesh->Nelements,
                 1, 0, 0, mesh->o_vgeo, mesh->o_cubDiffInterpT,
                 mesh->o_cubInterpT, mesh->o_cubProjectT, sc->o_compute,
                 sc->o_fieldOffsetScan, sc->vFieldOffset, sc->vCubatureOffset,
                 s, advector, sc->o_rho, advS);
  } else {
    launchKernel("nrs-UrstHex3D", mesh->Nelements, 0, mesh->o_vgeo,
                 f->fieldOffset, 0, u, u, advector);
    launchKernel("core-strongAdvectionVolumeHex3D", mesh->Nelements,
                 mesh->o_vgeo, mesh->o_D, f->fieldOffset, u, advector, advU);
    launchKernel("core-strongAdvectionVolumeScalarHex3D", mesh->Nelements,
                 1, 0, mesh->o_vgeo, mesh->o_D, sc->o_compute,
                 sc->o_fieldOffsetScan, sc->vFieldOffset, s, advector,
                 sc->o_rho, advS);
  }
}

inline void addMassWeighted(const occa::memory &destination,
                            const occa::memory &advection, int components,
                            dlong offset, dfloat factor)
{
  auto mesh = nrs->meshV;
  scratch.resize(mesh->Nlocal);
  for (int c = 0; c < components; ++c) {
    scratch.copyFrom(advection.slice(c * offset, mesh->Nlocal), mesh->Nlocal);
    platform->linAlg->axmy(mesh->Nlocal, 1.0, (components == 3 ? mesh->o_Jw : mesh->o_LMM), scratch);
    auto output = destination.slice(c * offset, mesh->Nlocal);
    platform->linAlg->axpby(mesh->Nlocal, factor, scratch, 1.0, output);
  }
}

inline void prepareConvectionHistory()
{
  // On a fixed mesh with constant density and native advection, each native
  // ADV history entry is N(U^n-j,S^n-j). Reconstruct it from pre-lag solution
  // slots, including restart slots; do not shift or access private ADV arrays.
  copy(extrapAdvU, nrs->fluid->o_JwF, nrs->fluid->fieldOffsetSum);
  copy(extrapAdvS, nrs->scalar->o_JwF, nrs->scalar->fieldOffsetSum);
  platform->linAlg->fill(extrapAdvU.size(), 0.0, extrapAdvU);
  platform->linAlg->fill(extrapAdvS.size(), 0.0, extrapAdvS);
  std::vector<dfloat> cu(nrs->fluid->o_coeffEXT.size());
  std::vector<dfloat> cs(nrs->scalar->o_coeffEXT.size());
  nrs->fluid->o_coeffEXT.copyTo(cu.data());
  nrs->scalar->o_coeffEXT.copyTo(cs.data());
  nekrsCheck(cu.size() != cs.size(), platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Picard convection requires matching velocity/scalar EXT orders.\n");
  for (size_t j = 0; j < cu.size(); ++j) {
    if (cu[j] == 0 && cs[j] == 0) continue;
    convection(startU.slice(j * nrs->fluid->fieldOffsetSum,
                            nrs->fluid->fieldOffsetSum),
               startS.slice(j * nrs->scalar->fieldOffsetSum,
                            nrs->scalar->fieldOffsetSum));
    addMassWeighted(extrapAdvU, advU, 3, nrs->fluid->fieldOffset, cu[j]);
    addMassWeighted(extrapAdvS, advS, 1, nrs->scalar->fieldOffsetSum, cs[j]);
  }
}

inline void savePicardBase()
{
  // Native RHS = BDF history + EXT(non-advection sources) - M EXT(N).
  copy(baseU, rhsU.values, rhsU.values.size());
  copy(baseS, rhsS.values, rhsS.values.size());
  platform->linAlg->axpby(baseU.size(), 1.0, extrapAdvU, 1.0, baseU);
  platform->linAlg->axpby(baseS.size(), 1.0, extrapAdvS, 1.0, baseS);
  copy(previousU, firstU, firstU.size());
  copy(previousP, firstP, firstP.size());
  copy(previousS, firstS, firstS.size());
}

inline bool picardCheck(int iteration, dfloat historyError, dfloat frozenError)
{
  const auto u = difference(nrs->fluid->o_U, previousU, 3, nrs->fluid->fieldOffset);
  const auto p = difference(nrs->fluid->o_P, previousP, 1, nrs->fluid->fieldOffset);
  const auto s = difference(nrs->scalar->o_S, previousS, 1, nrs->scalar->fieldOffsetSum);
  const dfloat change = std::max(u.relativeMax, std::max(p.relativeMax, s.relativeMax));
  const bool converged = iteration > 1 && change <= picardTolerance;
  if (platform->comm.mpiRank() == 0) {
    printf("picardInner step=%d pass=%d/%d U[max,rms]=[%.8e,%.8e] "
           "P[max,rms]=[%.8e,%.8e] S[max,rms]=[%.8e,%.8e] "
           "relativeMax=%.8e historyMax=%.8e frozenMax=%.8e status=%s\n",
           step, iteration, iterations, u.maxAbs, u.rms, p.maxAbs, p.rms,
           s.maxAbs, s.rms, change, historyError, frozenError,
           iteration == 1 ? "predictor" : converged ? "converged" :
           iteration >= iterations ? "maxPasses" : "iterating");
  }
  nekrsCheck(historyError != 0 || frozenError != 0 || !std::isfinite(change),
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Picard immutable state changed or nonfinite iterate.\n");
  copy(previousU, nrs->fluid->o_U, nrs->fluid->fieldOffsetSum);
  copy(previousP, nrs->fluid->o_P, nrs->fluid->fieldOffset);
  copy(previousS, nrs->scalar->o_S, nrs->scalar->fieldOffsetSum);
  return converged || iteration >= iterations;
}

inline void updateConvectionRhs()
{
  // Evaluate the full nonlinear product from the last solved fields BEFORE
  // restoring the native pressure predictor/initial guesses for the next pass.
  convection(previousU, previousS);
  nrs->fluid->o_JwF.copyFrom(baseU, baseU.size());
  nrs->scalar->o_JwF.copyFrom(baseS, baseS.size());
  addMassWeighted(nrs->fluid->o_JwF, advU, 3, nrs->fluid->fieldOffset, -1.0);
  addMassWeighted(nrs->scalar->o_JwF, advS, 1, nrs->scalar->fieldOffsetSum, -1.0);
}
