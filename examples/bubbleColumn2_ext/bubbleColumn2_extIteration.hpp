#pragma once

namespace bubbleColumn2_ext {
static double coupledStepStartTime = 0.0;
static int couplingMinIterations = 2, couplingMaxIterations = 20;
static dfloat couplingRelativeTolerance = 1e-5, couplingAbsoluteTolerance = 1e-8;
static dfloat couplingDivergenceTolerance = 1e-4;
static dfloat couplingRelaxation = 0.7, couplingRequireConvergence = 1.0;
static deviceMemory<dfloat> o_vmStepUg, o_vmStepUl;
static deviceMemory<dfloat> o_stepU, o_stepS, o_trialU, o_trialS;
static deviceMemory<dfloat> o_iterU, o_iterS, o_iterP, o_iterDelta, o_iterDiv;
static dfloat clipCumulativeAtStart, maskCumulativeAtStart[3], maskMagnitudeAtStart;

inline void readIterationParameters()
{
  auto read = [](const char *key, auto &value) {
    platform->par->extract("casedata", key, value);
  };
  read("couplingminiterations", couplingMinIterations);
  read("couplingmaxiterations", couplingMaxIterations);
  read("couplingrelativetolerance", couplingRelativeTolerance);
  read("couplingabsolutetolerance", couplingAbsoluteTolerance);
  read("couplingdivergencetolerance", couplingDivergenceTolerance);
  read("couplingrelaxation", couplingRelaxation);
  read("couplingrequireconvergence", couplingRequireConvergence);
  nekrsCheck(couplingMinIterations < 1 || couplingMaxIterations < couplingMinIterations
             || !std::isfinite(couplingRelativeTolerance) || couplingRelativeTolerance < 0
             || !std::isfinite(couplingAbsoluteTolerance) || couplingAbsoluteTolerance <= 0
             || !std::isfinite(couplingDivergenceTolerance) || couplingDivergenceTolerance <= 0
             || !std::isfinite(couplingRelaxation) || couplingRelaxation <= 0 || couplingRelaxation > 1,
             platform->comm.mpiComm(), EXIT_FAILURE, "%s", "Invalid coupling iteration parameters\n");
}

inline void restoreIterationAccounting()
{
  cumulativeAlphaClipDeltaVolume = clipCumulativeAtStart;
  cumulativeQgMaskDeltaMagnitudeIntegral = maskMagnitudeAtStart;
  for (int i = 0; i < 3; ++i) cumulativeQgMaskDeltaIntegral[i] = maskCumulativeAtStart[i];
}

inline void setCurrentRhsWeights()
{
  // BDF order remains unchanged. Nonlinear terms are evaluated at the iterate,
  // rather than extrapolated from physical-time source histories.
  for (auto memory : {nrs->o_coeffEXT, nrs->fluid->o_coeffEXT, nrs->scalar->o_coeffEXT}) {
    std::vector<dfloat> weights(memory.size(), 0.0);
    weights[0] = 1.0;
    memory.copyFrom(weights.data());
  }
}

inline void beginCoupledStep(double time)
{
  nekrsCheck(nrs->geom || nrs->neknek || nrs->scalar->Nsubsteps != 0
             || nrs->advectionSubcycingSteps != 0,
             platform->comm.mpiComm(), EXIT_FAILURE, "%s",
             "bubbleColumn2_ext requires fixed mesh, no NekNek and no advection subcycling\n");
  coupledStepStartTime = time;
  // Called once by initInnerStep, before lagSolution and native makeForcing.
  // These immutable copies contain u^n,u^{n-1},... used by the BDF RHS.
  o_stepU.resize(nrs->fluid->o_U.size());
  o_stepS.resize(nrs->scalar->o_S.size());
  o_trialU.resize(o_stepU.size()); o_trialS.resize(o_stepS.size());
  o_stepU.copyFrom(nrs->fluid->o_U); o_stepS.copyFrom(nrs->scalar->o_S);
  o_iterU.resize(nrs->fluid->fieldOffsetSum);
  o_iterS.resize(nrs->scalar->fieldOffsetSum);
  o_iterP.resize(nrs->fieldOffset);
  o_iterDelta.resize(nrs->fieldOffset); o_iterDiv.resize(nrs->fieldOffset);
  o_iterU.copyFrom(nrs->fluid->o_U, o_iterU.size());
  o_iterS.copyFrom(nrs->scalar->o_S, o_iterS.size());
  o_iterP.copyFrom(nrs->fluid->o_P, o_iterP.size());
  o_vmStepUg.resize(3*nrs->fieldOffset); o_vmStepUl.resize(3*nrs->fieldOffset);
  o_vmStepUg.copyFrom(o_ugPrevious); o_vmStepUl.copyFrom(o_ulPrevious);
  clipCumulativeAtStart = cumulativeAlphaClipDeltaVolume;
  maskMagnitudeAtStart = cumulativeQgMaskDeltaMagnitudeIntegral;
  for (int i = 0; i < 3; ++i) maskCumulativeAtStart[i] = cumulativeQgMaskDeltaIntegral[i];
  setCurrentRhsWeights();
  addExplicitSources(time);
}

inline void refreshCoupledRhs(double time)
{
  // Do not call initStep, lagSolution or setTimeIntegrationCoeffs here.
  setCurrentRhsWeights();
  nrs->fluid->o_Ue.copyFrom(nrs->fluid->o_U, nrs->fluid->fieldOffsetSum);
  nrs->scalar->o_Se.copyFrom(nrs->scalar->o_S, nrs->scalar->fieldOffsetSum);
  nrs->computeUrst();
  platform->linAlg->fill(nrs->fluid->fieldOffsetSum, 0.0, nrs->fluid->o_EXT);
  platform->linAlg->fill(nrs->scalar->fieldOffsetSum, 0.0, nrs->scalar->o_EXT);
  if (p.virtualMassEnabled != 0.0) {
    o_ugPrevious.copyFrom(o_vmStepUg); o_ulPrevious.copyFrom(o_vmStepUl);
    updateVirtualMassHistory();
    // This trial must not advance accepted phase-velocity history.
    o_ugPrevious.copyFrom(o_vmStepUg); o_ulPrevious.copyFrom(o_vmStepUl);
  }
  addExplicitSources(time);
  for (int is = 0; is < 4; ++is) {
    nrs->scalar->makeAdvection(is, time, nrs->tstep);
    nrs->scalar->makeExplicit(is, time, nrs->tstep);
  }
  nrs->fluid->makeAdvection(time, nrs->tstep);
  nrs->fluid->makeExplicit(time, nrs->tstep);
  // makeForcing reads slot zero as the previous physical-time solution.
  // Temporarily supply the immutable pre-lag BDF history, then restore all
  // current/lagged iterate slots before another solve or initial-guess update.
  o_trialU.copyFrom(nrs->fluid->o_U); o_trialS.copyFrom(nrs->scalar->o_S);
  nrs->fluid->o_U.copyFrom(o_stepU);
  nrs->scalar->o_S.copyFrom(o_stepS);
  nrs->scalar->makeForcing();
  nrs->fluid->makeForcing();
  nrs->fluid->o_U.copyFrom(o_trialU);
  nrs->scalar->o_S.copyFrom(o_trialS);
}

inline bool couplingConverged(int iteration)
{
  const auto N = nrs->meshV->Nlocal;
  const auto offset = nrs->fieldOffset;
  const auto comm = platform->comm.mpiComm();
  // Keep projections/limiters active, but discard diagnostic accumulation from
  // rejected trial stages. Physical-time history updates occur only at commit.
  clipVolumeVelocity();
  dfloat errors[4] = {0,0,0,0};
  auto check = [&](occa::memory current, const occa::memory &previous, int group) {
    // Explicit axpbyz is required: previous and current are distinct fields.
    platform->linAlg->axpbyz(N, couplingRelaxation, current,
                           1.0-couplingRelaxation, previous, current);
    platform->linAlg->axpbyz(N, 1.0, current, -1.0, previous, o_iterDelta);
    const dfloat change = platform->linAlg->amax(N, o_iterDelta, comm);
    const dfloat magnitude = platform->linAlg->amax(N, current, comm);
    nekrsCheck(!std::isfinite(change) || !std::isfinite(magnitude), comm, EXIT_FAILURE,
               "%s", "Nonfinite coupling iterate\n");
    errors[group] = std::max(errors[group], change /
        (couplingAbsoluteTolerance + couplingRelativeTolerance*magnitude));
  };
  auto alpha = nrs->scalar->o_solution("alpha");
  check(alpha, o_iterS.slice(nrs->scalar->fieldOffsetScan[0], N), 0);
  for (int c = 0; c < 3; ++c) {
    check(nrs->scalar->o_solution(c == 0 ? "qgx" : c == 1 ? "qgy" : "qgz"),
          o_iterS.slice(nrs->scalar->fieldOffsetScan[c+1], N), 1);
    check(nrs->fluid->o_U.slice(c*offset, N), o_iterU.slice(c*offset, N), 2);
  }
  check(nrs->fluid->o_P.slice(0, N), o_iterP.slice(0, N), 3);
  // Measure the raw solve increment (relaxation must not hide nonconvergence).
  for (auto &error : errors) error /= couplingRelaxation;
  opSEM::strongDivergence(nrs->meshV, offset, nrs->fluid->o_U, o_iterDiv);
  const dfloat divRms = platform->linAlg->weightedNorm2(
      N, nrs->meshV->o_LMM, o_iterDiv, comm) / std::sqrt(nrs->meshV->volume);
  const bool converged = iteration >= couplingMinIterations
      && errors[0] <= 1 && errors[1] <= 1 && errors[2] <= 1 && errors[3] <= 1
      && std::isfinite(divRms) && divRms <= couplingDivergenceTolerance;
  if (platform->comm.mpiRank() == 0) {
    printf("bubbleColumn2_ext coupling step=%d iter=%d scaledDelta(alpha,qg,um,p)="
           "(%.8e,%.8e,%.8e,%.8e) divRMS=%.8e converged=%d\n",
           nrs->tstep, iteration, errors[0],errors[1],errors[2],errors[3],divRms,int(converged));
  }
  const bool capped = iteration >= couplingMaxIterations;
  nekrsCheck(capped && !converged && couplingRequireConvergence != 0.0,
             comm, EXIT_FAILURE, "Coupling failed at step %d after %d iterations\n",
             nrs->tstep, iteration);
  if (capped && !converged && platform->comm.mpiRank() == 0)
    printf("bubbleColumn2_ext WARNING: accepting unconverged step %d by user option\n", nrs->tstep);
  if (converged || capped) return true;
  restoreIterationAccounting();
  o_iterU.copyFrom(nrs->fluid->o_U, o_iterU.size());
  o_iterS.copyFrom(nrs->scalar->o_S, o_iterS.size());
  o_iterP.copyFrom(nrs->fluid->o_P, o_iterP.size());
  refreshCoupledRhs(coupledStepStartTime + nrs->dt[0]);
  return false;
}
} // namespace bubbleColumn2_ext
