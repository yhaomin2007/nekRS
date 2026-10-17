#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

// A frozen-system control experiment, not nonlinear Picard convergence.
// All allocations belong to the UDF. No private solver history is accessed.
namespace frozenInner
{
static int enabled = 1;
static int iterations = 3;
static dfloat tolerance = 1e-9;
static int step = -1, completed = 0, requestedCheckpoint = 0;

struct Saved {
  occa::memory live;
  deviceMemory<dfloat> values;
  void capture(const occa::memory &field)
  {
    live = field;
    values.resize(field.size());
    values.copyFrom(field, field.size());
  }
  void restore() { live.copyFrom(values, values.size()); }
};

static Saved historyU, historyS, pressure, predictorU, predictorS, urst;
static Saved boundary, rhsU, rhsS, explicitU, explicitS, propU, propS, propSBefore, divergence;
static Saved coeffBdf, coeffExtU, coeffExtS;
static deviceMemory<dfloat> startU, startS, firstU, firstP, firstS, scratch;

inline void copy(deviceMemory<dfloat> &destination, const occa::memory &source, dlong N)
{
  destination.resize(N);
  destination.copyFrom(source, N);
}

struct Error { dfloat maxAbs = 0, rms = 0, relativeMax = 0; };

inline Error difference(const occa::memory &field, const occa::memory &reference,
                        int components, dlong offset)
{
  auto mesh = nrs->meshV;
  auto comm = platform->comm.mpiComm();
  scratch.resize(mesh->Nlocal);
  Error error;
  dfloat squares = 0, scale = 1;
  for (int c = 0; c < components; ++c) {
    auto actual = field.slice(c * offset, mesh->Nlocal);
    auto expected = reference.slice(c * offset, mesh->Nlocal);
    scratch.copyFrom(actual, mesh->Nlocal);
    platform->linAlg->axpby(mesh->Nlocal, -1.0, expected, 1.0, scratch);
    const dfloat lo = platform->linAlg->min(mesh->Nlocal, scratch, comm);
    const dfloat hi = platform->linAlg->max(mesh->Nlocal, scratch, comm);
    error.maxAbs = std::max(error.maxAbs, std::max(std::abs(lo), std::abs(hi)));
    const dfloat refLo = platform->linAlg->min(mesh->Nlocal, expected, comm);
    const dfloat refHi = platform->linAlg->max(mesh->Nlocal, expected, comm);
    scale = std::max(scale, std::max(std::abs(refLo), std::abs(refHi)));
    platform->linAlg->axmy(mesh->Nlocal, 1.0, scratch, scratch);
    squares += platform->linAlg->innerProd(mesh->Nlocal, mesh->o_LMM, scratch, comm);
  }
  nekrsCheck(!std::isfinite(squares) || !std::isfinite(error.maxAbs),
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Frozen-inner comparison encountered a nonfinite field.\n");
  error.rms = std::sqrt(std::max(static_cast<dfloat>(0), squares)
                        / (components * mesh->volume));
  error.relativeMax = error.maxAbs / scale;
  return error;
}

inline dfloat unchanged(const Saved &field, dlong skip = 0)
{
  const dlong count = field.values.size() - skip;
  if (!count) return 0;
  scratch.resize(count);
  scratch.copyFrom(field.live, count, 0, skip);
  platform->linAlg->axpby(count, -1.0, field.values.slice(skip, count), 1.0, scratch);
  const auto comm = platform->comm.mpiComm();
  return std::max(std::abs(platform->linAlg->min(count, scratch, comm)),
                  std::abs(platform->linAlg->max(count, scratch, comm)));
}

// Called once by initInnerStep, after predictors/Urst but BEFORE makeForcing
// and BEFORE the native first-pass lagSolution. No forcing is rebuilt here.
inline void beginStep(double)
{
  nekrsCheck(step == nrs->tstep, platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Frozen-inner demo: userSource called twice for one timestep.\n");
  step = nrs->tstep;
  completed = 0;
  requestedCheckpoint = nrs->checkpointStep;
  copy(startU, nrs->fluid->o_U, nrs->fluid->o_U.size());
  copy(startS, nrs->scalar->o_S, nrs->scalar->o_S.size());
  pressure.capture(nrs->fluid->o_P);
  predictorU.capture(nrs->fluid->o_Ue);
  predictorS.capture(nrs->scalar->o_Se);
  urst.capture(nrs->fluid->o_relUrst);
  boundary.capture(platform->app->bc->o_usrwrk);
  propSBefore.capture(nrs->scalar->o_prop);
  coeffBdf.capture(nrs->o_coeffBDF);
  coeffExtU.capture(nrs->fluid->o_coeffEXT);
  coeffExtS.capture(nrs->scalar->o_coeffEXT);
}

// This callback runs after each scalar/pressure/velocity solve and before
// UDF_ExecuteStep. The first pass is the native one-pass reference for this dt.
inline bool convergenceCheck(int iteration)
{
  nekrsCheck(step != nrs->tstep || iteration != completed + 1,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Frozen-inner demo: unexpected timestep/corrector sequence.\n");
  completed = iteration;
  if (iteration == 1) {
    copy(firstU, nrs->fluid->o_U, nrs->fluid->fieldOffsetSum);
    copy(firstP, nrs->fluid->o_P, nrs->fluid->fieldOffset);
    copy(firstS, nrs->scalar->o_S, nrs->scalar->fieldOffsetSum);
    // Native lagSolution has occurred exactly once. Preserve ALL shifted
    // BDF/EXT solution slots, but restore the pre-solve current slot on replay.
    historyU.capture(nrs->fluid->o_U);
    historyS.capture(nrs->scalar->o_S);
    historyU.values.copyFrom(startU, nrs->fluid->fieldOffsetSum);
    historyS.values.copyFrom(startS, nrs->scalar->fieldOffsetSum);
    // Public RHS/explicit arrays include makeForcing's one-time history lag.
    rhsU.capture(nrs->fluid->o_JwF);
    rhsS.capture(nrs->scalar->o_JwF);
    explicitU.capture(nrs->fluid->o_EXT);
    explicitS.capture(nrs->scalar->o_EXT);
    propU.capture(nrs->fluid->o_prop);
    propS.capture(nrs->scalar->o_prop);
    divergence.capture(nrs->fluid->o_div);
  }
  const auto u = difference(nrs->fluid->o_U, firstU, 3, nrs->fluid->fieldOffset);
  const auto p = difference(nrs->fluid->o_P, firstP, 1, nrs->fluid->fieldOffset);
  const auto s = difference(nrs->scalar->o_S, firstS, 1, nrs->scalar->fieldOffsetSum);
  const dfloat historyError = std::max(
      unchanged(historyU, nrs->fluid->fieldOffsetSum),
      unchanged(historyS, nrs->scalar->fieldOffsetSum));
  const dfloat propertyError = unchanged(propS);
  dfloat frozenError = std::max(unchanged(rhsU), unchanged(rhsS));
  for (auto *field : {&explicitU, &explicitS, &coeffBdf, &coeffExtU, &coeffExtS,
                      &boundary, &urst, &propU, &divergence}) {
    frozenError = std::max(frozenError, unchanged(*field));
  }
  if (platform->comm.mpiRank() == 0) {
    printf("frozenInner step=%d pass=%d/%d U[max,rms]=[%.8e,%.8e] "
           "P[max,rms]=[%.8e,%.8e] S[max,rms]=[%.8e,%.8e] "
           "historyMax=%.8e frozenMax=%.8e propertyMax=%.8e\n", step, iteration, iterations,
           u.maxAbs, u.rms, p.maxAbs, p.rms, s.maxAbs, s.rms,
           historyError, frozenError, propertyError);
    static std::ofstream csv;
    if (!csv.is_open()) {
      std::ifstream previous("picard_repeatability.csv");
      const bool header = previous.peek() == std::ifstream::traits_type::eof();
      csv.open("picard_repeatability.csv", std::ios::app);
      if (header) csv << "step,pass,passes,u_max,u_rms,p_max,p_rms,s_max,s_rms,history_max,frozen_max,property_max\n";
      csv << std::scientific << std::setprecision(16);
    }
    csv << step << ',' << iteration << ',' << iterations << ',' << u.maxAbs << ',' << u.rms
        << ',' << p.maxAbs << ',' << p.rms << ',' << s.maxAbs << ',' << s.rms
        << ',' << historyError << ',' << frozenError << ',' << propertyError << '\n';
    csv.flush();
  }
  const dfloat error = std::max(u.relativeMax, std::max(p.relativeMax, s.relativeMax));
  nekrsCheck(!std::isfinite(error) || error > tolerance || historyError != 0 || frozenError != 0
               || !std::isfinite(propertyError) || propertyError > tolerance,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "Frozen-inner repeatability failed: scaled field error=%g history=%g frozen=%g property=%g\n",
             error, historyError, frozenError, propertyError);
  return iteration >= iterations;
}

inline void restoreForNextPass()
{
  historyU.restore();
  historyS.restore();
  pressure.restore();
  predictorU.restore();
  predictorS.restore();
  urst.restore();
  boundary.restore();
  rhsU.restore();
  rhsS.restore();
  explicitU.restore();
  explicitS.restore();
  propU.restore();
  // Native evaluateProperties applies AVM AFTER the scalar solve. Replay
  // must start from its PRE-solve coefficients, not the first pass output.
  propSBefore.restore();
  divergence.restore();
  coeffBdf.restore();
  coeffExtU.restore();
  coeffExtS.restore();
  // runInnerStep suppresses checkpointStep on an unconverged pass. Restore
  // the original request for the next pass; do not write intermediate output.
  nrs->checkpointStep = requestedCheckpoint;
}

inline void configure()
{
  nekrsCheck(enabled != 0 && enabled != 1, platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "picardEnabled must be 0 or 1.\n");
  nekrsCheck(iterations < 1 || !std::isfinite(tolerance) || tolerance <= 0,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "innerIterations must be positive and repeatTolerance finite and positive.\n");
  if (!enabled) return;
  // These restrictions define the frozen-system control experiment. Expanding
  // to nonlinear/moving-mesh/multisession coupling requires a separate design.
  for (const auto *key : {"MOVING MESH", "LOWMACH", "FLUID PRESSURE RHO SPLITTING",
                           "CONSTANT FLOW RATE", "NEKNEK MULTIRATE TIMESTEPPER"}) {
    nekrsCheck(platform->options.compareArgs(key, "TRUE"), platform->comm.mpiComm(),
               EXIT_FAILURE, "Frozen-inner demo does not support %s.\n", key);
  }
  nekrsCheck(platform->options.compareArgs("FLUID VELOCITY EXTRAPOLATION", "FALSE"),
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Frozen-inner demo requires a separate velocity predictor array.\n");
  int sessions = 1;
  platform->options.getArgs("NEKNEK NUMBER OF SESSIONS", sessions);
  nekrsCheck(sessions > 1 || nrs->geom || nrs->neknek || nrs->advectionSubcycingSteps != 0,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Frozen-inner demo requires one session and no advection subcycling.\n");
  for (const auto *key : {"FLUID PRESSURE INITIAL GUESS", "FLUID VELOCITY INITIAL GUESS",
                           "SCALAR00 INITIAL GUESS"}) {
    nekrsCheck(!platform->options.compareArgs(key, "PREVIOUS"), platform->comm.mpiComm(),
               EXIT_FAILURE, "Frozen-inner demo requires initialGuess=previous for %s.\n", key);
  }
  nekrsCheck(!nrs->scalar || nrs->Nscalar != 1 || nrs->userSource || nrs->userProperties
               || nrs->postScalar || nrs->preFluid || nrs->userDivergence || nrs->userConvergenceCheck,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Frozen-inner demo expects turbPipe's single scalar and no extra callbacks.\n");
  nrs->userSource = &beginStep;
  nrs->userConvergenceCheck = &convergenceCheck;
}
} // namespace frozenInner
