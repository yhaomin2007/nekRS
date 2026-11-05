#pragma once

// Fixed-mesh one-through ordering: alpha -> native UM/pressure -> UG.
// No core changes; native history shifts occur exactly once in init/runStep.
namespace bubbleColumn { namespace oneThrough {
static std::vector<int> enabled;
static deviceMemory<dfloat> zeroAdvection, pressureAtSource, dragAtSource, pressureDelta;
static int preparedStep = -1, solvedStep = -1;

inline bool singlePass(int stage)
{
  nekrsCheck(stage != 1, platform->comm.mpiComm(), EXIT_FAILURE,
      "%s", "Method1 one-through requires exactly one native stage.\n");
  return true;
}

inline void beginStep(double time)
{
  nekrsCheck(preparedStep == nrs->tstep || preparedStep != solvedStep,
      platform->comm.mpiComm(), EXIT_FAILURE,
      "%s", "Method1 timestep initialization repeated or previous UG solve missing.\n");
  preparedStep = nrs->tstep;
  auto &scalar = nrs->scalar;
  scalar->compute = enabled;
  addExplicitSources(time);
  const dlong N = nrs->meshV->Nlocal, offset = nrs->fieldOffset;
  pressureAtSource.copyFrom(o_gradP, 3 * offset);
  dragAtSource.copyFrom(o_dragLambda, N);

  // Prepare UG's native BDF/EXT1/HPF forcing BEFORE lagSolution(). The
  // private native ADV arrays are inaccessible, but identically zero in this
  // case: all native gas-scalar advection is disabled from initialization.
  // Use a separate zero array, with full EXT history extents.
  for (const char *name : {"ugx", "ugy", "ugz"}) {
    const int is = scalar->nameToIndex.at(name);
    scalar->makeExplicit(is, time, nrs->tstep);
    launchKernel("scalar_t::sumMakef",
        N, nrs->meshV->o_LMM, 1.0 / nrs->dt[0],
        scalar->o_coeffEXT, scalar->o_coeffBDF,
        scalar->fieldOffsetScan[is], scalar->fieldOffsetSum,
        nrs->meshV->fieldOffset, scalar->o_transportCoeff(),
        scalar->o_S, zeroAdvection, scalar->o_EXT, scalar->o_JwF);
    scalar->compute[is] = 0;
  }
  // Native initStep now builds only alpha's forcing. Its makeForcing shifts
  // ALL scalar source histories once; native lagSolution also shifts ALL
  // scalar solution histories once, regardless of this host solve mask.
  // Keep device o_compute unchanged: gas scalar native advection stays off.
}

inline void solveGas(double time, int tstep)
{
  nekrsCheck(tstep != preparedStep || solvedStep == tstep
      || !nrs->timeStepConverged, platform->comm.mpiComm(), EXIT_FAILURE,
      "%s", "Method1 deferred UG solve called out of order or more than once.\n");
  auto &scalar = nrs->scalar;
  const dlong N = nrs->meshV->Nlocal, offset = nrs->fieldOffset;

  // Evaluate current p after the native projection. Spatial filter, optional
  // temporal relaxation, and inlet mask are shared with source evaluation.
  // This is the ONLY per-step temporal-filter advance.
  evaluateGasPressureGradient(true);
  const dfloat rhoEff = p.rhoGas
      + p.virtualMassEnabled * p.virtualMassCoefficient * p.rhoLiquid;
  const dfloat pressureScale = -p.gasPressureEnabled / rhoEff;
  for (int c = 0; c < 3; ++c) {
    auto delta = pressureDelta.slice(c * offset, offset);
    delta.copyFrom(o_gradP, N, 0, c * offset);
    platform->linAlg->axpby(N, -1.0,
        pressureAtSource.slice(c * offset, offset), 1.0, delta);
    platform->linAlg->axmy(N, pressureScale, nrs->meshV->o_LMM, delta);
    const char *names[] = {"ugx", "ugy", "ugz"};
    const int is = scalar->nameToIndex.at(names[c]);
    auto rhs = scalar->o_JwF.slice(scalar->fieldOffsetScan[is], N);
    platform->linAlg->axpby(N, 1.0, delta, 1.0, rhs);
  }

  // Properties evaluation for UM refreshed scratch drag coefficients. Restore
  // the source-time coefficient so the implicit drag and explicit UL load
  // still use the same frozen state.
  o_dragLambda.copyFrom(dragAtSource, N);
  scalar->compute = enabled;
  scalar->compute[scalar->nameToIndex.at("alpha")] = 0;
  scalar->applyDirichlet(time);
  scalar->solve(time, 1);
  scalar->compute = enabled;
  postProcessGasFlux();
  solvedStep = tstep;
}

inline void configure()
{
  const auto comm = platform->comm.mpiComm();
  nekrsCheck(nrs->Nscalar != 4 || nrs->geom || nrs->neknek
      || nrs->advectionSubcycingSteps || nrs->scalar->cvode
      || p.scalarExtrapolationEnabled, comm, EXIT_FAILURE,
      "%s", "Method1 one-through requires four fixed-mesh elliptic scalars, "
             "EXT1 scalar sources, and no subcycling/NekNek.\n");
  nekrsCheck(nrs->preFluid || nrs->userConvergenceCheck, comm, EXIT_FAILURE,
      "%s", "Method1 one-through cannot replace existing stage callbacks.\n");
  for (const char *key : {"MOVING MESH", "NEKNEK MULTIRATE TIMESTEPPER"})
    nekrsCheck(platform->options.compareArgs(key, "TRUE"), comm, EXIT_FAILURE,
        "Method1 one-through does not support %s.\n", key);
  enabled = nrs->scalar->compute;
  for (const char *name : {"alpha", "ugx", "ugy", "ugz"})
    nekrsCheck(!enabled[nrs->scalar->nameToIndex.at(name)], comm, EXIT_FAILURE,
        "Method1 requires enabled %s solve.\n", name);
  const dlong offset = nrs->fieldOffset;
  pressureAtSource.resize(3 * offset);
  pressureDelta.resize(3 * offset);
  dragAtSource.resize(offset);
  zeroAdvection.resize(nrs->scalar->o_coeffEXT.size() * nrs->scalar->fieldOffsetSum);
  platform->linAlg->fill(zeroAdvection.size(), 0.0, zeroAdvection);
  nrs->userConvergenceCheck = &singlePass;
  if (platform->comm.mpiRank() == 0)
    printf("Method1 one-through: ALPHA -> native UM/pressure -> UG (current pressure).\n");
}
} } // bubbleColumn::oneThrough
