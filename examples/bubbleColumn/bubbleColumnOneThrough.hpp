#pragma once
// Native loop solves alpha, UGX, UGY, UGZ, then native pressure and UM.
// Callbacks only: no field solves, compute-mask changes or history shifts here.
namespace bubbleColumn { namespace oneThrough {
static int logInterval = 10;
inline void stageLog(const char *stage)
{
  if (logInterval == 0 || (nrs->tstep != 1 && nrs->tstep % logInterval != 0)
      || platform->comm.mpiRank() != 0) return;
  const auto &s = nrs->scalar;
  printf("method1Order step=%d stage=%s hostSolve(alpha,ugx,ugy,ugz)=(%d,%d,%d,%d)\n",
      nrs->tstep, stage, s->compute[s->nameToIndex.at("alpha")],
      s->compute[s->nameToIndex.at("ugx")], s->compute[s->nameToIndex.at("ugy")],
      s->compute[s->nameToIndex.at("ugz")]);
  fflush(stdout);
}
inline void beginStep(double time)
{
  addExplicitSources(time);
  stageLog("ALPHA_UG_PENDING");
}
inline void postScalar(double time, int tstep)
{
  postProcessAlpha(time, tstep);
  postProcessGasFlux();
  stageLog("ALPHA_UG_DONE");
}
inline void preFluid(double, int) { stageLog("PRESSURE_UM_BEGIN"); }
inline void finishStep() { stageLog("PRESSURE_UM_DONE"); }
inline bool singlePass(int stage)
{
  nekrsCheck(stage != 1, platform->comm.mpiComm(), EXIT_FAILURE,
      "%s", "Method1 one-through requires exactly one native stage.\n");
  return true;
}
inline void configure()
{
  const auto comm = platform->comm.mpiComm();
  platform->par->extract("casedata", "onethroughloginterval", logInterval);
  nekrsCheck(logInterval < 0, comm, EXIT_FAILURE,
      "%s", "oneThroughLogInterval must be nonnegative.\n");
  nekrsCheck(nrs->Nscalar != 4 || nrs->geom || nrs->neknek
      || nrs->advectionSubcycingSteps || nrs->scalar->cvode,
      comm, EXIT_FAILURE, "%s", "Method1 requires four fixed-mesh elliptic gas scalars.\n");
  nekrsCheck(nrs->preFluid || nrs->userConvergenceCheck, comm, EXIT_FAILURE,
      "%s", "Method1 cannot overwrite existing stage callbacks.\n");
  int index = 0;
  for (const char *name : {"alpha", "ugx", "ugy", "ugz"}) {
    const int is = nrs->scalar->nameToIndex.at(name);
    nekrsCheck(is != index++ || !nrs->scalar->compute[is], comm, EXIT_FAILURE,
        "%s", "Method1 scalar order must be ALPHA, UGX, UGY, UGZ, all enabled.\n");
  }
  for (const char *key : {"MOVING MESH", "NEKNEK MULTIRATE TIMESTEPPER"})
    nekrsCheck(platform->options.compareArgs(key, "TRUE"), comm, EXIT_FAILURE,
        "Method1 does not support %s.\n", key);
  nrs->postScalar = &postScalar;
  nrs->preFluid = &preFluid;
  nrs->userConvergenceCheck = &singlePass;
  if (platform->comm.mpiRank() == 0)
    printf("Method1 order: ALPHA -> UG -> PRESSURE -> UM; convection=fluid EXT; "
           "UG pressure=strong %s (latest available completed pressure).\n",
        p.gasPressureGradientExtrapolation ? "EXT" : "latest");
}
} } // namespace bubbleColumn::oneThrough

