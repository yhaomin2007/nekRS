#pragma once

// Fixed-mesh one-through ordering: alpha -> native P/UM -> UG.
// Source/BDF/EXT preparation precedes native lagging; no native history is
// advanced a second time. Only the direct pressure load is refreshed for UG.
namespace bubbleColumn2 { namespace afterFluid {
static std::vector<int> enabled;
static deviceMemory<dfloat> zeroAdvection, pressureAtSource, pressureDelta;
static deviceMemory<dfloat> dragAtSource, gasDiffusion;
static dfloat pressureScale=0;
static int preparedStep=-1, solvedStep=-1, propertiesStep=-1;
static int logInterval=10;

inline void stageLog(const char *stage)
{
  if(logInterval==0 || (nrs->tstep!=1 && nrs->tstep%logInterval!=0)
      || platform->comm.mpiRank()!=0) return;
  const auto &s=nrs->scalar;
  printf("method2Order step=%d stage=%s hostSolve(alpha,ugx,ugy,ugz)=(%d,%d,%d,%d)\n",
      nrs->tstep,stage,s->compute[0],s->compute[1],s->compute[2],s->compute[3]);
  fflush(stdout);
}
inline bool singlePass(int stage)
{
  nekrsCheck(stage!=1,platform->comm.mpiComm(),EXIT_FAILURE,"%s",
      "Method2 UG-after-fluid requires exactly one native stage.\n");
  return true;
}
inline void postScalar(double time,int step)
{
  postProcessAlpha(time,step);
  stageLog("ALPHA_DONE");
}
inline void preFluid(double,int)
{
  stageLog("P_UM_BEGIN");
}
inline void properties(double time)
{
  updateProperties(time);
  if(preparedStep!=nrs->tstep || solvedStep==nrs->tstep) return;
  const char *names[]={"ugx","ugy","ugz"};
  // Capture post-alpha diffusion/AVM before UM can mutate scratch properties.
  for(int c=0;c<3;++c)
    gasDiffusion.copyFrom(nrs->scalar->o_diffusionCoeff(names[c]),
        nrs->meshV->Nlocal,c*nrs->fieldOffset,0);
  propertiesStep=nrs->tstep;
}
inline void beginStep(double time)
{
  // Adaptive-CFL probes at initialization may call userSource with tstep=0.
  if(nrs->tstep<=0) {addExplicitSources(time);return;}
  nekrsCheck(preparedStep==nrs->tstep || preparedStep!=solvedStep,
      platform->comm.mpiComm(),EXIT_FAILURE,"%s",
      "Method2 source preparation repeated or previous deferred UG solve missing.\n");
  preparedStep=nrs->tstep;
  auto &s=nrs->scalar;
  s->compute=enabled;
  addExplicitSources(time);
  const dlong N=nrs->meshV->Nlocal,O=nrs->fieldOffset;
  pressureAtSource.copyFrom(o_gradP,3*O);
  dragAtSource.copyFrom(o_dragLambda,N);
  const dfloat rhoEff=p.rhoGas+(p.explicitVirtualMassEnabled?0:p.virtualMassEnabled)
      *virtualMassRampFactor(nrs->tstep)*p.virtualMassCoefficient*p.rhoLiquid;
  pressureScale=-p.gasPressureEnabled/rhoEff;
  // Device gas-advection flags are permanently zero in this mode; source-side
  // native UM-advection cancellation for UG is also omitted. Its actual UG
  // cubature convection/EXT is already in the user source. Alpha is unchanged.
  for(const char *name:{"ugx","ugy","ugz"}) {
    const int is=s->nameToIndex.at(name);
    s->makeExplicit(is,time,nrs->tstep);
    launchKernel("scalar_t::sumMakef",N,nrs->meshV->o_LMM,1.0/nrs->dt[0],
        s->o_coeffEXT,s->o_coeffBDF,s->fieldOffsetScan[is],s->fieldOffsetSum,
        nrs->meshV->fieldOffset,s->o_transportCoeff(),s->o_S,
        zeroAdvection,s->o_EXT,s->o_JwF);
    s->compute[is]=0;
  }
  stageLog("ALPHA_PENDING");
  // Native makeForcing assembles only alpha and shifts ALL explicit histories
  // once. Native lagSolution subsequently shifts ALL solution histories once.
}
inline void solveGas(double time,int step)
{
  nekrsCheck(step!=preparedStep || solvedStep==step || propertiesStep!=step
      || !nrs->timeStepConverged,platform->comm.mpiComm(),EXIT_FAILURE,"%s",
      "Method2 deferred UG solve is out of order, duplicated or missing post-alpha properties.\n");
  auto &s=nrs->scalar;
  auto mesh=nrs->meshV;
  const dlong N=mesh->Nlocal,O=nrs->fieldOffset;
  stageLog("P_UM_DONE");
  if(p.gasPressureGradientMethod==0) {
    opSEM::strongGrad(mesh,O,nrs->fluid->o_P,o_gradPStrong);
  } else {
    launchKernel("core-wGradientVolumeHex3D",mesh->Nelements,mesh->o_vgeo,
        mesh->o_D,O,nrs->fluid->o_P,o_gradPWeak);
    oogs::startFinish(o_gradPWeak,mesh->dim,O,ogsDfloat,ogsAdd,mesh->oogs);
    platform->linAlg->axmyMany(N,mesh->dim,O,0,-1.0,mesh->o_invLMM,o_gradPWeak);
  }
  prepareGasPressureForcing(); // spatial selection/masks; temporal filter off
  const char *names[]={"ugx","ugy","ugz"};
  for(int c=0;c<3;++c) {
    auto delta=pressureDelta.slice(c*O,O);
    delta.copyFrom(o_gradP,N,0,c*O);
    platform->linAlg->axpby(N,-1.0,pressureAtSource.slice(c*O,O),1.0,delta);
    platform->linAlg->axmy(N,pressureScale,mesh->o_LMM,delta);
    const int is=s->nameToIndex.at(names[c]);
    auto rhs=s->o_JwF.slice(s->fieldOffsetScan[is],N);
    platform->linAlg->axpby(N,1.0,delta,1.0,rhs);
    s->o_diffusionCoeff(names[c]).copyFrom(gasDiffusion,N,0,c*O);
  }
  // Keep implicit drag and its source-time liquid-velocity load paired.
  o_dragLambda.copyFrom(dragAtSource,N);
  // This snapshot now describes the gradient ACTUALLY used in the UG solve.
  o_ugPredictorPressureGradient.copyFrom(o_gradP,3*O);
  s->compute=enabled;s->compute[0]=0;
  s->applyDirichlet(time);
  stageLog("UG_BEGIN_CURRENT_PRESSURE");
  s->solve(time,1);
  stageLog("UG_DONE");
  s->compute=enabled;
  solvedStep=step;
  updateProperties(time); // final reconstructed phase fields/properties, no lag
  stageLog("STEP_FIELDS_READY");
}
inline void configure()
{
  if(!p.gasSolveAfterFluid) return;
  const auto comm=platform->comm.mpiComm();
  platform->par->extract("casedata","gassolveorderloginterval",logInterval);
  nekrsCheck(logInterval<0 || nrs->Nscalar!=4 || nrs->geom || nrs->neknek
      || nrs->advectionSubcycingSteps || nrs->scalar->cvode
      || p.frozenGasVelocityEnabled || p.coupledVirtualMassEnabled
      || p.gasPressureGradientMethod>1 || p.gasPressureGradientFilterEnabled!=0,
      comm,EXIT_FAILURE,"%s",
      "Method2 deferred UG requires four solved fixed-mesh elliptic scalars, uncoupled VM, "
      "computed pressure without temporal filtering, and no subcycling/NekNek.\n");
  nekrsCheck(nrs->preFluid || nrs->userConvergenceCheck,comm,EXIT_FAILURE,"%s",
      "Method2 deferred UG cannot replace existing stage callbacks.\n");
  enabled=nrs->scalar->compute;
  for(int i=0;i<4;++i)nekrsCheck(!enabled[i],comm,EXIT_FAILURE,"%s",
      "Method2 deferred UG requires alpha and all UG solves enabled.\n");
  const dlong O=nrs->fieldOffset;
  pressureAtSource.resize(3*O);pressureDelta.resize(3*O);
  dragAtSource.resize(O);gasDiffusion.resize(3*O);
  zeroAdvection.resize(std::max(nrs->scalar->o_coeffEXT.size(),nrs->scalar->o_coeffBDF.size())
      *nrs->scalar->fieldOffsetSum);
  platform->linAlg->fill(zeroAdvection.size(),0.0,zeroAdvection);
  std::vector<dlong> advection(enabled.begin(),enabled.end());
  for(int i=1;i<4;++i)advection[i]=0;
  nrs->scalar->o_compute.copyFrom(advection.data());
  nrs->userSource=&beginStep;
  nrs->postScalar=&postScalar;
  nrs->preFluid=&preFluid;
  nrs->userProperties=&properties;
  nrs->userConvergenceCheck=&singlePass;
  if(platform->comm.mpiRank()==0)
    printf("Method2 solve order: ALPHA -> native P/UM -> UG(current pressure); "
           "one UG solve, pressure corrector OFF, convection fluid EXT retained.\n");
}
} } // bubbleColumn2::afterFluid
