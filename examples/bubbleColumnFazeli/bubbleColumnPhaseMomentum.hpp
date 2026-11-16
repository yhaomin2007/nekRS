#pragma once

namespace bubbleColumn { namespace phaseMomentum {
static int interval=10;
static deviceMemory<dfloat> mixtureDivergence,alphaHistory,alphaRegularization;
static deviceMemory<dfloat> mixtureExtrapolated,mixtureAlphaAdvection;
static std::vector<int> activeScalars;
inline bool due()
{ return nrs->tstep==1 || nrs->checkpointStep || (interval>0 && nrs->tstep%interval==0); }
inline void trace(const char *stage)
{
  if (due() && platform->comm.mpiRank()==0) {
    printf("method1Order step=%d stage=%s\n",nrs->tstep,stage); fflush(stdout);
  }
}
inline void allocate()
{
  const auto comm=platform->comm.mpiComm();
  nekrsCheck(nrs->Nscalar!=4 || nrs->geom || nrs->neknek ||
      nrs->advectionSubcycingSteps || nrs->scalar->cvode,comm,EXIT_FAILURE,
      "%s","Method1 requires four fixed-mesh elliptic scalars.\n");
  int index=0;
  for (const char *name:{"alpha","ugx","ugy","ugz"}) {
    const int is=nrs->scalar->nameToIndex.at(name);
    nekrsCheck(is!=index++ || !nrs->scalar->compute[is],comm,EXIT_FAILURE,
        "%s","Required scalar order: ALPHA, UGX, UGY, UGZ.\n");
    auto coeff=nrs->scalar->o_transportCoeff(name);
    nekrsCheck(platform->linAlg->min(nrs->meshV->Nlocal,coeff,comm)!=1 ||
        platform->linAlg->max(nrs->meshV->Nlocal,coeff,comm)!=1,comm,EXIT_FAILURE,
        "%s","Method1 scalars require transportCoeff=1.\n");
  }
  const dlong offset=nrs->fieldOffset;
  mixtureDivergence.resize(2*offset);
  platform->linAlg->fill(2*offset,0.0,mixtureDivergence);
  mixtureExtrapolated.resize(3*offset); mixtureAlphaAdvection.resize(offset);
  alphaHistory.resize(offset); alphaRegularization.resize(offset);
  activeScalars=nrs->scalar->compute;
  nekrsCheck(interval<0,comm,EXIT_FAILURE,"%s","diagnosticInterval must be nonnegative.\n");
}
inline void beginStep(double time)
{
  nrs->scalar->compute=activeScalars;
  // Native extrapolateSolution has already constructed UM^EXT. Preserve it
  // before any boundary updates or the native pressure/velocity solve.
  mixtureExtrapolated.copyFrom(nrs->fluid->o_Ue,3*nrs->fieldOffset);
  // UL is reconstructed from UM, alpha and UG. The existing gas-only source
  // includes explicit drag/stress and semi-implicit VM with extrapolated UL
  // material acceleration; no liquid momentum equation is solved.
  addExplicitSources(time);
  referenceCase::addSources();
  const dlong N=nrs->meshV->Nlocal;
  // Capture alpha's BDF load BEFORE native lagSolution discards the oldest
  // history. Native makeForcing will prepare/lag the UG forcing only.
  std::vector<dfloat> weights(nrs->scalar->o_coeffBDF.size());
  nrs->scalar->o_coeffBDF.copyTo(weights.data());
  platform->linAlg->fill(N,0.0,alphaHistory);
  const int ia=nrs->scalar->nameToIndex.at("alpha");
  for (int j=0;j<static_cast<int>(weights.size());++j) {
    auto input=nrs->scalar->o_S.slice(nrs->scalar->fieldOffsetScan[ia]+
        j*nrs->scalar->fieldOffsetSum,N);
    platform->linAlg->axpby(N,weights[j]/nrs->dt[0],input,1.0,alphaHistory);
  }
  // Retain the native HPFRT/GJP contribution evaluated on alpha^n, once.
  // No user stabilization or extra time-history advancement is introduced.
  auto explicitAlpha=nrs->scalar->o_explicitTerms("alpha");
  platform->linAlg->fill(N,0.0,explicitAlpha);
  nrs->scalar->makeExplicit(ia,time,nrs->tstep);
  alphaRegularization.copyFrom(explicitAlpha,N);
  nrs->scalar->compute[ia]=0; // native scalar sweep solves UG only
  trace("UG_BEGIN");
}
inline void postScalar(double time,int tstep)
{
  trace("UG_DONE");
  auto &scalar=nrs->scalar;
  auto mesh=nrs->meshV;
  const dlong N=mesh->Nlocal;
  const int ia=scalar->nameToIndex.at("alpha");
  // Solve alpha exactly once, after the UG sweep. Applying its BCs here
  // cannot overwrite the newly computed interior phase velocities.
  scalar->compute.assign(activeScalars.size(),0);
  scalar->compute[ia]=1;
  scalar->applyDirichlet(time);
  prepareGasAdvection(); // reads the just-solved UG, not o_Se or an EXT history
  evaluateAlphaTransport(); // alpha^n with UG^(n+1)
  platform->linAlg->axpby(N,1.0,referenceCase::source,1.0,o_alphaSource);
  platform->linAlg->axpby(N,1.0,o_alphaExplicitDiffusion,1.0,o_alphaSource);
  auto explicitAlpha=scalar->o_explicitTerms("alpha");
  explicitAlpha.copyFrom(o_alphaSource,N);
  platform->linAlg->axpby(N,1.0,alphaRegularization,1.0,explicitAlpha);
  // Same local mass-weighted RHS as native sumMakef for unit transportCoeff
  // and the configured scalar EXT1 carrier. BDF history was saved pre-lag.
  auto load=scalar->o_JwF.slice(scalar->fieldOffsetScan[ia],N);
  load.copyFrom(alphaHistory,N);
  platform->linAlg->axpby(N,1.0,explicitAlpha,1.0,load);
  platform->linAlg->axmy(N,1.0,mesh->o_LMM,load);
  trace("ALPHA_BEGIN_NEW_UG");
  scalar->solve(time,1); // only alpha enabled; no lagSolution/makeForcing here
  scalar->compute=activeScalars;
  postProcessAlpha(time,tstep);
  trace("ALPHA_DONE");
}
inline void preFluid(double time,int)
{
  // UM wall values depend on newly solved alpha, to keep reconstructed UL=0.
  nrs->fluid->applyDirichlet(time);
  // Preserve exactly the imposed target BEFORE the native pressure/UM solve.
  // Do not reconstruct it afterwards from updated alpha/phase velocities.
  if (nrs->checkpointStep)
    mixtureDivergence.copyFrom(nrs->fluid->o_div,nrs->meshV->Nlocal);
  trace("NATIVE_PRESSURE_UM_BEGIN");
}
inline void properties(double)
{
  // Alpha has completed its solve; all inputs to the target are now known.
  // This replaces the relative-flux target containing a lagged div(UM).
  evaluatePointwiseTerms();
  auto mesh=nrs->meshV;
  const dlong N=mesh->Nlocal,offset=nrs->fieldOffset;
  // Physical liquid stress is supplied explicitly; do not also diffuse UM.
  auto nativeViscosity=nrs->fluid->o_prop.slice(0,offset);
  platform->linAlg->fill(N,0.0,nativeViscosity);
  nrs->fluid->o_prop.slice(offset,offset).copyFrom(o_rhoM);
  if (nrs->tstep<=0) {
    platform->linAlg->fill(N,0.0,o_divSource);
    return;
  }
  // Evaluate UM^EXT.grad(alpha) directly at cubature points, then project
  // and assemble with the same quadrature as gas alpha transport.
  // Restore the gas advector afterwards; do not alter native Urst or UM.
  o_gasAdvector.copyFrom(mixtureExtrapolated,3*offset);
  evaluateGasCubatureTransport(false,true);
  mixtureAlphaAdvection.copyFrom(o_gasTransport,N);
  o_gasAdvector.copyFrom(o_ug,3*offset);
  // Reuse the pre-lag weighted BDF history. This includes alpha's actual
  // HPFRT/clipping/BC effects without adding any diffusion term twice.
  buildExplicitMixtureDivergenceKernel(N,p.rhoLiquid,p.rhoGas,nrs->g0/nrs->dt[0],
      nrs->scalar->o_solution("alpha"),alphaHistory,mixtureAlphaAdvection,o_divSource);
  referenceCase::addInjectionDivergenceKernel(N,p.rhoLiquid,p.rhoGas,
      nrs->scalar->o_solution("alpha"),referenceCase::source,o_divSource);
  assembleDivergence(o_divSource);
  trace("DIV_UM_EXPLICIT_UM_EXT");
}
inline bool singlePass(int stage)
{
  nekrsCheck(stage!=1,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","Method1 requires exactly one native stage.\n");
  return true;
}
inline void diagnostics(double time)
{
  if (!nrs->checkpointStep) return;
  const dlong N=nrs->meshV->Nlocal,offset=nrs->fieldOffset;
  auto target=mixtureDivergence.slice(0,N);
  auto actual=mixtureDivergence.slice(offset,N);
  opSEM::strongDivergence(nrs->meshV,offset,nrs->fluid->o_U,actual,true);
  opSEM::strongGrad(nrs->meshV,offset,nrs->fluid->o_P,o_gradPStrong);
  nrs->writeToFile("diag",time,nrs->meshV,
      {{"velocity",{o_gradPStrong.slice(0,N),o_gradPStrong.slice(offset,N),o_gradPStrong.slice(2*offset,N)}},
       {"scalar00",{target}},{"scalar01",{actual}}},true,true);
}
} } // namespace bubbleColumn::phaseMomentum
