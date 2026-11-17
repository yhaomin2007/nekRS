#pragma once

namespace bubbleColumn { namespace phaseMomentum {
static int interval=10;
static deviceMemory<dfloat> mixtureDivergence,alphaHistory;
static deviceMemory<dfloat> gasDivergenceLocal,gasDivergenceAverage,gasCompression;
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
  alphaHistory.resize(offset);
  gasDivergenceLocal.resize(offset);
  gasDivergenceAverage.resize(offset);
  gasCompression.resize(offset);
  activeScalars=nrs->scalar->compute;
  activeScalars[nrs->scalar->nameToIndex.at("alpha")]=0;
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
  // DG alpha owns its history; native scalar alpha stays disabled.
  trace("UG_BEGIN");
}
inline void postScalar(double time,int)
{
  trace("UG_DONE");
  trace("DG_ALPHA_BEGIN_NEW_UG");
  alphaDG::advance(time);
  trace("DG_ALPHA_DONE_CG_PROJECTION_READY");
}
inline void preFluid(double,int)
{
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
  nrs->fluid->o_prop.slice(0,offset).copyFrom(o_muM);
  nrs->fluid->o_prop.slice(offset,offset).copyFrom(o_rhoM);
  if (nrs->tstep<=0) {
    platform->linAlg->fill(N,0.0,o_divSource);
    return;
  }
  // Evaluate UM^EXT.grad(alpha) directly at cubature points, then project
  // and assemble using the inherited CG cubature operator. This target
  // uses projected alpha; DG transport itself uses collocated GLL fluxes.
  // Restore the gas advector afterwards; do not alter native Urst or UM.
  o_gasAdvector.copyFrom(mixtureExtrapolated,3*offset);
  evaluateGasCubatureTransport(false,true);
  mixtureAlphaAdvection.copyFrom(o_gasTransport,N);
  o_gasAdvector.copyFrom(o_ug,3*offset);
  // BDF1 of the projected DG update is its actual step-average change.
  // Projection is linear, mass-conserving, and not overwritten by scalar BCs.
  buildExplicitMixtureDivergenceKernel(N,p.rhoLiquid,p.rhoGas,nrs->g0/nrs->dt[0],
      nrs->scalar->o_solution("alpha"),alphaHistory,mixtureAlphaAdvection,o_divSource);
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

  // ExecuteStep has reconstructed the current UG from the solved scalars.
  // Use dedicated buffers: do not overwrite any transport/coupling workspace.
  opSEM::strongDivergence(nrs->meshV,offset,o_ug,gasDivergenceLocal,false);
  opSEM::strongDivergence(nrs->meshV,offset,o_ug,gasDivergenceAverage,true);
  platform->linAlg->axmyz(N,-1.0,alphaDG::solution,gasDivergenceLocal,gasCompression);
  // This product is a diagnostic of material compression, NOT an extra DG
  // source or an exact decomposition of the collocated discrete DG residual.
  nrs->writeToFile("gasDiag",time,nrs->meshV,
      {{"velocity",{o_ug.slice(0,N),o_ug.slice(offset,N),o_ug.slice(2*offset,N)}},
       {"scalar00",{gasDivergenceLocal.slice(0,N)}},
       {"scalar01",{gasDivergenceAverage.slice(0,N)}},
       {"scalar02",{gasCompression.slice(0,N)}},
       {"scalar03",{alphaDG::solution.slice(0,N)}}},true,true);
  if(alphaDG::divUgFilterStrength>0 && nrs->tstep>0)
    nrs->writeToFile("divUGFilter",time,nrs->meshV,
        {{"scalar00",{alphaDG::divUg.slice(0,N)}},
         {"scalar01",{alphaDG::filteredDivUg.slice(0,N)}},
         {"scalar02",{alphaDG::divUgCorrection.slice(0,N)}}},true,true);
  const auto comm=platform->comm.mpiComm();
  const dfloat lo=platform->linAlg->min(N,gasDivergenceLocal,comm);
  const dfloat hi=platform->linAlg->max(N,gasDivergenceLocal,comm);
  const dfloat clo=platform->linAlg->min(N,gasCompression,comm);
  const dfloat chi=platform->linAlg->max(N,gasCompression,comm);
  if(platform->comm.mpiRank()==0)
    printf("gasDiag time=%.9e divUG[min,max]=[%.9e,%.9e] "
           "minusAlphaDivUG[min,max]=[%.9e,%.9e]\n",time,lo,hi,clo,chi);
}
} } // namespace bubbleColumn::phaseMomentum
