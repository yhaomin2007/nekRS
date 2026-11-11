#pragma once

namespace bubbleColumn { namespace phaseMomentum {
static const char *gasNames[]={"ugx","ugy","ugz"};
static const char *liquidNames[]={"ulx","uly","ulz"};
static const char *termNames[]={"pressure","gravity","drag","viscosity","virtual_mass","convection_EXT"};
static const char *termFiles[]={"gpr","ggr","gdr","gvs","gvm","gcv"};
static int interval=10;
static bool masksReady=false;
static double sourceTime=0;
static occa::kernel stressKernel,predictorKernel,maskKernel,correctKernel,nonfiniteKernel,defectKernel,initializeMixtureKernel;
static deviceMemory<dfloat> stress,divStress,convection,convectionEXT,rhs,terms;
static deviceMemory<dfloat> freeG,freeL,deltaG,deltaL,before,after;
static deviceMemory<dfloat> bad;
static deviceMemory<dfloat> sourceAlpha,sourceRhsG,historyG,historyL,defectG,defectL;
static ExplicitHistory convectionHistory;

inline void registerKernels(deviceKernelProperties &info)
{
  const std::string request="bubbleColumn::phaseMomentum";
  if (platform->options.compareArgs("REGISTER ONLY","TRUE"))
    platform->kernelRequests.add(request,"bubbleColumnPhaseMomentum.okl",info);
  else {
    initializeMixtureKernel=platform->kernelRequests.load(request,"phaseInitializeMixture");
    stressKernel=platform->kernelRequests.load(request,"liquidStress");
    predictorKernel=platform->kernelRequests.load(request,"phasePredictor");
    maskKernel=platform->kernelRequests.load(request,"phaseMask");
    correctKernel=platform->kernelRequests.load(request,"phaseCorrect");
    nonfiniteKernel=platform->kernelRequests.load(request,"phaseNonfinite");
    defectKernel=platform->kernelRequests.load(request,"phaseMomentumDefect");
  }
}
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
  nekrsCheck(nrs->Nscalar!=7 || nrs->geom || nrs->neknek ||
      nrs->advectionSubcycingSteps || nrs->scalar->cvode,comm,EXIT_FAILURE,
      "%s","Method1 requires seven fixed-mesh elliptic scalars.\n");
  int index=0;
  for (const char *name:{"alpha","ugx","ugy","ugz","ulx","uly","ulz"}) {
    const int is=nrs->scalar->nameToIndex.at(name);
    nekrsCheck(is!=index++ || !nrs->scalar->compute[is],comm,EXIT_FAILURE,
        "%s","Required scalar order: ALPHA, UGX, UGY, UGZ, ULX, ULY, ULZ.\n");
    auto coeff=nrs->scalar->o_transportCoeff(name);
    nekrsCheck(platform->linAlg->min(nrs->meshV->Nlocal,coeff,comm)!=1 ||
        platform->linAlg->max(nrs->meshV->Nlocal,coeff,comm)!=1,comm,EXIT_FAILURE,
        "%s","Phase predictor scalars require transportCoeff=1.\n");
  }
  const dlong offset=nrs->fieldOffset;
  stress.resize(9*offset); divStress.resize(3*offset);
  convection.resize(3*offset); convectionEXT.resize(3*offset);
  rhs.resize(3*offset); terms.resize(18*offset);
  freeG.resize(3*offset); freeL.resize(3*offset);
  deltaG.resize(3*offset); deltaL.resize(3*offset);
  before.resize(3*offset); after.resize(3*offset);
  bad.resize(offset);
  sourceAlpha.resize(offset); sourceRhsG.resize(3*offset);
  historyG.resize(3*offset); historyL.resize(3*offset);
  defectG.resize(3*offset); defectL.resize(3*offset);
  for (auto *v:{&deltaG,&deltaL,&before,&after,&terms,&defectG,&defectL})
    platform->linAlg->fill(v->size(),0.0,*v);
  nekrsCheck(interval<0,comm,EXIT_FAILURE,"%s","diagnosticInterval must be nonnegative.\n");
}

// Fresh-start initialization only. Read the actual phase scalar fields;
// never call this before a pressure solve during time stepping.
inline void initializeMixture()
{
  reconstructGasVelocity();
  reconstructLiquidVelocity();
  initializeMixtureKernel(nrs->meshV->Nlocal,nrs->fieldOffset,
      p.rhoGas,p.rhoLiquid,nrs->scalar->o_solution("alpha"),o_ug,o_ul,nrs->fluid->o_U);
}

inline void beginStep(double time)
{
  // Reuse conservative alpha transport, density/drift stress and native UM forcing.
  // No writes to UM or to its native time coefficients/history.
  addExplicitSources(time);
  auto mesh=nrs->meshV;
  const dlong N=mesh->Nlocal, offset=nrs->fieldOffset;
  // The same cubature operator and fluid EXT coefficients as UG convection.
  o_gasAdvector.copyFrom(o_ul,3*offset);
  evaluateGasCubatureTransport(false);
  for (int c=0;c<3;++c)
    convection.copyFrom(o_gasTransport,N,c*offset,(c+1)*offset);
  convectionHistory.apply(convection,convectionEXT,3,true);
  o_gasAdvector.copyFrom(o_ug,3*offset);
  stressKernel(N,offset,p.muLiquid,nrs->scalar->o_solution("alpha"),o_gradUl,stress);
  for (int c=0;c<3;++c) {
    auto row=stress.slice(3*c*offset,3*offset);
    auto div=divStress.slice(c*offset,offset);
    opSEM::strongDivergence(mesh,offset,row,div);
  }
  predictorKernel(N,offset,p.rhoGas,p.rhoLiquid,
      p.virtualMassEnabled*p.virtualMassCoefficient,p.alphaFloor,
      p.gravity[0],p.gravity[1],p.gravity[2],nrs->scalar->o_solution("alpha"),
      o_ug,o_ul,o_gradP,o_dragLambda,o_divGasStress,divStress,
      o_convectionEXT,convectionEXT,o_ugSource,rhs,terms);
  if (due()) {
    sourceAlpha.copyFrom(nrs->scalar->o_solution("alpha"),N);
    sourceRhsG.copyFrom(o_ugSource,3*offset);
    // Snapshot weighted BDF history BEFORE native lagSolution drops its oldest
    // slot. Never infer the history from final o_S or from a difference / dt.
    std::vector<dfloat> weights(nrs->scalar->o_coeffBDF.size());
    nrs->scalar->o_coeffBDF.copyTo(weights.data());
    for (int c=0;c<3;++c) for (int phase=0;phase<2;++phase) {
      const int is=nrs->scalar->nameToIndex.at(phase?liquidNames[c]:gasNames[c]);
      auto output=(phase?historyL:historyG).slice(c*offset,N);
      platform->linAlg->fill(N,0.0,output);
      for (int j=0;j<static_cast<int>(weights.size());++j) {
        auto input=nrs->scalar->o_S.slice(nrs->scalar->fieldOffsetScan[is]+
            j*nrs->scalar->fieldOffsetSum,N);
        platform->linAlg->axpby(N,weights[j]/nrs->dt[0],input,1.0,output);
      }
    }
  }
  for (int c=0;c<3;++c) {
    nrs->scalar->o_explicitTerms(gasNames[c]).copyFrom(o_ugSource,N,0,c*offset);
    nrs->scalar->o_explicitTerms(liquidNames[c]).copyFrom(rhs,N,0,c*offset);
  }
  sourceTime=time; // userSource is called with the completed step's time t^n.
  trace("ALPHA_UG_UL_PENDING");
}
inline void postScalar(double time,int tstep)
{
  postProcessAlpha(time,tstep);
  trace("ALPHA_UG_UL_DONE");
}
inline void preFluid(double,int) { trace("NATIVE_PRESSURE_UM_BEGIN"); }
inline bool singlePass(int stage)
{
  nekrsCheck(stage!=1,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","Method1 requires exactly one native stage.\n");
  return true;
}
inline void initializeMasks()
{
  if (masksReady) return;
  const dlong offset=nrs->fieldOffset;
  platform->linAlg->fill(3*offset,1.0,freeG);
  platform->linAlg->fill(3*offset,1.0,freeL);
  for (int c=0;c<3;++c) for (int phase=0;phase<2;++phase) {
    const char *name=phase?liquidNames[c]:gasNames[c];
    const int is=nrs->scalar->nameToIndex.at(name);
    auto elliptic=nrs->scalar->ellipticSolver[is];
    nekrsCheck(!elliptic,platform->comm.mpiComm(),EXIT_FAILURE,
        "Phase solver for %s is not initialized.\n",name);
    auto mask=(phase?freeL:freeG).slice(c*offset,offset);
    if (elliptic->Nmasked()) maskKernel(elliptic->Nmasked(),elliptic->o_maskIds(),mask);
  }
  masksReady=true;
}
inline void correct()
{
  trace("NATIVE_PRESSURE_UM_DONE");
  initializeMasks(); // elliptic solvers need not exist during UDF_Setup
  reconstructGasVelocity(); reconstructLiquidVelocity();
  const dlong N=nrs->meshV->Nlocal, offset=nrs->fieldOffset;
  correctKernel(N,offset,p.rhoGas,p.rhoLiquid,
      p.virtualMassEnabled*p.virtualMassCoefficient,p.alphaFloor,nrs->g0/nrs->dt[0],
      nrs->scalar->o_solution("alpha"),nrs->fluid->o_U,o_dragLambda,freeG,freeL,
      o_ug,o_ul,deltaG,deltaL,before,after);
  for (int c=0;c<3;++c) {
    nrs->scalar->o_solution(gasNames[c]).copyFrom(o_ug,N,0,c*offset);
    nrs->scalar->o_solution(liquidNames[c]).copyFrom(o_ul,N,0,c*offset);
  }
  trace("PHASE_CORRECTION_DONE");
}
inline void reportVector(const char *name,deviceMemory<dfloat> v)
{
  const auto comm=platform->comm.mpiComm();
  nonfiniteKernel(nrs->meshV->Nlocal,nrs->fieldOffset,v,bad);
  const bool invalid=platform->linAlg->max(nrs->meshV->Nlocal,bad,comm)>0;
  for (int c=0;c<3;++c) {
    auto component=v.slice(c*nrs->fieldOffset,nrs->meshV->Nlocal);
    const dfloat lo=platform->linAlg->min(nrs->meshV->Nlocal,component,comm);
    const dfloat hi=platform->linAlg->max(nrs->meshV->Nlocal,component,comm);
    if (platform->comm.mpiRank()==0)
      printf("method1Terms step=%d sourceTime=%.8e field=%s component=%d min=%.8e max=%.8e nonfinite=%d\n",
          nrs->tstep,sourceTime,name,c,lo,hi,int(invalid));
  }
}
inline void writeVector(const char *name,double time,deviceMemory<dfloat> v)
{
  const dlong offset=nrs->fieldOffset,N=nrs->meshV->Nlocal;
  nrs->writeToFile(name,time,nrs->meshV,{{"velocity",
      {v.slice(0,N),v.slice(offset,N),v.slice(2*offset,N)}}});
}
inline void diagnostics(double time)
{
  if (due()) {
    if (nrs->tstep>0) {
      opSEM::strongGrad(nrs->meshV,nrs->fieldOffset,nrs->fluid->o_P,o_gradPStrong);
      defectKernel(nrs->meshV->Nlocal,nrs->fieldOffset,p.rhoGas,p.rhoLiquid,
          p.virtualMassEnabled*p.virtualMassCoefficient,p.alphaFloor,nrs->g0/nrs->dt[0],
          sourceAlpha,o_ug,o_ul,historyG,historyL,sourceRhsG,rhs,terms,
          o_gradPStrong,freeG,freeL,defectG,defectL);
    }
    reportVector("UG",o_ug); reportVector("UL",o_ul);
    reportVector("deltaUG",deltaG); reportVector("deltaUL",deltaL);
    reportVector("mixtureDefectBefore",before); reportVector("mixtureDefectAfter",after);
    reportVector("physicalMomentumDefectG_free",defectG); reportVector("physicalMomentumDefectL_free",defectL);
    for (int t=0;t<6;++t) reportVector(termNames[t],terms.slice(3*t*nrs->fieldOffset,3*nrs->fieldOffset));
    fflush(stdout);
  }
  if (!nrs->checkpointStep) return;
  writeVector("ug",time,o_ug); writeVector("ul",time,o_ul);
  writeVector("deltaUg",time,deltaG); writeVector("deltaUl",time,deltaL);
  writeVector("mixBefore",time,before); writeVector("mixAfter",time,after);
  writeVector("gmr",time,defectG); writeVector("lmr",time,defectL);
  for (int t=0;t<6;++t)
    writeVector(termFiles[t],time,terms.slice(3*t*nrs->fieldOffset,3*nrs->fieldOffset));
}
} } // namespace bubbleColumn::phaseMomentum
