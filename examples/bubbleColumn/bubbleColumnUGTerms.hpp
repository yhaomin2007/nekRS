#pragma once
#include <limits>

// Read-only snapshots of the equation actually assembled for this step.
// All equation fields are accelerations; masks and coefficients have separate files.
namespace bubbleColumn { namespace ugTerms {
enum Term {
  Pressure=0, Convection=1, Gravity=2, DragLiquid=3, VmLiquid=4, GasStress=5,
  DiffusionComp=6, UserRhs=7, NativeRegularization=8, DragImplicit=9,
  DiffusionImplicit=10, TimeDerivative=11, RhsSum=12, Balance=13,
  SourceDefect=14, PostprocessRate=15, ConvectionRaw=16, DragNet=17,
  TimeDiagonal=18, TimeHistory=19, DragGasExplicit=20, Count=21
};
static const char *names[Count] = {
  "pressure", "convection_EXT", "gravity", "drag_liquid", "VM_liquid",
  "gas_stress", "diffusion_compensation", "user_rhs", "native_regularization",
  "drag_implicit", "diffusion_implicit_weak", "BDF_time", "rhs_sum",
  "balance_free", "source_defect", "postprocess_rate", "convection_raw",
  "drag_net", "BDF_diagonal", "BDF_history", "drag_gas_explicit"
};
static const char *prefixes[Count] = {
  "gpr","gcv","ggr","gdl","gvm","gvs","gdc","gus","ghp","gdi",
  "gnd","gdt","grs","grb","gse","gpc","gcr","gdn","gbd","gbh","gde"
};
static int enabled=1, interval=10, sourceStep=-1, solvedStep=-1;
static bool freeMaskReady=false;
static double sourceTime=0.0;
static deviceMemory<dfloat> terms, solved, lambdaSource, diffusion, nativeExplicit, freeMask, bad, coefficients;
static occa::kernel captureKernel, finishKernel, postprocessKernel, maskKernel, nonfiniteKernel;
static std::ofstream csv;

inline void registerKernels(deviceKernelProperties &info)
{
  const std::string request="bubbleColumn::ugTerms";
  if(platform->options.compareArgs("REGISTER ONLY","TRUE")) {
    platform->kernelRequests.add(request,"bubbleColumnUGTerms.okl",info);
  } else {
    captureKernel=platform->kernelRequests.load(request,"ugCaptureTerms");
    finishKernel=platform->kernelRequests.load(request,"ugFinishTerms");
    postprocessKernel=platform->kernelRequests.load(request,"ugPostprocessTerm");
    maskKernel=platform->kernelRequests.load(request,"ugTermFreeMask");
    nonfiniteKernel=platform->kernelRequests.load(request,"ugTermNonfinite");
  }
}
inline deviceMemory<dfloat> field(int term)
{ return terms.slice(3 * term * nrs->fieldOffset, 3 * nrs->fieldOffset); }

// Match the assembled weak equation when native HPF/source loads differ
// between element-local copies of a shared node.
inline void assembleVector(deviceMemory<dfloat> &vector)
{
  auto mesh=nrs->meshV;const dlong offset=nrs->fieldOffset;
  platform->linAlg->axmyMany(mesh->Nlocal,3,offset,0,1.0,mesh->o_LMM,vector);
  oogs::startFinish(vector,3,offset,ogsDfloat,ogsAdd,mesh->oogs);
  platform->linAlg->axmyMany(mesh->Nlocal,3,offset,0,1.0,mesh->o_invLMM,vector);
}

inline bool due()
{ return nrs->tstep==1 || (interval>0 && nrs->tstep%interval==0) || nrs->checkpointStep; }

inline void statistics(int term, const char *stage, double time)
{
  if(!enabled || !due()) return;
  const auto comm=platform->comm.mpiComm();
  const dlong N=nrs->meshV->Nlocal, offset=nrs->fieldOffset;
  auto vector=field(term);
  nonfiniteKernel(N,offset,vector,bad);
  const bool invalid=platform->linAlg->max(N,bad,comm)!=0.0;
  dfloat lo[3],hi[3];
  for(int c=0;c<3;++c) {
    auto component=vector.slice(c*offset,N);
    lo[c]=invalid?std::numeric_limits<dfloat>::quiet_NaN():platform->linAlg->min(N,component,comm);
    hi[c]=invalid?std::numeric_limits<dfloat>::quiet_NaN():platform->linAlg->max(N,component,comm);
  }
  if(platform->comm.mpiRank()!=0) return;
  printf("UGTERMS step=%d stage=%s term=%s x=[%.8e,%.8e] y=[%.8e,%.8e] "
         "z=[%.8e,%.8e] nonfinite=%d\n",
      nrs->tstep,stage,names[term],lo[0],hi[0],lo[1],hi[1],lo[2],hi[2],int(invalid));
  if(!csv.is_open()) {
    std::ifstream old("bubbleColumn_ug_terms.csv");
    const bool header=old.peek()==std::ifstream::traits_type::eof();
    csv.open("bubbleColumn_ug_terms.csv",std::ios::app);
    if(header) csv<<"step,time,source_time,stage,term,component,min,max,nonfinite\n";
    csv<<std::scientific<<std::setprecision(16);
  }
  const char *components[]={"x","y","z"};
  for(int c=0;c<3;++c)
    csv<<nrs->tstep<<','<<time<<','<<sourceTime<<','<<stage<<','<<names[term]<<','
       <<components[c]<<','<<lo[c]<<','<<hi[c]<<','<<int(invalid)<<'\n';
  csv.flush();
  fflush(stdout);
}

inline void configure()
{
  platform->par->extract("casedata","ugtermdiagnosticsenabled",enabled);
  platform->par->extract("casedata","ugtermdiagnosticsinterval",interval);
  nekrsCheck((enabled!=0&&enabled!=1)||interval<0,
      platform->comm.mpiComm(),EXIT_FAILURE,"%s",
      "UG diagnostics require enabled=0/1 and a nonnegative interval.\n");
  if(!enabled) return;
  const dlong offset=nrs->fieldOffset;
  terms.resize(Count*3*offset); solved.resize(3*offset);
  diffusion.resize(3*offset); nativeExplicit.resize(3*offset);
  freeMask.resize(3*offset); lambdaSource.resize(offset); bad.resize(offset); coefficients.resize(2*offset);
  platform->linAlg->fill(terms.size(),0.0,terms);
  platform->linAlg->fill(3*offset,1.0,freeMask);
  // UDF_Setup runs before scalar->setupEllipticSolver(). Allocate only;
  // solver mask pointers are not valid until time integration starts.
  freeMaskReady=false;
  if(platform->comm.mpiRank()==0)
    printf("UGTERMS enabled: acceleration units; source snapshots + pre-postprocess "
           "solved terms; interval=%d, first step and every checkpoint. "
           "Balance uses free weak-equation rows, current zero-Neumann UG BCs.\n",interval);
}

inline void initializeFreeMask()
{
  if(!enabled || freeMaskReady) return;
  const auto comm=platform->comm.mpiComm();
  const dlong offset=nrs->fieldOffset;
  const char *gasNames[]={"ugx","ugy","ugz"};
  // Validate every pointer before dereferencing any solver. This makes an
  // accidental early call fail with a useful message rather than SIGSEGV.
  for(const char *name:gasNames) {
    const int is=nrs->scalar->nameToIndex.at(name);
    nekrsCheck(is<0 || static_cast<size_t>(is)>=nrs->scalar->ellipticSolver.size(),
        comm,EXIT_FAILURE,"%s","UG diagnostic scalar solver table is not ready.\n");
    nekrsCheck(!nrs->scalar->ellipticSolver[is],comm,EXIT_FAILURE,
        "UG diagnostic elliptic solver for %s is not ready; initialize masks after scalar solver setup.\n",name);
  }
  platform->linAlg->fill(3*offset,1.0,freeMask);
  for(int c=0;c<3;++c) {
    const int is=nrs->scalar->nameToIndex.at(gasNames[c]);
    const dlong count=nrs->scalar->ellipticSolver[is]->Nmasked();
    if(count) {
      auto component=freeMask.slice(c*offset,offset);
      maskKernel(count,nrs->scalar->ellipticSolver[is]->o_maskIds(),component);
    }
  }
  freeMaskReady=true;
}

inline void captureSource(double time)
{
  if(!enabled || nrs->tstep<1) return;
  const dlong N=nrs->meshV->Nlocal,offset=nrs->fieldOffset;
  sourceTime=time; sourceStep=nrs->tstep;
  lambdaSource.copyFrom(o_dragLambda,N);
  // Native scalar EXT1 carries these selected sources. Copy only physical nodes.
  const char *gasNames[]={"ugx","ugy","ugz"};
  for(int c=0;c<3;++c)
    nativeExplicit.copyFrom(nrs->scalar->o_explicitTerms(gasNames[c]),N,c*offset,0);
  const dfloat virtualInertia=p.virtualMassEnabled*p.virtualMassCoefficient*p.rhoLiquid;
  coefficients.copyFrom(nrs->scalar->o_solution("alpha"),N);
  auto inertia=coefficients.slice(offset,N);
  platform->linAlg->fill(N,p.rhoGas+virtualInertia,inertia);
  captureKernel(N,offset,p.rhoGas,p.rhoGas+virtualInertia,virtualInertia,
      p.gasPressureEnabled,p.alphaFloor,p.gravity[0],p.gravity[1],p.gravity[2],
      p.subtractUgDiffusion[0],p.subtractUgDiffusion[1],p.subtractUgDiffusion[2],
      nrs->scalar->o_solution("alpha"),o_gradP,lambdaSource,o_ul,o_ug,
      o_liquidAccelerationEXT,o_divGasStress,o_ugDiffusionDivergence,
      o_convectionEXT,o_convectionSample,nativeExplicit,terms);
  // Cache BDF history BEFORE native lagSolution. Computing it afterwards from
  // o_S would lose the oldest slot at the configured BDF order.
  auto history=field(TimeHistory);
  platform->linAlg->fill(3*offset,0.0,history);
  std::vector<dfloat> weights(nrs->scalar->o_coeffBDF.size());
  nrs->scalar->o_coeffBDF.copyTo(weights.data());
  for(int c=0;c<3;++c) {
    const int is=nrs->scalar->nameToIndex.at(gasNames[c]);
    auto output=history.slice(c*offset,N);
    for(size_t h=0;h<weights.size();++h) {
      auto input=nrs->scalar->o_S.slice(
          h*nrs->scalar->fieldOffsetSum+nrs->scalar->fieldOffsetScan[is],N);
      platform->linAlg->axpby(N,weights[h]/nrs->dt[0],input,1.0,output);
    }
  }
  for(int term:{Pressure,Convection,Gravity,DragLiquid,VmLiquid,GasStress,
      DiffusionComp,UserRhs,SourceDefect,ConvectionRaw,TimeHistory,DragGasExplicit})
  {
    auto vector=field(term);
    assembleVector(vector);
    statistics(term,"source",time);
  }
}

inline void captureSolved(double)
{
  if(!enabled || nrs->tstep<1) return;
  nekrsCheck(sourceStep!=nrs->tstep,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","UG diagnostics missing matching source snapshot.\n");
  // First postScalar call: native scalar elliptic solvers now exist and UG
  // has just been solved. Fixed mesh/BCs let us retain this mask thereafter.
  initializeFreeMask();
  const dlong N=nrs->meshV->Nlocal,offset=nrs->fieldOffset;
  const char *gasNames[]={"ugx","ugy","ugz"};
  for(int c=0;c<3;++c) {
    solved.copyFrom(nrs->scalar->o_solution(gasNames[c]),N,c*offset,0);
    // Native makeExplicit has now added HPF/other regularization to slot zero.
    // makeForcing shifts histories but leaves that slot unchanged (EXT1 carrier).
    nativeExplicit.copyFrom(nrs->scalar->o_explicitTerms(gasNames[c]),N,c*offset,0);
    auto component=diffusion.slice(c*offset,offset);
    launchKernel("core-weakLaplacianHex3D",nrs->meshV->Nelements,1,
        o_scalarFieldOffsetScan,nrs->meshV->o_ggeo,nrs->meshV->o_D,
        nrs->scalar->o_diffusionCoeff(gasNames[c]),
        nrs->scalar->o_solution(gasNames[c]),component);
    oogs::startFinish(component,1,0,ogsDfloat,ogsAdd,nrs->meshV->oogs);
    platform->linAlg->axmy(N,-1.0,nrs->meshV->o_invLMM,component);
  }
  assembleVector(nativeExplicit);
  finishKernel(N,offset,nrs->g0/nrs->dt[0],solved,lambdaSource,
      nativeExplicit,diffusion,freeMask,terms);
  solvedStep=nrs->tstep;
}

inline void capturePostprocessed(double time)
{
  if(!enabled || nrs->tstep<1) return;
  postprocessKernel(nrs->meshV->Nlocal,nrs->fieldOffset,nrs->g0/nrs->dt[0],
      solved,nrs->scalar->o_solution("ugx"),nrs->scalar->o_solution("ugy"),
      nrs->scalar->o_solution("ugz"),terms);
  for(int term:{NativeRegularization,DragImplicit,DiffusionImplicit,
      TimeDerivative,RhsSum,Balance,PostprocessRate,DragNet,TimeDiagonal})
    statistics(term,"solved",time);
}

inline void writeCheckpoint(double time)
{
  if(!enabled || nrs->tstep<1 || !nrs->checkpointStep) return;
  nekrsCheck(sourceStep!=nrs->tstep || solvedStep!=nrs->tstep,
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","UG diagnostic snapshot step mismatch.\n");
  const dlong N=nrs->meshV->Nlocal,offset=nrs->fieldOffset;
  for(int term=0;term<Count;++term) {
    statistics(term,"checkpoint",time);
    auto vector=field(term);
    std::vector<deviceMemory<dfloat>> components={
      vector.slice(0*offset,N),vector.slice(1*offset,N),vector.slice(2*offset,N)};
    nrs->writeToFile(prefixes[term],time,nrs->meshV,{{"velocity",components}});
  }
  std::vector<deviceMemory<dfloat>> mask={
      freeMask.slice(0*offset,N),freeMask.slice(1*offset,N),freeMask.slice(2*offset,N)};
  nrs->writeToFile("gmk",time,nrs->meshV,{{"velocity",mask}});
  std::vector<deviceMemory<dfloat>> alpha={coefficients.slice(0,N)};
  std::vector<deviceMemory<dfloat>> lambda={lambdaSource.slice(0,N)};
  std::vector<deviceMemory<dfloat>> inertia={coefficients.slice(offset,N)};
  nrs->writeToFile("gcf",time,nrs->meshV,
      {{"scalar00",alpha},{"scalar01",lambda},{"scalar02",inertia}});
}
} } // namespace bubbleColumn::ugTerms
