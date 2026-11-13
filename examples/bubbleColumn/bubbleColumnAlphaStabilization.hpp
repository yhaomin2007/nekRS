#pragma once
#include <limits>

// Case-local integration of a reusable weak anisotropic diffusion operator.
// Requires bubbleColumnTerms.hpp (and its native EXT history helper) first.
namespace bubbleColumn { namespace alphaStabilization {
static int mode=1,interval=10;
static dfloat crosswind=2e-5,streamline=2e-5,epsilon=1e-6;
static dfloat inverseLengthBound=0;
static occa::kernel fluxKernel,diagnosticKernel;
static deviceMemory<dfloat> weightedGradient,flux,rawRhs,usedRhs;
static deviceMemory<dfloat> gradient,gradientUgZ,divUg,fields,historyRate,baselineDiffusion,scratch;
static ExplicitHistory rhsHistory;
static std::ofstream csv;
inline dfloat transverse() { return mode==1 || mode==4 ? crosswind:0; }
inline dfloat longitudinal() { return mode==3 || mode==4 ? streamline:0; }
inline bool enabled() { return transverse()>0 || longitudinal()>0; }
inline bool due()
{ return nrs->tstep==1 || nrs->checkpointStep || (interval>0 && nrs->tstep%interval==0); }

inline void readParameters()
{
  auto read=[](const char *key,auto &v){platform->par->extract("casedata",key,v);};
  read("alphastabilizationmode",mode);
  read("alphacrosswinddiffusivity",crosswind);
  read("alphastreamlinediffusivity",streamline);
  read("alphadirectionepsilon",epsilon);
  read("diagnosticinterval",interval);
  nekrsCheck((mode!=0 && mode!=1 && mode!=3 && mode!=4) || interval<0 ||
      !std::isfinite(crosswind) || crosswind<0 || !std::isfinite(streamline) || streamline<0 ||
      !std::isfinite(epsilon) || epsilon<=0,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","Alpha stabilization: modes 0,1,3,4 only; diffusivities >=0 and direction epsilon >0 must be finite. "
      "Adaptive mode 2 and SUPG mode 5 are not implemented.\n");
}
inline void registerKernels(deviceKernelProperties &info)
{
  const std::string request="bubbleColumn::alphaStabilization";
  if (platform->options.compareArgs("REGISTER ONLY","TRUE"))
    platform->kernelRequests.add(request,"bubbleColumnAlphaStabilization.okl",info);
  else {
    fluxKernel=platform->kernelRequests.load(request,"alphaStabilizationFlux");
    diagnosticKernel=platform->kernelRequests.load(request,"alphaStabilizationDiagnostics");
  }
}

// Upper bound on lambda_max(M_e^-1 K_e) for unit isotropic diffusion:
// trace(M_e^-1/2 K_e M_e^-1/2). Includes GLL weights, geometry and all N+1 nodes.
// The assembled CG Rayleigh quotient is bounded by the largest element bound.
inline dfloat geometryBound(int nq,int nvgeo,int ne,
    const std::vector<dfloat> &geo,const std::vector<dfloat> &D)
{
  const int np=nq*nq*nq, stride[3]={1,nq,nq*nq};
  dfloat bound=0;
  for (int e=0;e<ne;++e) {
    auto g=[&](int n,int id){return geo[e*nvgeo*np+n+id*np];};
    dfloat trace=0;
    for (int n=0;n<np;++n) {
      const int pos[3]={n%nq,(n/nq)%nq,n/(nq*nq)};
      const dfloat mass=g(n,JWID);
      if (!(mass>0)) return std::numeric_limits<dfloat>::infinity();
      dfloat diagonal[3]={0,0,0};
      for (int d=0;d<3;++d) {
        dfloat norm=0;
        for (int c=0;c<3;++c) {
          const dfloat metric=g(n,3*d+c);
          norm+=metric*metric;
          diagonal[c]+=metric*D[pos[d]*nq+pos[d]];
        }
        for (int j=0;j<nq;++j) if (j!=pos[d]) {
          const int other=n+(j-pos[d])*stride[d];
          const dfloat otherMass=g(other,JWID),derivative=D[pos[d]*nq+j];
          if (!(otherMass>0)) return std::numeric_limits<dfloat>::infinity();
          trace+=mass*norm*derivative*derivative/otherMass;
        }
      }
      for (int c=0;c<3;++c) trace+=diagonal[c]*diagonal[c];
    }
    bound=std::max(bound,trace);
  }
  return bound;
}
inline void allocate()
{
  const dlong offset=nrs->fieldOffset;
  weightedGradient.resize(3*offset); flux.resize(3*offset);
  rawRhs.resize(offset); usedRhs.resize(offset);
  gradient.resize(3*offset); gradientUgZ.resize(3*offset);
  divUg.resize(offset); fields.resize(11*offset);
  historyRate.resize(offset); baselineDiffusion.resize(offset); scratch.resize(offset);
  platform->linAlg->fill(offset,0.0,usedRhs);
  platform->linAlg->fill(offset,0.0,historyRate);
  if (enabled()) {
    auto mesh=nrs->meshV;
    std::vector<dfloat> geo(mesh->Nelements*mesh->Nvgeo*mesh->Np),D(mesh->Nq*mesh->Nq);
    mesh->o_vgeo.copyTo(geo.data()); mesh->o_D.copyTo(D.data());
    const dfloat local=geometryBound(mesh->Nq,mesh->Nvgeo,mesh->Nelements,geo,D);
    MPI_Allreduce(&local,&inverseLengthBound,1,MPI_DFLOAT,MPI_MAX,platform->comm.mpiComm());
    nekrsCheck(!std::isfinite(inverseLengthBound),platform->comm.mpiComm(),EXIT_FAILURE,
        "%s","Alpha stabilization requires positive finite element quadrature masses.\n");
  }
  if (platform->comm.mpiRank()==0)
    printf("alphaStabilization mode=%d Dcross=%.8e Dstream=%.8e directionEpsilon=%.8e "
           "weak natural no-flux operator; added diffusion=explicit fluid EXT; geometryBound=%.8e\n",
        mode,transverse(),longitudinal(),epsilon,inverseLengthBound);
}

inline void evaluateDiffusion()
{
  auto mesh=nrs->meshV; const dlong N=mesh->Nlocal,offset=nrs->fieldOffset;
  if (!enabled()) { platform->linAlg->fill(N,0.0,rawRhs); return; }
  // Native gradientVolume returns ELEMENT-LOCAL JW*grad(alpha). Do not
  // gather/average it before applying the tensor: that would change G^T W D G.
  opSEM::strongGrad(mesh,offset,nrs->scalar->o_solution("alpha"),weightedGradient,false);
  fluxKernel(N,offset,transverse(),longitudinal(),epsilon,mesh->o_vgeo,o_ug,weightedGradient,flux);
  // Native weak divergence is +G^T W q. The diffusion RHS needs its NEGATIVE.
  // No surface load => natural zero added normal flux at Neumann walls/outlet.
  opSEM::divergence(mesh,offset,flux,rawRhs);
  oogs::startFinish(rawRhs,1,0,ogsDfloat,ogsAdd,mesh->oogs);
  platform->linAlg->axmy(N,-1.0,mesh->o_invLMM,rawRhs);
}
inline void addSources()
{
  const dlong N=nrs->meshV->Nlocal;
  // Preserve native pre-lag BDF history for the diagnostic split residual.
  if (due()) {
    platform->linAlg->fill(N,0.0,historyRate);
    std::vector<dfloat> w(nrs->scalar->o_coeffBDF.size());
    nrs->scalar->o_coeffBDF.copyTo(w.data());
    const int is=nrs->scalar->nameToIndex.at("alpha");
    for (int j=0;j<static_cast<int>(w.size());++j) {
      auto old=nrs->scalar->o_S.slice(nrs->scalar->fieldOffsetScan[is]+j*nrs->scalar->fieldOffsetSum,N);
      platform->linAlg->axpby(N,w[j]/nrs->dt[0],old,1.0,historyRate);
    }
  }
  if (!enabled()) return; // off mode never changes scalar or mixture forcing
  evaluateDiffusion();
  rhsHistory.apply(rawRhs,usedRhs,1,true);
  platform->linAlg->axpby(N,1.0,usedRhs,1.0,o_alphaSource);
  auto explicitTerms=nrs->scalar->o_explicitTerms("alpha");
  platform->linAlg->axpby(N,1.0,usedRhs,1.0,explicitTerms);
  // Existing Method1 target adds this selected explicit diffusion once, after
  // transport EXT. Do NOT extrapolate it again or add it to transport history.
  platform->linAlg->axpby(N,1.0,usedRhs,1.0,o_alphaExplicitDiffusion);
}

inline void diagnostics(double time)
{
  if (!due()) return;
  auto mesh=nrs->meshV; const dlong N=mesh->Nlocal,offset=nrs->fieldOffset;
  const auto comm=platform->comm.mpiComm();
  auto alpha=nrs->scalar->o_solution("alpha");
  reconstructGasVelocity(); evaluateDiffusion(); // current raw fields; never advances EXT history
  opSEM::strongGrad(mesh,offset,alpha,gradient,true);
  opSEM::strongGrad(mesh,offset,nrs->scalar->o_solution("ugz"),gradientUgZ,true);
  opSEM::strongDivergence(mesh,offset,o_ug,divUg,true);
  // The baseline numerical diffusion actually used the pre-clipping solution.
  launchKernel("core-weakLaplacianHex3D",mesh->Nelements,1,o_scalarFieldOffsetScan,
      mesh->o_ggeo,mesh->o_D,nrs->scalar->o_diffusionCoeff("alpha"),
      o_alphaBeforePostprocess,baselineDiffusion);
  oogs::startFinish(baselineDiffusion,1,0,ogsDfloat,ogsAdd,mesh->oogs);
  platform->linAlg->axmy(N,-1.0,mesh->o_invLMM,baselineDiffusion);
  diagnosticKernel(N,offset,transverse(),longitudinal(),epsilon,
      nrs->tstep>0?nrs->g0/nrs->dt[0]:0,alpha,o_ug,gradient,gradientUgZ,divUg,
      rawRhs,usedRhs,historyRate,o_convectionEXT,baselineDiffusion,fields);
  const dfloat lo=platform->linAlg->min(N,alpha,comm),hi=platform->linAlg->max(N,alpha,comm);
  const dfloat volume=platform->linAlg->innerProd(N,mesh->o_LMM,alpha,comm);
  const dfloat netRaw=platform->linAlg->innerProd(N,mesh->o_LMM,rawRhs,comm);
  const dfloat netUsed=platform->linAlg->innerProd(N,mesh->o_LMM,usedRhs,comm);
  platform->linAlg->axpby(N,1.0,rawRhs,0.0,scratch);
  platform->linAlg->axmy(N,1.0,alpha,scratch);
  const dfloat energy=platform->linAlg->innerProd(N,mesh->o_LMM,scratch,comm);
  const dfloat budgetDefect=platform->linAlg->innerProd(N,mesh->o_LMM,fields.slice(10*offset,N),comm);
  reconstructGasFlux();
  const dfloat advectiveBoundary=mesh->surfaceAreaNormalMultiplyVectorIntegrate(offset,o_allBoundaryIDs,o_qg);
  const dfloat theta=nrs->dt[0]*std::max(transverse(),longitudinal())*inverseLengthBound;
  if (platform->comm.mpiRank()==0) {
    printf("alphaStabilization step=%d time=%.8e alpha=[%.8e,%.8e] Vgas=%.8e "
           "netStab(raw,EXT)=[%.8e,%.8e] energyRaw=%.8e advBoundary=%.8e "
           "splitBudgetDefect=%.8e clipDeltaV=%.8e explicitDiffusionBoundDt=%.8e\n",
        nrs->tstep,time,lo,hi,volume,netRaw,netUsed,energy,advectiveBoundary,budgetDefect,alphaClipDeltaVolume,theta);
    if (!csv.is_open()) {
      std::ifstream previous("bubbleColumn_alpha_stabilization.csv");
      const bool header=previous.peek()==std::ifstream::traits_type::eof();
      csv.open("bubbleColumn_alpha_stabilization.csv",std::ios::app);
      if (header) csv<<"step,time,mode,alpha_min,alpha_max,gas_volume,net_stab_raw,net_stab_EXT,energy_raw,advective_boundary,split_budget_defect,clip_delta_volume,explicit_diffusion_bound_dt\n";
      csv<<std::scientific<<std::setprecision(16);
    }
    csv<<nrs->tstep<<','<<time<<','<<mode<<','<<lo<<','<<hi<<','<<volume<<','<<netRaw<<','<<netUsed<<','<<energy<<','
       <<advectiveBoundary<<','<<budgetDefect<<','<<alphaClipDeltaVolume<<','<<theta<<'\n';
    csv.flush(); fflush(stdout);
  }
  if (!nrs->checkpointStep) return;
  std::vector<std::tuple<std::string,std::vector<deviceMemory<dfloat>>>> list;
  for (int i=0;i<11;++i) list.push_back({"scalar"+(i<10?std::string("0"):std::string())+std::to_string(i),{fields.slice(i*offset,N)}});
  nrs->writeToFile("ast",time,mesh,list,true,true);
  nrs->writeToFile("astGrad",time,mesh,{{"velocity",{gradient.slice(0,N),gradient.slice(offset,N),gradient.slice(2*offset,N)}}},true,true);
}
} } // namespace bubbleColumn::alphaStabilization
