#pragma once

namespace bubbleColumn { namespace alphaStabilization { namespace residual {
static dfloat cwScale=1,supgScale=1,gradientChange=1e-5,maxDiffusion=1e-3;
static dfloat relativeTolerance=1e-5,absoluteTolerance=1e-7;
static int maxIterations=200,lastIterations=0;
static dfloat initialNorm=0,finalNorm=0;
static occa::kernel gradientKernel,residualFluxKernel,maskKernel;
static deviceMemory<dfloat> velocity,weightedDiv,weightedGrad,baselineFlux,weightedLap;
static deviceMemory<dfloat> cwFlux,supgFlux,cwWeak,supgWeak,totalWeak;
static deviceMemory<dfloat> baselineRhs,lambdaTime,ax,residualStrong,buffer,fields,displayGradient;
static deviceMemory<dfloat> baseResidual,operatorVector,linearStep,coefficients,zeroRhs;
static std::vector<deviceMemory<dfloat>> krylov;
static bool coefficientsFrozen=false;
static int lastCycles=0;
static dfloat projectedNorm=0;
static std::ofstream csv;
inline bool active() { return mode==2 || mode==5 || mode==6; }
inline void readParameters()
{
  auto read=[](const char *key,auto &v){platform->par->extract("casedata",key,v);};
  read("alphacrosswindscale",cwScale);read("alphastreamlinescale",supgScale);
  read("alphalowergradientchange",gradientChange);read("alphamaxresidualdiffusivity",maxDiffusion);
  read("alphastabilizationreltol",relativeTolerance);read("alphastabilizationabstol",absoluteTolerance);
  read("alphastabilizationmaxiterations",maxIterations);
  nekrsCheck(!std::isfinite(cwScale)||cwScale<0||!std::isfinite(supgScale)||supgScale<0||
      !std::isfinite(gradientChange)||gradientChange<=0||!std::isfinite(maxDiffusion)||maxDiffusion<=0||
      !std::isfinite(relativeTolerance)||relativeTolerance<=0||!std::isfinite(absoluteTolerance)||absoluteTolerance<=0||
      maxIterations<1,platform->comm.mpiComm(),EXIT_FAILURE,"%s","Invalid alpha residual stabilization parameters.\n");
}
inline void registerKernels(deviceKernelProperties &info)
{
  const std::string request="bubbleColumn::alphaResidual";
  if(platform->options.compareArgs("REGISTER ONLY","TRUE"))
    platform->kernelRequests.add(request,"bubbleColumnAlphaResidual.okl",info);
  else {
    gradientKernel=platform->kernelRequests.load(request,"alphaResidualGradient");
    residualFluxKernel=platform->kernelRequests.load(request,"alphaResidualFlux");
    maskKernel=platform->kernelRequests.load(request,"alphaResidualMask");
  }
}
inline void allocate()
{
  if(!active())return;
  const dlong o=nrs->fieldOffset;
  velocity.resize(3*o);weightedDiv.resize(o);weightedGrad.resize(3*o);
  baselineFlux.resize(3*o);weightedLap.resize(o);
  cwFlux.resize(3*o);supgFlux.resize(3*o);
  for(auto *v:{&cwWeak,&supgWeak,&totalWeak,&baselineRhs,&lambdaTime,&ax,&residualStrong,&buffer,&baseResidual,&operatorVector,&linearStep,&zeroRhs})v->resize(o);
  fields.resize(12*o);displayGradient.resize(3*o);coefficients.resize(2*o);
  platform->linAlg->fill(o,0.0,zeroRhs);
  krylov.resize(21);for(auto &v:krylov)v.resize(o);
  platform->linAlg->fill(12*o,0.0,fields);
  if(platform->comm.mpiRank()==0)
    printf("alphaResidual: mode=%d Codina-type crosswind=%d SUPG=%d; current BDF/EXT residual, "
           "predictor-lagged coefficients, linear alpha-only GMRES; no constant added viscosity.\n",mode,mode!=5,mode!=2);
}
inline void beginStep()
{
  if(!active())return;
  const dlong N=nrs->meshV->Nlocal,o=nrs->fieldOffset;
  velocity.copyFrom(o_ug,3*o); // source-time gas velocity, never native UM
  opSEM::strongDivergence(nrs->meshV,o,velocity,weightedDiv,false);
  platform->linAlg->fill(N,0.0,historyRate);
  std::vector<dfloat>w(nrs->scalar->o_coeffBDF.size());nrs->scalar->o_coeffBDF.copyTo(w.data());
  const int is=nrs->scalar->nameToIndex.at("alpha");
  for(int j=0;j<static_cast<int>(w.size());++j) {
    auto old=nrs->scalar->o_S.slice(nrs->scalar->fieldOffsetScan[is]+j*nrs->scalar->fieldOffsetSum,N);
    platform->linAlg->axpby(N,w[j]/nrs->dt[0],old,1.0,historyRate);
  }
}
inline void buildWeak(const deviceMemory<dfloat> &alpha,bool homogeneous=false)
{
  auto mesh=nrs->meshV;const dlong N=mesh->Nlocal,o=nrs->fieldOffset;
  auto D=nrs->scalar->o_diffusionCoeff("alpha");
  opSEM::strongGrad(mesh,o,alpha,weightedGrad,false);
  gradientKernel(N,o,mesh->o_vgeo,weightedGrad,D,baselineFlux);
  opSEM::strongDivergence(mesh,o,baselineFlux,weightedLap,false);
  residualFluxKernel(N,o,nrs->g0/nrs->dt[0],nrs->dt[0],mesh->N,mode!=5,mode!=2,
      cwScale,supgScale,gradientChange,maxDiffusion,epsilon,int(coefficientsFrozen),coefficients,
      mesh->o_vgeo,alpha,velocity,weightedGrad,weightedLap,weightedDiv,
      homogeneous?zeroRhs:historyRate,homogeneous?zeroRhs:o_convectionEXT,D,cwFlux,supgFlux,fields);
  opSEM::divergence(mesh,o,cwFlux,cwWeak);
  opSEM::divergence(mesh,o,supgFlux,supgWeak);
  totalWeak.copyFrom(cwWeak,N);platform->linAlg->axpby(N,1.0,supgWeak,1.0,totalWeak);
}
inline void evaluate(){buildWeak(nrs->scalar->o_solution("alpha"));}
// Linear stabilized Jacobian with the same frozen coefficients. Histories and
// known transport are zero ONLY in this homogeneous operator action.
inline void apply(const deviceMemory<dfloat> &v,deviceMemory<dfloat> &out)
{
  auto mesh=nrs->meshV;const dlong N=mesh->Nlocal;
  const int is=nrs->scalar->nameToIndex.at("alpha");
  auto elliptic=nrs->scalar->ellipticSolver[is];
  auto D=nrs->scalar->o_diffusionCoeff("alpha");
  buildWeak(v,true);
  elliptic->Ax(D,lambdaTime,v,out);
  platform->linAlg->axpby(N,1.0,totalWeak,1.0,out);
  oogs::startFinish(out,1,0,ogsDfloat,ogsAdd,mesh->oogs);
  platform->linAlg->axmy(N,1.0,mesh->o_invLMM,out);
  if(elliptic->Nmasked())maskKernel(elliptic->Nmasked(),elliptic->o_maskIds(),out);
}
inline dfloat norm()
{
  auto mesh=nrs->meshV;const dlong N=mesh->Nlocal;
  auto alpha=nrs->scalar->o_solution("alpha");
  const int is=nrs->scalar->nameToIndex.at("alpha");
  auto elliptic=nrs->scalar->ellipticSolver[is];
  auto diffusion=nrs->scalar->o_diffusionCoeff("alpha");
  elliptic->Ax(diffusion,lambdaTime,alpha,ax); // unassembled element Helmholtz action
  residualStrong.copyFrom(ax,N);
  platform->linAlg->axpby(N,-1.0,baselineRhs,1.0,residualStrong);
  platform->linAlg->axpby(N,1.0,totalWeak,1.0,residualStrong);
  oogs::startFinish(residualStrong,1,0,ogsDfloat,ogsAdd,mesh->oogs);
  platform->linAlg->axmy(N,1.0,mesh->o_invLMM,residualStrong);
  if(elliptic->Nmasked())maskKernel(elliptic->Nmasked(),elliptic->o_maskIds(),residualStrong);
  buffer.copyFrom(residualStrong,N);platform->linAlg->axmy(N,1.0,residualStrong,buffer);
  return std::sqrt(platform->linAlg->innerProd(N,mesh->o_LMM,buffer,platform->comm.mpiComm())/mesh->volume);
}

// LMM holds element quadrature weights; duplicated shared-node values therefore
// participate with their local weight, exactly as in native volume integrals.
inline dfloat dot(const deviceMemory<dfloat> &a,const deviceMemory<dfloat> &b)
{
  const auto N=nrs->meshV->Nlocal;
  buffer.copyFrom(a,N);platform->linAlg->axmy(N,1.0,b,buffer);
  return platform->linAlg->innerProd(N,nrs->meshV->o_LMM,buffer,platform->comm.mpiComm())/nrs->meshV->volume;
}
inline dfloat length(const deviceMemory<dfloat> &a){return std::sqrt(std::max(dfloat(0),dot(a,a)));}

// One restarted linear GMRES cycle. A/h is applied directly, without finite
// differences, perturbing alpha, differentiating |R|, or changing the sensor.
// All basis vectors have zero Dirichlet rows; the fixed inlet is preserved.
inline void direction(int budget,dfloat tolerance)
{
  const dlong N=nrs->meshV->Nlocal;
  const dfloat beta=length(baseResidual),h=nrs->g0/nrs->dt[0];
  krylov[0].copyFrom(baseResidual,N);platform->linAlg->scale(N,-1/beta,krylov[0]);
  dfloat H[21][20]={},cs[20]={},sn[20]={},g[21]={},y[20]={};g[0]=beta;
  int count=0;
  for(int j=0;j<std::min(20,budget);++j) {
    apply(krylov[j],operatorVector);
    platform->linAlg->scale(N,1/h,operatorVector);
    for(int pass=0;pass<2;++pass)for(int i=0;i<=j;++i) {
      const dfloat c=dot(krylov[i],operatorVector);H[i][j]+=c;
      platform->linAlg->axpby(N,-c,krylov[i],1.0,operatorVector);
    }
    H[j+1][j]=length(operatorVector);
    const bool breakdown=H[j+1][j]<1e-12;
    if(!breakdown) {
      krylov[j+1].copyFrom(operatorVector,N);
      platform->linAlg->scale(N,1/H[j+1][j],krylov[j+1]);
    }
    for(int i=0;i<j;++i) {
      const dfloat t=cs[i]*H[i][j]+sn[i]*H[i+1][j];
      H[i+1][j]=-sn[i]*H[i][j]+cs[i]*H[i+1][j];H[i][j]=t;
    }
    const dfloat r=std::hypot(H[j][j],H[j+1][j]);
    nekrsCheck(!std::isfinite(r)||r<1e-30,platform->comm.mpiComm(),EXIT_FAILURE,"%s",
        "Alpha linear GMRES breakdown with frozen residual coefficients.\n");
    cs[j]=H[j][j]/r;sn[j]=H[j+1][j]/r;H[j][j]=r;H[j+1][j]=0;
    g[j+1]=-sn[j]*g[j];g[j]*=cs[j];count=j+1;
    if(std::abs(g[j+1])<=tolerance || breakdown)break;
  }
  for(int i=count-1;i>=0;--i) {
    y[i]=g[i];for(int j=i+1;j<count;++j)y[i]-=H[i][j]*y[j];y[i]/=H[i][i];
  }
  platform->linAlg->fill(N,0.0,linearStep);
  for(int i=0;i<count;++i)platform->linAlg->axpby(N,y[i]/h,krylov[i],1.0,linearStep);
  lastIterations+=count;projectedNorm=std::abs(g[count]);
}
inline void solve(double time,int)
{
  if(!active())return;
  auto mesh=nrs->meshV;const dlong N=mesh->Nlocal,o=nrs->fieldOffset;
  nekrsCheck(nrs->scalar->userImplicitLinearTerm || platform->app->bc->hasRobin("SCALAR00"),
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","Residual alpha stabilization requires the existing unit-transport, non-Robin alpha Helmholtz equation.\n");
  static bool checkedTransport=false;
  if(!checkedTransport) {
    auto rho=nrs->scalar->o_transportCoeff("alpha");
    const auto comm=platform->comm.mpiComm();
    const dfloat low=platform->linAlg->min(N,rho,comm),high=platform->linAlg->max(N,rho,comm);
    nekrsCheck(!std::isfinite(low)||!std::isfinite(high)||std::abs(low-1)>1e-14||std::abs(high-1)>1e-14,comm,EXIT_FAILURE,"%s",
        "Residual alpha stabilization requires transportCoeff=1.\n");
    checkedTransport=true;
  }
  const int is=nrs->scalar->nameToIndex.at("alpha");
  auto nativeRhs=deviceMemory<dfloat>(nrs->scalar->o_JwF).slice(nrs->scalar->fieldOffsetScan[is],N);
  baselineRhs.copyFrom(nativeRhs,N);
  auto alpha=nrs->scalar->o_solution("alpha");
  platform->linAlg->fill(N,nrs->g0/nrs->dt[0],lambdaTime);
  // Compute the nonlinear sensor once on the native predictor, then lag it.
  // It remains spatially varying and is refreshed at EVERY timestep.
  coefficientsFrozen=false;evaluate();
  coefficients.copyFrom(fields,N,0,4*o);coefficients.copyFrom(fields,N,o,5*o);
  fields.copyFrom(fields,N,11*o,8*o); // predictor residual that selected Dcw
  coefficientsFrozen=true;
  initialNorm=norm();finalNorm=initialNorm;lastIterations=0;lastCycles=0;projectedNorm=0;
  const dfloat tolerance=absoluteTolerance+relativeTolerance*initialNorm;
  while(std::isfinite(finalNorm) && finalNorm>tolerance && lastIterations<maxIterations) {
    baseResidual.copyFrom(residualStrong,N);
    direction(maxIterations-lastIterations,tolerance*0.5);
    platform->linAlg->axpby(N,1.0,linearStep,1.0,alpha);
    evaluate();finalNorm=norm();++lastCycles;
  }
  nekrsCheck(!std::isfinite(finalNorm)||finalNorm>tolerance,platform->comm.mpiComm(),EXIT_FAILURE,
      "Alpha frozen-coefficient linear solve failed: step=%d mode=%d GMRES=%d cycles=%d "
      "initialRms=%.8e trueRms=%.8e projectedRms=%.8e target=%.8e dt=%.8e. "
      "No unconverged alpha is accepted.\n",nrs->tstep,mode,lastIterations,lastCycles,
      initialNorm,finalNorm,projectedNorm,tolerance,nrs->dt[0]);
  // Store exactly the final accepted conservative weak source, with NO EXT.
  usedRhs.copyFrom(totalWeak,N);oogs::startFinish(usedRhs,1,0,ogsDfloat,ogsAdd,mesh->oogs);
  platform->linAlg->axmy(N,-1.0,mesh->o_invLMM,usedRhs);
  platform->linAlg->axpby(N,1.0,usedRhs,1.0,o_alphaExplicitDiffusion);
  for(int i=0;i<2;++i) {
    auto f=fields.slice((6+i)*o,N);f.copyFrom(i?supgWeak:cwWeak,N);
    oogs::startFinish(f,1,0,ogsDfloat,ogsAdd,mesh->oogs);
    platform->linAlg->axmy(N,-1.0,mesh->o_invLMM,f);
  }
  fields.copyFrom(usedRhs,N,9*o,0);fields.copyFrom(residualStrong,N,10*o,0);
  if(due()&&platform->comm.mpiRank()==0)
    printf("alphaResidualSolve step=%d time=%.8e solver=lagged-linear GMRES=%d cycles=%d weakRms(initial,final,target)=[%.8e,%.8e,%.8e]\n",
        nrs->tstep,time,lastIterations,lastCycles,initialNorm,finalNorm,tolerance);
}
inline void diagnostics(double time)
{
  if(!active()||nrs->tstep<1||!due())return;
  auto mesh=nrs->meshV;const dlong N=mesh->Nlocal,o=nrs->fieldOffset;const auto comm=platform->comm.mpiComm();
  auto alpha=nrs->scalar->o_solution("alpha"),dcw=fields.slice(4*o,N),tau=fields.slice(5*o,N);
  const dfloat dmax=platform->linAlg->max(N,dcw,comm),tmax=platform->linAlg->max(N,tau,comm);
  const dfloat volume=platform->linAlg->innerProd(N,mesh->o_LMM,alpha,comm);
  const dfloat net=platform->linAlg->innerProd(N,mesh->o_LMM,usedRhs,comm);
  if(platform->comm.mpiRank()==0) {
    printf("alphaResidual step=%d DcwMax=%.8e tauMax=%.8e Vgas=%.8e netWeakStabilization=%.8e clipDeltaV=%.8e\n",
        nrs->tstep,dmax,tmax,volume,net,alphaClipDeltaVolume);
    if(!csv.is_open()) {
      std::ifstream prior("bubbleColumn_alpha_residual.csv");const bool header=prior.peek()==std::ifstream::traits_type::eof();
      csv.open("bubbleColumn_alpha_residual.csv",std::ios::app);
      if(header)csv<<"step,time,mode,iterations,initial_weak_rms,final_weak_rms,Dcw_max,tau_max,gas_volume,net_stabilization,clip_delta_volume\n";
      csv<<std::scientific<<std::setprecision(16);
    }
    csv<<nrs->tstep<<','<<time<<','<<mode<<','<<lastIterations<<','<<initialNorm<<','<<finalNorm<<','<<dmax<<','<<tmax<<','<<volume<<','<<net<<','<<alphaClipDeltaVolume<<'\n';csv.flush();
  }
  if(!nrs->checkpointStep)return;
  std::vector<std::tuple<std::string,std::vector<deviceMemory<dfloat>>>> list;
  for(int i=0;i<12;++i)list.push_back({"scalar"+(i<10?std::string("0"):std::string())+std::to_string(i),{fields.slice(i*o,N)}});
  nrs->writeToFile("ars",time,mesh,list,true,true);
  auto preclipAlpha=fields.slice(0,N);opSEM::strongGrad(mesh,o,preclipAlpha,displayGradient,true);
  nrs->writeToFile("arsGrad",time,mesh,{{"velocity",{displayGradient.slice(0,N),displayGradient.slice(o,N),displayGradient.slice(2*o,N)}}},true,true);
}
} } } // bubbleColumn::alphaStabilization::residual
