#pragma once
#include "mesh3D.h"
#include <limits>

namespace bubbleColumn { namespace alphaDG {
static dfloat cfl=0.1,backflow=0.0;
static int maxSubsteps=10000,limiter=1;
static dlong nf=0;
static ogs_t *faceOgs=nullptr;
static deviceMemory<dfloat> solution,old,flux,rhs,trace,traceSum,surface,boundary;
static deviceMemory<dfloat> rate,badMean,limited;
static deviceMemory<dlong> incidence;
static deviceMemory<int> boundaryIds;
static occa::kernel volumeKernel,traceKernel,surfaceKernel,liftKernel,rkKernel,rateKernel,limitKernel;
static dfloat initialVolume=0,boundaryIntegral=0,lastBalance=0,lastProjectionError=0;
static dfloat lastCfl=0,lastLimited=0;
static int lastSubsteps=0;

inline void configure()
{
  platform->par->extract("casedata","alphadgcfl",cfl);
  platform->par->extract("casedata","alphadgmaxsubsteps",maxSubsteps);
  platform->par->extract("casedata","alphadglimiter",limiter);
  platform->par->extract("casedata","alphadgbackflowvalue",backflow);
  const auto comm=platform->comm.mpiComm();
  nekrsCheck(!std::isfinite(cfl) || cfl<=0 || cfl>0.5 || maxSubsteps<1 ||
      (limiter!=0 && limiter!=1) || !std::isfinite(backflow) || backflow<0 || backflow>1,
      comm,EXIT_FAILURE,"%s","Invalid alpha DG parameters.\n");
  nekrsCheck(!platform->options.compareArgs("BDF ORDER","1"),comm,EXIT_FAILURE,
      "%s","First DG prototype requires tombo1/BDF1 for consistent split pressure target.\n");
  nekrsCheck(!platform->options.getArgs("RESTART FILE NAME").empty(),comm,EXIT_FAILURE,
      "%s","DG prototype requires a fresh start: CG checkpoints do not contain DG state.\n");
  int requestedLD=0;
  platform->par->extract("casedata","laminardispersionenabled",requestedLD);
  nekrsCheck(requestedLD!=0,comm,EXIT_FAILURE,"%s","LD is disabled in bubbleColumn_dg.\n");
}
inline void registerKernels(deviceKernelProperties &info)
{
  const std::string request="bubbleColumn-dg-alpha";
  if(platform->options.compareArgs("REGISTER ONLY","TRUE"))
    platform->kernelRequests.add(request,"alphaDG.okl",info);
  else {
    volumeKernel=platform->kernelRequests.load(request,"dgVolumeFlux");
    traceKernel=platform->kernelRequests.load(request,"dgTrace");
    surfaceKernel=platform->kernelRequests.load(request,"dgSurfaceFlux");
    liftKernel=platform->kernelRequests.load(request,"dgLift");
    rkKernel=platform->kernelRequests.load(request,"dgRK");
    rateKernel=platform->kernelRequests.load(request,"dgRate");
    limitKernel=platform->kernelRequests.load(request,"dgLimit");
  }
}
inline dfloat volume(const occa::memory &a)
{
  return platform->linAlg->innerProd(nrs->meshV->Nlocal,nrs->meshV->o_LMM,a,
      platform->comm.mpiComm());
}
inline void project()
{
  auto a=nrs->scalar->o_solution("alpha");
  a.copyFrom(solution,nrs->meshV->Nlocal);
  assembleDivergence(a); // M_CG^-1 Q^T M_element alphaDG
  // Never impose scalar Dirichlet values or clip this projection afterwards:
  // those operations would change its volume and the pressure target.
  lastProjectionError=volume(a)-volume(solution);
}
inline void allocate()
{
  auto m=nrs->meshV;const auto comm=platform->comm.mpiComm();
  const dlong N=m->Nlocal,offset=nrs->fieldOffset;
  nf=m->Nelements*m->Nfaces*m->Nfp;
  nekrsCheck(m->Nfaces!=6 || !m->globalFaceIds || !m->vmapM ||
      nf>std::numeric_limits<int>::max(),comm,EXIT_FAILURE,
      "%s","DG requires conforming hex face IDs and a supported local face count.\n");
  const auto d=nrs->scalar->o_diffusionCoeff("alpha");
  nekrsCheck(platform->linAlg->max(N,d,comm)!=0 || platform->linAlg->min(N,d,comm)!=0,
      comm,EXIT_FAILURE,"%s","DG alpha diffusion must be zero; no DG viscous operator is implemented.\n");
  solution.resize(offset);old.resize(offset);rhs.resize(offset);flux.resize(3*offset);
  trace.resize(2*nf);traceSum.resize(2*nf);surface.resize(nf);boundary.resize(nf);
  rate.resize(offset);badMean.resize(m->Nelements);limited.resize(m->Nelements);
  boundaryIds.resize(m->Nelements*6);boundaryIds.copyFrom(m->EToB,m->Nelements*6);
  // Face-interior expanded-grid IDs distinguish the different faces sharing
  // a vertex/edge. Ordinary globalIds would incorrectly mix 4 or 8 traces.
  faceOgs=ogsSetup(nf,m->globalFaceIds,comm,0,platform->device.occaDevice());
  std::vector<dfloat> multiplicity(nf,1.0),sg(nf*Nsgeo),sx(nf),sy(nf),sz(nf),sw(nf);
  m->o_sgeo.copyTo(sg.data());
  for(dlong f=0;f<nf;++f) {
    sx[f]=sg[f*Nsgeo+NXID];sy[f]=sg[f*Nsgeo+NYID];sz[f]=sg[f*Nsgeo+NZID];
    sw[f]=sg[f*Nsgeo+WSJID];
  }
  ogsGatherScatter(multiplicity.data(),ogsDfloat,ogsAdd,faceOgs);
  ogsGatherScatter(sx.data(),ogsDfloat,ogsAdd,faceOgs);
  ogsGatherScatter(sy.data(),ogsDfloat,ogsAdd,faceOgs);
  ogsGatherScatter(sz.data(),ogsDfloat,ogsAdd,faceOgs);
  ogsGatherScatter(sw.data(),ogsDfloat,ogsAdd,faceOgs);
  std::vector<dlong> map(6*N,-1);
  bool invalid=false;
  const dfloat tol=std::max(dfloat(1e-10),1000*std::numeric_limits<dfloat>::epsilon());
  for(dlong f=0;f<nf;++f) {
    const int b=m->EToB[f/m->Nfp];
    invalid|=b>3 || multiplicity[f]!=(b>0?1.0:2.0);
    if(b<=0) {
      invalid|=std::abs(sx[f])+std::abs(sy[f])+std::abs(sz[f])>tol;
      invalid|=std::abs(sw[f]-2*sg[f*Nsgeo+WSJID])>tol*sg[f*Nsgeo+WSJID];
    }
    const dlong n=m->vmapM[f];const int localFace=(f/m->Nfp)%6;
    if(n<0 || n>=N) invalid=true;
    else map[6*n+localFace]=f;
  }
  nekrsCheck(invalid,comm,EXIT_FAILURE,"%s",
      "DG face pairing/geometry check failed; use a conforming mesh with IDs 1=inlet,2=outlet,3=walls.\n");
  incidence.resize(6*N);incidence.copyFrom(map);
  solution.copyFrom(nrs->scalar->o_solution("alpha"),N);
  initialVolume=volume(solution);
  project();
  nrs->scalar->compute[nrs->scalar->nameToIndex.at("alpha")]=0;
  if(platform->comm.mpiRank()==0)
    printf("alphaDG: face-pair communication validated; LD OFF; SSPRK3, CFL=%g, limiter=%d.\n"
           "Native alpha solve OFF. Native alpha scalar stores only the conservative CG projection.\n",
        cfl,limiter);
}
inline dfloat inletValue(double time)
{
  if(p.startupMode==0) return p.alphaInlet;
  const dfloat s=std::max(dfloat(0),std::min(dfloat(1),
      dfloat((time-p.inletRampStartTime)/p.alphaInletRampDuration)));
  return p.alphaInlet*s*s*s*(10+s*(-15+6*s));
}
inline dfloat evaluate(double time)
{
  auto m=nrs->meshV;const dlong N=m->Nlocal,O=nrs->fieldOffset;
  volumeKernel(N,O,solution,o_ug,flux);
  // Native weak divergence returns D^T M flux WITHOUT global assembly.
  opSEM::divergence(m,O,flux,rhs);
  traceKernel(nf,O,m->o_vmapM,m->o_sgeo,solution,o_ug,trace);
  traceSum.copyFrom(trace,2*nf);
  ogsGatherScatterMany(traceSum,2,nf,ogsDfloat,ogsAdd,faceOgs);
  surfaceKernel(nf,inletValue(time),backflow,boundaryIds,m->o_sgeo,
      trace,traceSum,surface,boundary);
  liftKernel(N,incidence,m->o_LMM,surface,rhs);
  return platform->linAlg->sum(nf,boundary,platform->comm.mpiComm());
}
inline void limit()
{
  auto m=nrs->meshV;const auto comm=platform->comm.mpiComm();
  limitKernel(m->Nelements,limiter,m->o_LMM,solution,badMean,limited);
  const dfloat bad=platform->linAlg->max(m->Nelements,badMean,comm);
  nekrsCheck(bad>0,comm,EXIT_FAILURE,"%s",
      "DG produced a nonfinite value or inadmissible element mean. Reduce alphaDGCfl or fluid dt; no mean clipping is performed.\n");
  lastLimited+=platform->linAlg->sum(m->Nelements,limited,comm);
}
inline void advance(double endTime)
{
  auto m=nrs->meshV;const auto comm=platform->comm.mpiComm();
  const dlong N=m->Nlocal,O=nrs->fieldOffset;
  const dfloat dt=nrs->dt[0];
  reconstructGasVelocity(); // newly solved CG gas velocity, frozen in this step
  const dfloat minWeight=*std::min_element(m->gllw,m->gllw+m->Nq);
  rateKernel(N,O,1.0/minWeight,m->o_vgeo,o_ug,rate);
  const dfloat maxRate=platform->linAlg->max(N,rate,comm);
  const double requested=std::ceil(dt*maxRate/cfl);
  nekrsCheck(!std::isfinite(requested) || requested>maxSubsteps,comm,EXIT_FAILURE,
      "DG requires %.0f substeps (limit %d); reduce fluid dt.\n",requested,maxSubsteps);
  lastSubsteps=std::max(1,static_cast<int>(requested));
  const dfloat h=dt/lastSubsteps;
  lastCfl=h*maxRate;lastLimited=0;
  const dfloat before=volume(solution);
  dfloat integrated=0;
  for(int s=0;s<lastSubsteps;++s) {
    const double t=endTime-dt+s*h;
    old.copyFrom(solution,N);
    const dfloat b0=evaluate(t);
    rkKernel(N,0.0,h,old,rhs,solution);limit();
    const dfloat b1=evaluate(t+h);
    rkKernel(N,0.75,h,old,rhs,solution);limit();
    const dfloat b2=evaluate(t+0.5*h);
    rkKernel(N,1.0/3.0,h,old,rhs,solution);limit();
    integrated+=h*(b0/6.0+b1/6.0+2.0*b2/3.0);
  }
  boundaryIntegral+=integrated;
  lastBalance=volume(solution)-before+integrated;
  project();
}
inline void diagnostics(double time)
{
  auto m=nrs->meshV;const auto comm=platform->comm.mpiComm();
  const dfloat lo=platform->linAlg->min(m->Nlocal,solution,comm);
  const dfloat hi=platform->linAlg->max(m->Nlocal,solution,comm);
  const dfloat v=volume(solution);
  if(platform->comm.mpiRank()==0)
    printf("alphaDG time=%.9e substeps=%d CFL=%.6g min=%.9e max=%.9e V=%.12e "
           "stepBalance=%.3e cumulativeBalance=%.3e projectionDeltaV=%.3e limitedStages=%.0f\n",
        time,lastSubsteps,lastCfl,lo,hi,v,lastBalance,v-initialVolume+boundaryIntegral,
        lastProjectionError,lastLimited);
}
inline void write(double time)
{
  // Nek field output stores element-local values; do not gather-scatter DG.
  nrs->writeToFile("alphaDG",time,nrs->meshV,
      {{"scalar00",{solution.slice(0,nrs->meshV->Nlocal)}}},true,true);
}
} }
