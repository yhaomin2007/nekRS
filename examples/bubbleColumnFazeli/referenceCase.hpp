#pragma once

// Kept separate from the Method1 snapshot to expose all benchmark adaptations.
namespace bubbleColumn { namespace referenceCase {
static dfloat gasFlowLph=48.0, sigma=0.072, injectionVelocity=0.0;
static dfloat xMin=0.08,xMax=0.12,yMin=0.01,yMax=0.04,zMin=0.01,zMax=0.04;
static deviceMemory<dfloat> source, injectionRate, stress, stressDivergence;
static occa::kernel initializePressureKernel,injectionMomentumKernel;
static occa::kernel addInjectionDivergenceKernel,liquidStressKernel,addLiquidStressForceKernel;

inline dfloat eotvos()
{ return std::abs(p.gravity[2])*(p.rhoLiquid-p.rhoGas)*p.bubbleDiameter*p.bubbleDiameter/sigma; }
inline dfloat modifiedEotvos()
{
  const dfloat factor=1.0+0.163*std::pow(eotvos(),0.757);
  return eotvos()*factor*factor;
}
inline dfloat cdRe(dfloat re)
{
  return std::max(std::min(16.0*(1.0+0.15*std::pow(re,0.687)),48.0),
      (8.0/3.0)*eotvos()/(eotvos()+4.0)*re);
}
inline void configure()
{
  auto read=[&](const char *key,auto &value) { platform->par->extract("casedata",key,value); };
  read("gasflowlph",gasFlowLph); read("surfacetension",sigma);
  read("sourcexmin",xMin); read("sourcexmax",xMax);
  read("sourceymin",yMin); read("sourceymax",yMax);
  read("sourcezmin",zMin); read("sourcezmax",zMax);
  nekrsCheck(!std::isfinite(gasFlowLph) || gasFlowLph<=0 ||
      !std::isfinite(sigma) || sigma<=0 || p.rhoLiquid<=p.rhoGas ||
      !std::isfinite(xMin) || !std::isfinite(xMax) || xMin<0 || xMax>0.20 || xMin>=xMax ||
      !std::isfinite(yMin) || !std::isfinite(yMax) || yMin<0 || yMax>0.05 || yMin>=yMax ||
      !std::isfinite(zMin) || !std::isfinite(zMax) || zMin<=0 || zMax>=0.45 || zMin>=zMax,
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","Invalid Fazeli source/physical parameters.\n");
  // Solve Eq. (16) with the SAME Cd as the momentum equation; bisection
  // avoids fixed-point oscillation between Tomiyama drag branches.
  dfloat lo=0.0,hi=10.0;
  const dfloat rhs=(p.rhoLiquid-p.rhoGas)*std::abs(p.gravity[2]);
  for (int i=0;i<100;++i) {
    const dfloat u=0.5*(lo+hi),re=p.rhoLiquid*p.bubbleDiameter*u/p.muLiquid;
    const dfloat drag=0.75*cdRe(re)*p.muLiquid*u/(p.bubbleDiameter*p.bubbleDiameter);
    if (drag>rhs) hi=u; else lo=u;
  }
  injectionVelocity=0.5*(lo+hi);
  p.ugInitial=p.ugInlet=injectionVelocity;
}
inline void registerKernels(deviceKernelProperties &info)
{
  info.define("p_REFERENCE_EO")=eotvos();
  info.define("p_REFERENCE_EOD")=modifiedEotvos();
  const std::string req="bubbleColumnFazeli-reference";
  if (platform->options.compareArgs("REGISTER ONLY","TRUE"))
    platform->kernelRequests.add(req,"referenceCase.okl",info);
  else {
    initializePressureKernel=platform->kernelRequests.load(req,"initializeReferencePressure");
    injectionMomentumKernel=platform->kernelRequests.load(req,"injectionMomentum");
    addInjectionDivergenceKernel=platform->kernelRequests.load(req,"addInjectionDivergence");
    liquidStressKernel=platform->kernelRequests.load(req,"liquidStress");
    addLiquidStressForceKernel=platform->kernelRequests.load(req,"addLiquidStressForce");
  }
}
inline void allocate()
{
  auto mesh=nrs->meshV;
  const auto comm=platform->comm.mpiComm();
  const dlong N=mesh->Nlocal,offset=nrs->fieldOffset;
  source.resize(offset); injectionRate.resize(offset);
  stress.resize(9*offset); stressDivergence.resize(3*offset);
  std::vector<dfloat> mask(offset,0.0);
  // Element-centre box indicator; source box edges must coincide with mesh
  // faces. Its mass projection makes a unique CG source at shared nodes.
  const dfloat tol=1e-10;
  const auto [x,y,z]=mesh->xyzHost();
  bool misaligned=false;
  for (dlong e=0;e<mesh->Nelements;++e) {
    const dlong first=e*mesh->Np,last=first+mesh->Np-1;
    const dfloat xc=0.5*(x[first]+x[last]);
    const dfloat yc=0.5*(y[first]+y[last]);
    const dfloat zc=0.5*(z[first]+z[last]);
    if (xc>xMin && xc<xMax && yc>yMin && yc<yMax && zc>zMin && zc<zMax) {
      misaligned=misaligned || x[first]<xMin-tol || x[last]>xMax+tol ||
          y[first]<yMin-tol || y[last]>yMax+tol ||
          z[first]<zMin-tol || z[last]>zMax+tol;
      std::fill(mask.begin()+first,mask.begin()+last+1,1.0);
    }
  }
  nekrsCheck(misaligned,comm,EXIT_FAILURE,"%s",
      "Source box must align with structured mesh faces.\n");
  source.copyFrom(mask);
  assembleDivergence(source);
  const dfloat volume=platform->linAlg->innerProd(N,mesh->o_LMM,source,comm);
  nekrsCheck(volume<=0,comm,EXIT_FAILURE,"%s","Injection box contains no elements.\n");
  const dfloat specifiedVolume=(xMax-xMin)*(yMax-yMin)*(zMax-zMin);
  nekrsCheck(std::abs(volume-specifiedVolume)>1e-8*specifiedVolume,comm,EXIT_FAILURE,
      "%s","Source box is not mesh-aligned or mesh has the wrong dimensions.\n");
  const dfloat flow=gasFlowLph*1e-3/3600.0;
  platform->linAlg->scale(N,flow/volume,source);
  const dfloat integratedFlow=platform->linAlg->innerProd(N,mesh->o_LMM,source,comm);
  nekrsCheck(std::abs(integratedFlow-flow)>1e-10*flow,comm,EXIT_FAILURE,
      "%s","Discrete gas source does not integrate to the requested flow.\n");
  platform->linAlg->fill(N,0.0,injectionRate);
  // Wall UM is alpha*rhoG*Uinj/rhoM, hence reconstructed liquid is at rest.
  platform->app->bc->o_usrwrk=nrs->scalar->o_solution("alpha");
  nrs->scalar->userImplicitLinearTerm=[](double,int is)->occa::memory {
    return is==nrs->scalar->nameToIndex.at("alpha") ? occa::memory() : injectionRate;
  };
  if (platform->comm.mpiRank()==0) {
    printf("Fazeli reference: Q=%.12e m3/s (%.8g L/h), discrete injection volume=%.12e m3, S=%.12e /s\n",
        flow,gasFlowLph,volume,flow/volume);
    printf("Tomiyama: Eo=%.10g Eod=%.10g; Eq16 Uinj=%.10g m/s; MP-LD Cdis=%.10g\n",
        eotvos(),modifiedEotvos(),injectionVelocity,p.laminarDispersionCoefficient);
  }
}
inline void addSources()
{
  const dlong N=nrs->meshV->Nlocal,offset=nrs->fieldOffset;
  injectionMomentumKernel(N,offset,p.rhoLiquid,p.rhoGas,
      p.virtualMassEnabled*p.virtualMassCoefficient,p.alphaFloor,injectionVelocity,
      nrs->scalar->o_solution("alpha"),source,nrs->fluid->o_U,injectionRate,o_ugSource,o_mixtureForce);
  // Brennen's total mixture stress is tauL, NOT beta*tauL+alpha*tauG.
  // -D^T M tauL is a weak volume load. Zero natural top traction is used;
  // no boundary stress flux is added. Dirichlet wall rows are imposed by core.
  liquidStressKernel(N,offset,p.muLiquid,o_gradUl,stress);
  for (int i=0;i<3;++i) {
    auto div=stressDivergence.slice(i*offset,offset);
    opSEM::divergence(nrs->meshV,offset,stress.slice(3*i*offset,3*offset),div);
    oogs::startFinish(div,1,0,ogsDfloat,ogsAdd,nrs->meshV->oogs);
    platform->linAlg->axmy(N,1.0,nrs->meshV->o_invLMM,div);
    addLiquidStressForceKernel(N,o_rhoM,div,o_mixtureForce.slice(i*offset,offset));
    nrs->fluid->o_explicitTerms().copyFrom(o_mixtureForce,N,i*offset,i*offset);
    const char *names[3]={"ugx","ugy","ugz"};
    nrs->scalar->o_explicitTerms(names[i]).copyFrom(o_ugSource,N,0,i*offset);
  }
}
inline void diagnostics()
{
  const dlong N=nrs->meshV->Nlocal;
  const auto comm=platform->comm.mpiComm();
  auto alpha=nrs->scalar->o_solution("alpha");
  const dfloat lo=platform->linAlg->min(N,alpha,comm),hi=platform->linAlg->max(N,alpha,comm);
  const dfloat volume=platform->linAlg->innerProd(N,nrs->meshV->o_LMM,alpha,comm);
  const dfloat outflow=nrs->meshV->surfaceAreaNormalMultiplyVectorIntegrate(
      nrs->fieldOffset,o_outletBoundaryID,o_qg);
  if (platform->comm.mpiRank()==0)
    printf("fazeliMonitor step=%d alphaMin=%.12e alphaMax=%.12e gasVolume=%.12e m3 Q=%.12e gasOutflow=%.12e m3/s\n",
        nrs->tstep,lo,hi,volume,gasFlowLph*1e-3/3600.0,outflow);
}
} }
