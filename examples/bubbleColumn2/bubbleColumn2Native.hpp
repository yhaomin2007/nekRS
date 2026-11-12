#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "opSEM.hpp"
#include "avm.hpp"

namespace bubbleColumn2 {
struct Parameters {
  dfloat rhoLiquid=998.2,rhoGas=1.2,muLiquid=1e-3,muGas=1.8e-5;
  dfloat alphaInlet=.016667,ugInlet=.3,ugInitial=.3;
  dfloat initialPlumeHeight=.05,initialPlumeThickness=.01,alphaFloor=1e-6;
  dfloat gravity[3]={0,0,-9.81},bubbleDiameter=.003;
  dfloat virtualMassEnabled=1,virtualMassCoefficient=.5;
  int virtualMassStartStep=100,virtualMassRampStep=10000,virtualMassTimeDerivativeOrder=1;
  int alphaClippingEnabled=1,stabilityMonitorInterval=10;
};
static Parameters p;
static deviceMemory<dfloat> o_ug,o_ul,o_qg,o_gradUg,o_gradUl,o_gradUv,o_gradP,o_pressureSample;
static deviceMemory<dfloat> o_rhoM,o_rhoPressure,o_muM,o_alphaSource,o_ugSource;
static deviceMemory<dfloat> o_dragLambda,o_implicitDrag,o_dragLoad,o_previousDragLoad;
static deviceMemory<dfloat> o_mixtureDragSource,o_mixtureInterphaseAcceleration,o_mixtureForce;
static deviceMemory<dfloat> o_driftStress,o_divDriftStress,o_gasStress,o_divGasStress;
static deviceMemory<dfloat> o_exactMixtureStress,o_baseNativeStress,o_divExactMixtureStress,o_divBaseNativeStress;
static deviceMemory<dfloat> o_virtualMassRelativeAcceleration,o_liquidAcceleration;
static deviceMemory<dfloat> o_ulPrevious,o_ugPrevious,o_ulPrevious2,o_ugPrevious2;
static deviceMemory<dfloat> o_ulHistory,o_ulEXT,o_gasPredicted,o_gasUrst,o_diffBase,o_inletWeight;
static deviceMemory<dfloat> o_inverseGllSpacing,o_gasCfl;
static int liquidHistoryStep=-1,virtualMassHistorySamples=0;
static dfloat virtualMassPreviousDt=0;
static occa::kernel reconstructGasVelocityKernel,buildGasFluxKernel,buildLiquidVelocityKernel;
static occa::kernel buildEquationTermsKernel,addGasStressKernel,buildMixtureForceKernel;
static occa::kernel buildMixtureStressCorrectionKernel,addMixtureStressCorrectionKernel;
static occa::kernel updateVirtualMassHistoryKernel,initializePlumeKernel,clipAlphaKernel;
static occa::kernel compressionKernel,liquidEXTKernel,replaceDragKernel;
static const char *names[4]={"alpha","ugx","ugy","ugz"};

inline void registerKernels(deviceKernelProperties &info)
{
  if(platform->options.compareArgs("REGISTER ONLY","TRUE")) {
    platform->kernelRequests.add("method2Physics","bubbleColumn2Equations.okl",info);
    platform->kernelRequests.add("method2Native","bubbleColumn2Native.okl",info);
  } else {
#define LOAD(K,F) K=platform->kernelRequests.load("method2Physics",F)
    LOAD(reconstructGasVelocityKernel,"reconstructGasVelocity");
    LOAD(buildGasFluxKernel,"buildGasFlux");LOAD(buildLiquidVelocityKernel,"buildLiquidVelocity");
    LOAD(buildEquationTermsKernel,"buildEquationTerms");LOAD(addGasStressKernel,"addGasStress");
    LOAD(buildMixtureForceKernel,"buildMixtureForce");
    LOAD(buildMixtureStressCorrectionKernel,"buildMixtureStressCorrection");
    LOAD(addMixtureStressCorrectionKernel,"addMixtureStressCorrection");
    LOAD(updateVirtualMassHistoryKernel,"updateVirtualMassHistory");
    LOAD(initializePlumeKernel,"initializePlume");LOAD(clipAlphaKernel,"clipAlpha");
#undef LOAD
    compressionKernel=platform->kernelRequests.load("method2Native","alphaCompression");
    liquidEXTKernel=platform->kernelRequests.load("method2Native","liquidEXT");
    replaceDragKernel=platform->kernelRequests.load("method2Native","replaceGasDragLoad");
  }
}
inline dfloat virtualMassRampFactor(int step)
{
  if(step<=p.virtualMassStartStep)return 0;
  return p.virtualMassRampStep==0?1:std::min<dfloat>(1,dfloat(step-p.virtualMassStartStep)/p.virtualMassRampStep);
}
inline void reconstructLiquidVelocity()
{
  auto m=nrs->meshV;const dlong O=nrs->fieldOffset;
  reconstructGasVelocityKernel(m->Nlocal,O,0,p.alphaFloor,nrs->scalar->o_solution("alpha"),
      nrs->scalar->o_solution("ugx"),nrs->scalar->o_solution("ugy"),nrs->scalar->o_solution("ugz"),o_ug);
  buildGasFluxKernel(m->Nlocal,O,nrs->scalar->o_solution("alpha"),o_ug,o_qg);
  buildLiquidVelocityKernel(m->Nlocal,O,p.alphaFloor,nrs->scalar->o_solution("alpha"),nrs->fluid->o_U,o_qg,o_ul);
}
inline void evaluatePhysics()
{
  auto m=nrs->meshV;const dlong O=nrs->fieldOffset;auto alpha=nrs->scalar->o_solution("alpha");
  reconstructLiquidVelocity();
  opSEM::strongGradVec(m,O,o_ug,o_gradUg);opSEM::strongGradVec(m,O,o_ul,o_gradUl);
  opSEM::strongGradVec(m,O,nrs->fluid->o_U,o_gradUv);opSEM::strongGrad(m,O,nrs->fluid->o_P,o_gradP);
  buildEquationTermsKernel(m->Nlocal,O,p.rhoLiquid,p.rhoGas,p.muLiquid,p.muGas,p.alphaFloor,
      1.0,1.0,1.0,0.0,0.0,1.0,p.bubbleDiameter,p.virtualMassEnabled*virtualMassRampFactor(nrs->tstep),
      p.virtualMassCoefficient,1,0,p.gravity[0],p.gravity[1],p.gravity[2],alpha,nrs->fluid->o_U,
      o_ug,o_ul,o_gradUg,o_virtualMassRelativeAcceleration,o_liquidAcceleration,o_gradP,
      o_rhoM,o_rhoPressure,o_muM,o_driftStress,o_gasStress,o_alphaSource,o_ugSource,o_dragLambda,
      o_mixtureInterphaseAcceleration,o_mixtureDragSource);
  compressionKernel(m->Nlocal,O,alpha,o_gradUg,o_alphaSource);
  const dfloat rhoEff=p.rhoGas+p.virtualMassEnabled*virtualMassRampFactor(nrs->tstep)*p.virtualMassCoefficient*p.rhoLiquid;
  for(int c=0;c<3;++c){
    auto row=o_gasStress.slice(3*c*O,3*O);auto div=o_divGasStress.slice(c*O,O);
    opSEM::strongDivergence(m,O,row,div);
    addGasStressKernel(m->Nlocal,p.alphaFloor,rhoEff,alpha,div,o_ugSource.slice(c*O,O));
    row=o_driftStress.slice(3*c*O,3*O);div=o_divDriftStress.slice(c*O,O);
    opSEM::strongDivergence(m,O,row,div);
  }
  buildMixtureForceKernel(m->Nlocal,O,p.alphaFloor,1.0,p.gravity[0],p.gravity[1],p.gravity[2],
      alpha,o_divDriftStress,o_mixtureInterphaseAcceleration,o_mixtureForce);
  buildMixtureStressCorrectionKernel(m->Nlocal,O,p.rhoLiquid,p.rhoGas,p.muLiquid,p.muGas,
      alpha,o_rhoPressure,o_gradUl,o_gradUg,o_gradUv,o_exactMixtureStress,o_baseNativeStress);
  for(int c=0;c<3;++c){
    auto row=o_exactMixtureStress.slice(3*c*O,3*O);auto div=o_divExactMixtureStress.slice(c*O,O);
    opSEM::strongDivergence(m,O,row,div);
    row=o_baseNativeStress.slice(3*c*O,3*O);div=o_divBaseNativeStress.slice(c*O,O);
    opSEM::strongDivergence(m,O,row,div);
  }
  addMixtureStressCorrectionKernel(m->Nlocal,O,1.0,o_rhoPressure,o_divExactMixtureStress,o_divBaseNativeStress,o_mixtureForce);
}
inline void applyGasVelocityAVM()
{
  for(const auto name:names){
    const auto prefix="SCALAR"+scalarDigitStr(nrs->scalar->nameToIndex.at(name))+" ";
    if(!platform->options.compareArgs(prefix+"REGULARIZATION METHOD","AVM_AVERAGED_MODAL_DECAY"))continue;
    nekrsCheck(nrs->meshV->N<5,platform->comm.mpiComm(),EXIT_FAILURE,"%s","AVM requires polynomialOrder >= 5.\n");
    dfloat width=1,decay=2,scaling=1,noise=0;
    platform->options.getArgs(prefix+"REGULARIZATION AVM ACTIVATION WIDTH",width);
    platform->options.getArgs(prefix+"REGULARIZATION AVM DECAY THRESHOLD",decay);
    platform->options.getArgs(prefix+"REGULARIZATION AVM SCALING COEFF",scaling);
    platform->options.getArgs(prefix+"REGULARIZATION AVM ABSOLUTE TOL",noise);
    const bool c0=platform->options.compareArgs(prefix+"REGULARIZATION AVM C0","TRUE");
    auto eps=avm::viscosity(nrs->fieldOffset,o_ug,nrs->scalar->o_solution(name),noise,scaling,decay,width,c0);
    auto diff=nrs->scalar->o_diffusionCoeff(name);
    platform->linAlg->axpby(nrs->meshV->Nlocal,1.0,eps,1.0,diff);
  }
}
inline void updateProperties(double)
{
  evaluatePhysics();
  nrs->fluid->o_prop.slice(0,nrs->fieldOffset).copyFrom(o_muM);
  nrs->fluid->o_prop.slice(nrs->fieldOffset,nrs->fieldOffset).copyFrom(o_rhoPressure);
  for(int i=0;i<4;++i)nrs->scalar->o_diffusionCoeff(names[i]).copyFrom(o_diffBase,nrs->meshV->Nlocal,0,i*nrs->fieldOffset);
  applyGasVelocityAVM();
}
inline void pinGasDragHistory()
{
  auto &s=nrs->scalar;const dlong N=nrs->meshV->Nlocal,O=nrs->fieldOffset;
  // Pin ONLY the already-extrapolated gas drag load across source slots so
  // native EXT does not extrapolate UL twice. All other sources use native EXT.
  if(nrs->tstep>0){
    for(int j=1;j<s->o_coeffEXT.size();++j)for(int c=0;c<3;++c){
      auto old=s->o_EXT.slice(j*s->fieldOffsetSum+s->fieldOffsetScan[c+1],N);
      platform->linAlg->axpby(N,1.0,o_dragLoad.slice(c*O,O),1.0,old);
      platform->linAlg->axpby(N,-1.0,o_previousDragLoad.slice(c*O,O),1.0,old);
    }
    o_previousDragLoad.copyFrom(o_dragLoad,3*O);
  }
}
inline void addExplicitSources(double)
{
  auto &s=nrs->scalar;const dlong N=nrs->meshV->Nlocal,O=nrs->fieldOffset;
  evaluatePhysics();
  // Native operator uses current UG, then native makeForcing extrapolates the
  // complete operator. Only BC prediction uses native extrapolated scalars.
  for(int c=0;c<3;++c)o_gasPredicted.copyFrom(s->o_Se,N,c*O,s->fieldOffsetScan[c+1]);
  s->computeUrst(); // separate scalar Urst; never overwrite fluid UM Urst
  liquidEXTKernel(N,O,static_cast<int>(s->o_coeffEXT.size()),static_cast<int>(liquidHistoryStep<0),
      static_cast<int>(nrs->tstep>0 && liquidHistoryStep!=nrs->tstep),s->o_coeffEXT,o_ul,o_ulHistory,o_ulEXT);
  if(nrs->tstep>0)liquidHistoryStep=nrs->tstep;
  replaceDragKernel(N,O,o_dragLambda,o_ul,o_ulEXT,o_ugSource,o_dragLoad);
  o_implicitDrag.copyFrom(o_dragLambda,N);o_pressureSample.copyFrom(o_gradP,3*O);
  s->o_explicitTerms("alpha").copyFrom(o_alphaSource,N);
  for(int c=0;c<3;++c)s->o_explicitTerms(names[c+1]).copyFrom(o_ugSource,N,0,c*O);
  pinGasDragHistory();
  // Full UM drag is left in normal source history and extrapolated natively.
  nrs->fluid->o_explicitTerms().copyFrom(o_mixtureForce,3*O);
}
inline occa::memory implicitGasDrag(double,int is)
{
  return is>=1 && is<=3?static_cast<occa::memory>(o_implicitDrag):o_NULL;
}
inline void postProcessAlpha(double,int)
{
  if(p.alphaClippingEnabled)clipAlphaKernel(nrs->meshV->Nlocal,0.0,1.0,nrs->scalar->o_solution("alpha"));
}
inline void enforceZeroDivergence(double) {}

inline void virtualMassTimeCoefficients(dfloat dt, dfloat previousDt,
                                       bool secondOrder, dfloat &c0, dfloat &c1, dfloat &c2)
{
  c0 = 1.0 / dt;
  c1 = -c0;
  c2 = 0.0;
  if (secondOrder) {
    c0 = (2.0 * dt + previousDt) / (dt * (dt + previousDt));
    c1 = -(dt + previousDt) / (dt * previousDt);
    c2 = dt / (previousDt * (dt + previousDt));
  }
}
inline void reconstructGasVelocity() { reconstructLiquidVelocity(); }

inline void updateVirtualMassHistory()
{
  auto mesh = nrs->meshV;
  const dlong offset = nrs->fieldOffset;
  auto alpha = nrs->scalar->o_solution("alpha");

  reconstructGasVelocity();
  buildLiquidVelocityKernel(mesh->Nlocal,
                            offset,
                            p.alphaFloor,
                            alpha,
                            nrs->fluid->o_U,
                            o_qg,
                            o_ul);
  opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg);
  opSEM::strongGradVec(mesh, offset, o_ul, o_gradUl);
  dfloat c0, c1, c2;
  virtualMassTimeCoefficients(nrs->dt[0], virtualMassPreviousDt,
      p.virtualMassTimeDerivativeOrder == 2 && virtualMassHistorySamples >= 1,
      c0, c1, c2);
  updateVirtualMassHistoryKernel(mesh->Nlocal,
                                 offset,
                                 c0, c1, c2,
                                 o_ug,
                                 o_ul,
                                 o_ulPrevious,
                                 o_ugPrevious,
                                 o_ulPrevious2,
                                 o_ugPrevious2,
                                 o_gradUl,
                                 o_gradUg,
                                 o_virtualMassRelativeAcceleration,
                                 o_liquidAcceleration);
  virtualMassPreviousDt = nrs->dt[0];
  ++virtualMassHistorySamples;
}
inline dfloat computeGasCfl()
{
  auto mesh = nrs->meshV;
  launchKernel("nrs-cflHex3D",
               mesh->Nelements,
               nrs->dt[0],
               mesh->o_vgeo,
               o_inverseGllSpacing,
               nrs->fieldOffset,
               o_ug,
               nrs->geom ? nrs->geom->o_U : o_NULL,
               o_gasCfl);
  return platform->linAlg->max(
      mesh->Nelements, o_gasCfl, platform->comm.mpiComm());
}

inline void allocate()
{
  auto m=nrs->meshV;const dlong O=nrs->fieldOffset;
  for(auto mem:{&o_ug,&o_ul,&o_qg,&o_gradP,&o_pressureSample,&o_ugSource,&o_dragLoad,&o_previousDragLoad,
      &o_mixtureDragSource,&o_mixtureInterphaseAcceleration,&o_mixtureForce,&o_divDriftStress,&o_divGasStress,
      &o_divExactMixtureStress,&o_divBaseNativeStress,&o_virtualMassRelativeAcceleration,&o_liquidAcceleration,
      &o_ulPrevious,&o_ugPrevious,&o_ulPrevious2,&o_ugPrevious2,&o_ulEXT,&o_gasPredicted}) {
    mem->resize(3*O);platform->linAlg->fill(3*O,0.0,*mem);
  }
  for(auto mem:{&o_gradUg,&o_gradUl,&o_gradUv,&o_driftStress,&o_gasStress,&o_exactMixtureStress,&o_baseNativeStress})mem->resize(9*O);
  for(auto mem:{&o_rhoM,&o_rhoPressure,&o_muM,&o_alphaSource,&o_dragLambda,&o_implicitDrag,&o_inletWeight})mem->resize(O);
  o_diffBase.resize(4*O);
  for(int i=0;i<4;++i)o_diffBase.copyFrom(nrs->scalar->o_diffusionCoeff(names[i]),m->Nlocal,i*O,0);
  o_ulHistory.resize(nrs->scalar->o_coeffEXT.size()*3*O);
  o_gasUrst.resize(3*nrs->scalar->vCubatureOffset);
  std::vector<dfloat> weight(O,1.0);
  for(dlong e=0;e<m->Nelements;++e)for(int f=0;f<m->Nfaces;++f)if(m->EToB[e*m->Nfaces+f]==3)
    for(int j=0;j<m->Nfp;++j)weight[m->vmapM[(e*m->Nfaces+f)*m->Nfp+j]]=0;
  o_inletWeight.copyFrom(weight);
  oogs::startFinish(o_inletWeight,1,0,ogsDfloat,ogsMin,m->oogs);
  platform->app->bc->o_usrwrk=o_inletWeight;
  std::vector<dfloat> inverse(m->N+1);
  for(int j=0;j<=m->N;++j){
    dfloat h=j==0?m->gllz[1]-m->gllz[0]:j==m->N?m->gllz[j]-m->gllz[j-1]:.5*(m->gllz[j+1]-m->gllz[j-1]);
    inverse[j]=1/h;
  }
  o_inverseGllSpacing.resize(m->N+1);o_inverseGllSpacing.copyFrom(inverse);
  o_gasCfl.resize(m->Nelements);
}
inline void initializeHistory()
{
  // Distinct scalar velocity/Urst buffers: fluid UM advection remains native.
  nrs->scalar->o_U=o_ug;nrs->scalar->o_Ue=o_gasPredicted;nrs->scalar->o_relUrst=o_gasUrst;
  evaluatePhysics();
  o_gasPredicted.copyFrom(o_ug);
  o_ulPrevious.copyFrom(o_ul);o_ulPrevious2.copyFrom(o_ul);
  o_ugPrevious.copyFrom(o_ug);o_ugPrevious2.copyFrom(o_ug);
  liquidHistoryStep=-1;virtualMassHistorySamples=0;virtualMassPreviousDt=0;
}
inline void writeVector(const std::string &prefix,double time,deviceMemory<dfloat> field)
{
  std::vector<deviceMemory<dfloat>> components;
  for(int c=0;c<3;++c)components.push_back(field.slice(c*nrs->fieldOffset,nrs->meshV->Nlocal));
  nrs->writeToFile(prefix,time,nrs->meshV,{{"velocity",components}});
}
} // namespace bubbleColumn2
