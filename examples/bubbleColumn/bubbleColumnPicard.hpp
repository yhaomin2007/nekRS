#pragma once

// User-side block Picard. Native fluid ADV/EXT and BDF history are never
// rebuilt or shifted here. UM convection stays on the native EXT path.
namespace bubbleColumn { namespace picard {
static int mode = 0, maxPasses = 6, acceptAtCap = 0;
static dfloat rtol = 1e-6, alphaAtol = 1e-8, ugAtol = 1e-7;
static dfloat umAtol = 1e-7, pressureAtol = 1e-3, divAtol = 1e-6, repeatTol = 1e-9;
static int step = -1, pass = 0, checkpointRequest = 0;
static bool corrector = false;

struct Saved {
  occa::memory live;
  deviceMemory<dfloat> data;
  void take(const occa::memory &x) { live=x; data.resize(x.size()); data.copyFrom(x,x.size()); }
  void restore() { live.copyFrom(data,data.size()); }
};
inline void copy(deviceMemory<dfloat> &a, const occa::memory &b, dlong count)
{ a.resize(count); a.copyFrom(b,count); }

static Saved uHistory, sHistory, pressureStart, uPredictor, sPredictor, nativeUrst;
static Saved rhsU, rhsS, extU, extS, propertiesU, propertiesS, dragStart;
static Saved sourceAlphaHistory, gradStart, liquidStart, ulPreviousStart;
static Saved nativeBdf, fluidExt, scalarExt;
static bool gradInitializedStart, gradInitializedReplay;
static int gradStepStart, alphaHistoryStepStart;
static deviceMemory<dfloat> startU, startS, firstU, firstS, firstP, firstDiv;
static deviceMemory<dfloat> previousU, previousS, previousP, previousDiv;
static deviceMemory<dfloat> baseU, baseS, nativeUserU, nativeUserS;
static deviceMemory<dfloat> userHistoryU, userHistoryS, extUserU, extUserS;
static deviceMemory<dfloat> freshUserU, freshUserS, pressureAnchor, oldUl, iterateUrst;
static deviceMemory<dfloat> scratch, measuredDiv;

struct Counters {
  dfloat alphaFilter, alphaClip, q[3], magnitude;
  void take() {
    alphaFilter=cumulativeAlphaFilterDeltaVolume; alphaClip=cumulativeAlphaClipDeltaVolume;
    for(int c=0;c<3;++c) q[c]=cumulativeQgMaskDeltaIntegral[c];
    magnitude=cumulativeQgMaskDeltaMagnitudeIntegral;
  }
  void restore() const {
    cumulativeAlphaFilterDeltaVolume=alphaFilter; cumulativeAlphaClipDeltaVolume=alphaClip;
    for(int c=0;c<3;++c) cumulativeQgMaskDeltaIntegral[c]=q[c];
    cumulativeQgMaskDeltaMagnitudeIntegral=magnitude;
  }
};
static Counters countersStart;

// Each history contains ONLY the original userSource sample, before native
// HPF is added. These match the native EXT source slots exactly, independently
// of the accepted Picard source. Native predictor-source history is preserved.
inline void pushUserSample(deviceMemory<dfloat> &history,
                           const occa::memory &sample, int states, dlong stride)
{
  if (!history.isInitialized()) {
    history.resize(states*stride);
    platform->linAlg->fill(history.size(),0.0,history);
  }
  for(int h=states-1;h>0;--h) history.copyFrom(history,stride,h*stride,(h-1)*stride);
  history.copyFrom(sample,stride);
}
inline void extrapolateUser(deviceMemory<dfloat> &out, const deviceMemory<dfloat> &history,
                            const occa::memory &coeff, dlong stride)
{
  out.resize(stride); platform->linAlg->fill(stride,0.0,out);
  std::vector<dfloat> weights(coeff.size()); coeff.copyTo(weights.data());
  for(size_t h=0;h<weights.size();++h) {
    auto input=history.slice(h*stride,stride);
    platform->linAlg->axpby(stride,weights[h],input,1.0,out);
  }
}

// Apply the exact native fixed-mesh mass weights. Fluid forcing is an
// acceleration load (native sumMakef does not multiply it by rho). Scalars
// require transportCoeff=1, as already enforced by this case.
inline void addLoad(const occa::memory &destination, const occa::memory &source,
                    bool fluid, dfloat factor)
{
  auto mesh=nrs->meshV;
  const int count=fluid?3:4;
  scratch.resize(mesh->Nlocal);
  for(int c=0;c<count;++c) {
    const dlong offset=fluid?c*nrs->fluid->fieldOffset:nrs->scalar->fieldOffsetScan[c];
    scratch.copyFrom(source,mesh->Nlocal,0,offset);
    const auto mass=fluid?mesh->o_Jw:mesh->o_LMM;
    platform->linAlg->axmy(mesh->Nlocal,1.0,mass,scratch);
    auto output=destination.slice(offset,mesh->Nlocal);
    platform->linAlg->axpby(mesh->Nlocal,factor,scratch,1.0,output);
  }
}

inline void pressureFilter()
{
  const dlong N=nrs->meshV->Nlocal, offset=nrs->fieldOffset;
  if(p.gasPressureGradientFilterEnabled==0.0 || !gradInitializedStart) {
    o_gradP.copyFrom(o_gradPSpatial,3*offset);
  } else {
    o_gradP.copyFrom(pressureAnchor,3*offset);
    for(int c=0;c<3;++c) {
      auto output=o_gradP.slice(c*offset,offset);
      auto input=o_gradPSpatial.slice(c*offset,offset);
      platform->linAlg->axpby(N,p.gasPressureGradientFilterWeight,input,
          1.0-p.gasPressureGradientFilterWeight,output);
    }
  }
  pressureGradientFilterInitialized=true;
  pressureGradientFilterStep=nrs->tstep;
  // The original evaluator applies the inlet mask AFTER this hook.
}
inline void buildIterateUrst()
{
  auto mesh=nrs->meshV;
  iterateUrst.resize(nrs->fluid->o_relUrst.size());
  if(platform->options.compareArgs("ADVECTION TYPE","CUBATURE")) {
    launchKernel("nrs-UrstCubatureHex3D",mesh->Nelements,0,mesh->o_cubvgeo,
        mesh->o_cubInterpT,nrs->fluid->fieldOffset,0,nrs->fluid->cubatureOffset,
        nrs->fluid->o_U,nrs->fluid->o_U,iterateUrst);
  } else {
    launchKernel("nrs-UrstHex3D",mesh->Nelements,0,mesh->o_vgeo,
        nrs->fluid->fieldOffset,0,nrs->fluid->o_U,nrs->fluid->o_U,iterateUrst);
  }
  // Only continuity's material-advection diagnostics use this override.
  // Native fluid convection continues to use the original o_relUrst.
  o_divergenceUrstOverride=iterateUrst;
}
inline void reducedAlphaRhs()
{
  const dlong N=nrs->meshV->Nlocal;
  if(p.mixtureDivergenceForm==2) {
    o_alphaMixtureRhs.copyFrom(o_alphaRelativeRhs,N);
    if(p.subtractAlphaDiffusion!=0.0) {
      platform->linAlg->axpby(N,1.0,o_alphaSource,1.0,o_alphaMixtureRhs);
      platform->linAlg->axpby(N,-1.0,o_alphaTransportOnly,1.0,o_alphaMixtureRhs);
    }
  } else {
    o_alphaMixtureRhs.copyFrom(o_alphaSource,N);
  }
  assembleDivergence(o_alphaMixtureRhs);
}
inline void afterScalarPhysics()
{
  postProcessGasFlux();
  if(corrector) prepareGasAdvection();
  evaluatePointwiseTerms();
  if(corrector) {
    // alpha/UG are newly solved, UM is the preceding complete iterate.
    o_mixtureAdvectorUsed.copyFrom(nrs->fluid->o_U,3*nrs->fieldOffset);
    prepareRelativeDivergenceRhs();
    subtractScalarDiffusion();
    buildIterateUrst();
  }
}
inline void postScalar(double time,int tstep)
{
  countersStart.restore();
  postProcessAlpha(time,tstep);
}
inline void preFluid(double,int)
{
  if(!corrector) return;
  evaluateMixtureForce();
  nrs->fluid->o_JwF.copyFrom(baseU,baseU.size());
  addLoad(nrs->fluid->o_JwF,o_mixtureForce,true,1.0);
}

inline void beginStep(double time)
{
  nekrsCheck(step==nrs->tstep,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","bubble Picard: repeated timestep initialization.\n");
  step=nrs->tstep; pass=0; corrector=false; checkpointRequest=nrs->checkpointStep;
  picardPressureGradientUpdate=nullptr; picardAlphaRhs=nullptr;
  o_divergenceUrstOverride=occa::memory();
  countersStart.take();
  copy(startU,nrs->fluid->o_U,nrs->fluid->o_U.size());
  copy(startS,nrs->scalar->o_S,nrs->scalar->o_S.size());
  pressureStart.take(nrs->fluid->o_P);
  uPredictor.take(nrs->fluid->o_Ue); sPredictor.take(nrs->scalar->o_Se);
  nativeUrst.take(nrs->fluid->o_relUrst);
  nativeBdf.take(nrs->o_coeffBDF); fluidExt.take(nrs->fluid->o_coeffEXT);
  scalarExt.take(nrs->scalar->o_coeffEXT);
  copy(pressureAnchor,o_gradP,o_gradP.size());
  copy(oldUl,o_ulPrevious,o_ulPrevious.size());
  gradInitializedStart=pressureGradientFilterInitialized;
  addExplicitSources(time); // native predictor remains exactly the old path
  copy(nativeUserU,nrs->fluid->o_explicitTerms(),nrs->fluid->fieldOffsetSum);
  copy(nativeUserS,nrs->scalar->o_EXT,nrs->scalar->fieldOffsetSum);
  pushUserSample(userHistoryU,nativeUserU,nrs->fluid->o_coeffEXT.size(),nrs->fluid->fieldOffsetSum);
  pushUserSample(userHistoryS,nativeUserS,nrs->scalar->o_coeffEXT.size(),nrs->scalar->fieldOffsetSum);
  extrapolateUser(extUserU,userHistoryU,nrs->fluid->o_coeffEXT,nrs->fluid->fieldOffsetSum);
  extrapolateUser(extUserS,userHistoryS,nrs->scalar->o_coeffEXT,nrs->scalar->fieldOffsetSum);
  propertiesU.take(nrs->fluid->o_prop); propertiesS.take(nrs->scalar->o_prop);
  dragStart.take(o_dragLambda); sourceAlphaHistory.take(o_alphaTransportRhsHistory);
  gradStart.take(o_gradP); liquidStart.take(o_liquidAcceleration); ulPreviousStart.take(o_ulPrevious);
  gradInitializedReplay=pressureGradientFilterInitialized;
  gradStepStart=pressureGradientFilterStep; alphaHistoryStepStart=alphaTransportHistoryStep;
}

struct Change { dfloat maximum=0,rms=0,scaled=0; };
inline Change difference(const occa::memory &a,const occa::memory &b,
                         dlong offset,int components,dfloat atol,bool scalar=false)
{
  const auto comm=platform->comm.mpiComm(); auto mesh=nrs->meshV;
  scratch.resize(mesh->Nlocal); Change result; dfloat norm=0,squares=0;
  for(int c=0;c<components;++c) {
    const dlong pos=scalar?nrs->scalar->fieldOffsetScan[c]:c*offset;
    auto x=a.slice(pos,mesh->Nlocal), y=b.slice(pos,mesh->Nlocal);
    scratch.copyFrom(x,mesh->Nlocal);
    platform->linAlg->axpby(mesh->Nlocal,-1.0,y,1.0,scratch);
    const dfloat lo=platform->linAlg->min(mesh->Nlocal,scratch,comm);
    const dfloat hi=platform->linAlg->max(mesh->Nlocal,scratch,comm);
    const dfloat yl=platform->linAlg->min(mesh->Nlocal,y,comm);
    const dfloat yh=platform->linAlg->max(mesh->Nlocal,y,comm);
    nekrsCheck(!std::isfinite(lo)||!std::isfinite(hi)||!std::isfinite(yl)||!std::isfinite(yh),
        comm,EXIT_FAILURE,"%s","Nonfinite bubble Picard iterate.\n");
    result.maximum=std::max(result.maximum,std::max(std::abs(lo),std::abs(hi)));
    norm=std::max(norm,std::max(std::abs(yl),std::abs(yh)));
    platform->linAlg->axmy(mesh->Nlocal,1.0,scratch,scratch);
    squares+=platform->linAlg->innerProd(mesh->Nlocal,mesh->o_LMM,scratch,comm);
  }
  nekrsCheck(!std::isfinite(squares),comm,EXIT_FAILURE,"%s","Nonfinite Picard norm.\n");
  result.rms=std::sqrt(std::max(dfloat(0),squares)/(components*mesh->volume));
  result.scaled=result.maximum/(atol+rtol*norm);
  return result;
}
inline dfloat unchanged(const Saved &x,dlong skip=0)
{
  const dlong count=x.data.size()-skip; if(!count)return 0;
  scratch.resize(count);scratch.copyFrom(x.live,count,0,skip);
  auto ref=x.data.slice(skip,count);
  platform->linAlg->axpby(count,-1.0,ref,1.0,scratch);
  const auto comm=platform->comm.mpiComm();
  dfloat lo=platform->linAlg->min(count,scratch,comm),hi=platform->linAlg->max(count,scratch,comm);
  nekrsCheck(!std::isfinite(lo)||!std::isfinite(hi),comm,EXIT_FAILURE,"%s","Nonfinite history.\n");
  return std::max(std::abs(lo),std::abs(hi));
}
inline bool convergence(int iteration)
{
  nekrsCheck(step!=nrs->tstep || iteration!=pass+1,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","Unexpected bubble Picard pass.\n");
  pass=iteration;
  if(pass==1) {
    uHistory.take(nrs->fluid->o_U);sHistory.take(nrs->scalar->o_S);
    uHistory.data.copyFrom(startU,nrs->fluid->fieldOffsetSum);
    sHistory.data.copyFrom(startS,nrs->scalar->fieldOffsetSum);
    rhsU.take(nrs->fluid->o_JwF);rhsS.take(nrs->scalar->o_JwF);
    extU.take(nrs->fluid->o_EXT);extS.take(nrs->scalar->o_EXT);
    copy(firstU,nrs->fluid->o_U,nrs->fluid->fieldOffsetSum);
    copy(firstS,nrs->scalar->o_S,nrs->scalar->fieldOffsetSum);
    copy(firstP,nrs->fluid->o_P,nrs->fluid->fieldOffset);
    copy(firstDiv,nrs->fluid->o_div,nrs->fieldOffset);
    copy(previousU,firstU,firstU.size());copy(previousS,firstS,firstS.size());
    copy(previousP,firstP,firstP.size());copy(previousDiv,firstDiv,firstDiv.size());
    copy(baseU,rhsU.data,rhsU.data.size());copy(baseS,rhsS.data,rhsS.data.size());
    addLoad(baseU,extUserU,true,-1.0);addLoad(baseS,extUserS,false,-1.0);
  }
  dfloat immutable=std::max(unchanged(uHistory,nrs->fluid->fieldOffsetSum),
                           unchanged(sHistory,nrs->scalar->fieldOffsetSum));
  for(auto *x:{&nativeUrst,&nativeBdf,&fluidExt,&scalarExt,&extU,&extS,&sourceAlphaHistory,&ulPreviousStart})
    immutable=std::max(immutable,unchanged(*x));
  const auto &refU=mode==1?firstU:previousU;
  const auto &refS=mode==1?firstS:previousS;
  auto um=difference(nrs->fluid->o_U,refU,nrs->fluid->fieldOffset,3,umAtol);
  auto pressure=difference(nrs->fluid->o_P,mode==1?firstP:previousP,nrs->fieldOffset,1,pressureAtol);
  const int ai=nrs->scalar->nameToIndex.at("alpha");
  auto alpha=difference(nrs->scalar->o_solution("alpha"),
      refS.slice(nrs->scalar->fieldOffsetScan[ai],nrs->fieldOffset),nrs->fieldOffset,1,alphaAtol);
  dfloat gasError=0,gasMax=0;
  for(const char *name:{"ugx","ugy","ugz"}) {
    const int i=nrs->scalar->nameToIndex.at(name);
    auto q=difference(nrs->scalar->o_solution(name),refS.slice(nrs->scalar->fieldOffsetScan[i],nrs->fieldOffset),
        nrs->fieldOffset,1,ugAtol);
    gasError=std::max(gasError,q.scaled);gasMax=std::max(gasMax,q.maximum);
  }
  auto div=difference(nrs->fluid->o_div,mode==1?firstDiv:previousDiv,nrs->fieldOffset,1,divAtol);
  const dfloat change=std::max({alpha.scaled,gasError,um.scaled,pressure.scaled,div.scaled});
  measuredDiv.resize(nrs->fieldOffset);
  opSEM::strongDivergence(nrs->meshV,nrs->fluid->fieldOffset,nrs->fluid->o_U,measuredDiv,true);
  auto mismatch=difference(measuredDiv,nrs->fluid->o_div,nrs->fieldOffset,1,divAtol);
  const bool converged=mode==2 && pass>1 && change<=1;
  const char *status=mode==1?"frozen":pass==1?"predictor":converged?"converged":pass>=maxPasses?"maxPasses":"iterating";
  if(platform->comm.mpiRank()==0) {
    printf("bubblePicard step=%d pass=%d/%d mode=%d alphaMax=%.8e ugMax=%.8e "
        "umMax=%.8e pMax=%.8e divMax=%.8e scaledMax=%.8e "
        "historyMax=%.8e strongDivMismatchRms=%.8e status=%s\n",step,pass,maxPasses,mode,
        alpha.maximum,gasMax,um.maximum,pressure.maximum,div.maximum,change,immutable,mismatch.rms,status);
    static std::ofstream csv;
    if(!csv.is_open()) {
      std::ifstream old("bubbleColumn_picard.csv");bool header=old.peek()==std::ifstream::traits_type::eof();
      csv.open("bubbleColumn_picard.csv",std::ios::app);
      if(header)csv<<"step,pass,mode,alpha_max,ug_max,um_max,p_max,div_max,scaled_max,history_max,strong_div_mismatch_rms,status\n";
      csv<<std::scientific<<std::setprecision(16);
    }
    csv<<step<<','<<pass<<','<<mode<<','<<alpha.maximum<<','<<gasMax<<','<<um.maximum<<','
       <<pressure.maximum<<','<<div.maximum<<','<<change<<','<<immutable<<','<<mismatch.rms<<','<<status<<'\n';
    csv.flush();
  }
  nekrsCheck(immutable!=0,platform->comm.mpiComm(),EXIT_FAILURE,"%s","Picard changed immutable history.\n");
  if(mode==1) {
    const dfloat error=std::max({alpha.maximum,gasMax,um.maximum,pressure.maximum,div.maximum});
    nekrsCheck(error>repeatTol,platform->comm.mpiComm(),EXIT_FAILURE,
        "Frozen bubble repeat failed: max difference %g.\n",error);
  }
  const bool capped=pass>=maxPasses;
  nekrsCheck(mode==2 && capped && maxPasses>1 && !converged && !acceptAtCap,
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","Bubble Picard reached maxPasses without convergence.\n");
  copy(previousU,nrs->fluid->o_U,nrs->fluid->fieldOffsetSum);
  copy(previousS,nrs->scalar->o_S,nrs->scalar->fieldOffsetSum);
  copy(previousP,nrs->fluid->o_P,nrs->fluid->fieldOffset);
  copy(previousDiv,nrs->fluid->o_div,nrs->fieldOffset);
  return mode==1?capped:(converged||capped);
}

inline void nextPass(double time)
{
  // No history/timestep entry points are called. Keep the latest complete
  // iterate live in coupled mode; frozen mode restores the first solve inputs.
  if(mode==1) {
    uHistory.restore();sHistory.restore();pressureStart.restore();
    propertiesU.restore();propertiesS.restore();dragStart.restore();
    gradStart.restore();liquidStart.restore();ulPreviousStart.restore();
    pressureGradientFilterInitialized=gradInitializedReplay;pressureGradientFilterStep=gradStepStart;
    alphaTransportHistoryStep=alphaHistoryStepStart;
    rhsU.restore();rhsS.restore();
  } else {
    corrector=true;
    picardPressureGradientUpdate=&pressureFilter;
    picardAlphaRhs=&reducedAlphaRhs;
    // Physical-time acceleration anchored to UL^n, never a Picard difference.
    if(p.virtualMassEnabled!=0.0) {
      o_ulPrevious.copyFrom(oldUl,oldUl.size());
      updateVirtualMassHistory();
      o_ulPrevious.copyFrom(oldUl,oldUl.size());
    }
    picardSkipSourceHistory=true;
    addExplicitSources(time);
    picardSkipSourceHistory=false;
    copy(freshUserS,nrs->scalar->o_EXT,nrs->scalar->fieldOffsetSum);
    copy(freshUserU,nrs->fluid->o_explicitTerms(),nrs->fluid->fieldOffsetSum);
    nrs->scalar->o_JwF.copyFrom(baseS,baseS.size());
    addLoad(nrs->scalar->o_JwF,freshUserS,false,1.0);
    // preFluid will rebuild the mixture load with the newly solved scalars.
    nrs->fluid->o_JwF.copyFrom(baseU,baseU.size());
    addLoad(nrs->fluid->o_JwF,freshUserU,true,1.0);
  }
  uPredictor.restore();sPredictor.restore();nativeUrst.restore();
  extU.restore();extS.restore();
  countersStart.restore();
  nrs->checkpointStep=checkpointRequest;
}

inline void configure()
{
  nekrsCheck(mode<0||mode>2||maxPasses<1||(acceptAtCap!=0&&acceptAtCap!=1),
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","Invalid bubble Picard mode/pass/cap controls.\n");
  for(dfloat x:{rtol,alphaAtol,ugAtol,umAtol,pressureAtol,divAtol,repeatTol})
    nekrsCheck(!std::isfinite(x)||x<=0,platform->comm.mpiComm(),EXIT_FAILURE,
        "%s","Picard tolerances must be finite and positive.\n");
  if(!mode)return;
  for(const char *key:{"MOVING MESH","LOWMACH","FLUID PRESSURE RHO SPLITTING",
      "CONSTANT FLOW RATE","NEKNEK MULTIRATE TIMESTEPPER"})
    nekrsCheck(platform->options.compareArgs(key,"TRUE"),platform->comm.mpiComm(),EXIT_FAILURE,
        "Bubble Picard does not support %s.\n",key);
  nekrsCheck(nrs->geom||nrs->neknek||nrs->advectionSubcycingSteps||nrs->Nscalar!=4,
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","Bubble Picard requires one fixed-mesh four-scalar session.\n");
  nekrsCheck(platform->options.compareArgs("FLUID VELOCITY EXTRAPOLATION","FALSE"),
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","Bubble Picard requires a separate native velocity predictor.\n");
  nekrsCheck(nrs->scalar->cvode || nrs->fluid->userAdvectionTerm,
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","Bubble Picard requires elliptic gas scalars and native UM advection.\n");
  nekrsCheck(nrs->preFluid || nrs->userConvergenceCheck,
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","Bubble Picard cannot overwrite existing inner callbacks.\n");
  nekrsCheck(mode==2&&p.divergenceExtrapolationEnabled!=0.0,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","Coupled Picard requires divergenceExtrapolationEnabled=0.\n");
  nrs->userSource=&beginStep;
  nrs->postScalar=&postScalar;
  nrs->preFluid=&preFluid;
  nrs->userConvergenceCheck=&convergence;
  picardAfterScalarPhysics=&afterScalarPhysics;
}
} } // bubbleColumn::picard
