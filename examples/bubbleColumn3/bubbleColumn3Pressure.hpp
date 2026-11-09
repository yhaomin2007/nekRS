#pragma once

namespace bubbleColumn3 {
// Case-local extraction of native solvePressure's weak divergence + boundary
// flux + variable-coefficient Poisson solve. Its native stress/BDF construction
// is replaced by the six scalar Helmholtz predictors, which already include
// BDF, convection, viscosity, gravity and diagonal drag.
// Own this solver here: assigning it to fluid->ellipticSolverP would cause the
// native fluid->solve() call to solve pressure again.
static std::unique_ptr<elliptic> pressureSolver;
static deviceMemory<dfloat> betaG,betaL,betaM,pressureDiffusion,pressureRhs,pressureGrad,pressureIncrement;
static occa::kernel mobilityKernel,surfaceKernel;

inline void allocatePressureStorage()
{
  const auto O=nrs->fieldOffset;
  betaG.resize(O);betaL.resize(O);betaM.resize(O);pressureDiffusion.resize(O);
  pressureRhs.resize(O);pressureGrad.resize(3*O);
}
inline void setupPressure()
{
  auto m=nrs->meshV;
  // Construct the multigrid hierarchy with the actual first-step, normalized
  // coefficient. No dummy coefficient or uninitialized timestep is used.
  auto EToB=m->createEToB([](int id)->int {
    if(id<1) return ellipticBcType::NO_OP;
    return id==2 ? ellipticBcType::DIRICHLET : ellipticBcType::NEUMANN;
  });
  pressureSolver=std::make_unique<elliptic>("elliptic phasepressure",m,0,EToB,pressureDiffusion,o_NULL);
  nekrsCheck(pressureSolver->nullSpace(),platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","bubbleColumn3 requires a pressure outlet with boundary ID 2.\n");
}
// Case-local version of nrs_t::computeCFL(mesh,U,dt), whose overload is
// private. Use the registered native CFL kernel without replacing UM.
inline dfloat phaseCFL(const deviceMemory<dfloat>& velocity)
{
  auto m=nrs->meshV;
  static deviceMemory<dfloat> invSpacing;
  if(!invSpacing.size()) {
    std::vector<dfloat> spacing(m->N+1);
    for(int i=0;i<=m->N;++i) {
      const dfloat dx=(i==0)?m->gllz[1]-m->gllz[0]
          :(i==m->N)?m->gllz[i]-m->gllz[i-1]
          :0.5*(m->gllz[i+1]-m->gllz[i-1]);
      spacing[i]=1/dx;
    }
    invSpacing.resize(spacing.size());invSpacing.copyFrom(spacing);
  }
  auto cfl=platform->deviceMemoryPool.reserve<dfloat>(m->Nelements);
  launchKernel("nrs-cflHex3D",m->Nelements,nrs->dt[0],m->o_vgeo,
      invSpacing,nrs->fieldOffset,velocity,o_NULL,cfl);
  std::vector<dfloat> values(m->Nelements);cfl.copyTo(values.data());
  dfloat result=0;bool bad=false;
  for(const auto value:values) {
    bad|=!std::isfinite(value);result=std::max(result,value);
  }
  // One collective per rank, regardless of its local element count.
  nekrsCheck(bad,platform->comm.mpiComm(),EXIT_FAILURE,"%s","Nonfinite phase CFL.\n");
  MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_DFLOAT,MPI_MAX,platform->comm.mpiComm());
  return result;
}
inline double divergenceNorm()
{
  auto m=nrs->meshV;
  auto div=opSEM::strongDivergence(m,nrs->fieldOffset,nrs->fluid->o_U);
  return platform->linAlg->weightedNorm2(m->Nlocal,m->o_LMM,div,platform->comm.mpiComm());
}

// Host copies below are for monitor-step diagnostics.
inline std::vector<double> pressureHost(const occa::memory& a)
{
  std::vector<dfloat> v(nrs->meshV->Nlocal);a.copyTo(v.data(),v.size());
  return std::vector<double>(v.begin(),v.end());
}
inline std::vector<dlong> pressureMask(elliptic* e)
{
  std::vector<dlong> ids(e->Nmasked());
  if(!ids.empty()) e->o_maskIds().copyTo(ids.data(),ids.size());
  return ids;
}
inline void zeroPressureRows(std::vector<double>& v,const std::vector<dlong>& ids)
{ for(const auto i:ids) v[i]=0; }
inline std::vector<double> assembledWeak(const occa::memory& velocity)
{
  auto m=nrs->meshV;
  deviceMemory<dfloat> r;r.resize(nrs->fieldOffset);
  opSEM::divergence(m,nrs->fieldOffset,velocity,r);
  oogs::startFinish(r,1,0,ogsDfloat,ogsAdd,m->oogs);
  return pressureHost(r);
}
// elliptic::_solve clears its internal coefficient pointers before returning.
// op() requires those pointers to be bound and is NOT safe after solve().
// Ax() binds the supplied coefficients for this call and returns element-local
// contributions. Assemble those here; callers mask pressure Dirichlet rows.
inline void nativePressureAction(const occa::memory& q,occa::memory& result)
{
  pressureSolver->Ax(pressureDiffusion,o_NULL,q,result);
  oogs::startFinish(result,1,0,ogsDfloat,ogsAdd,nrs->meshV->oogs);
}
// Match fluidSolver_t::solveVelocity: unassembled weak pressure load enters
// the momentum RHS BEFORE the native Helmholtz solve. With unit scalar
// transport coefficients, divide the force equation by each phase density.
inline void addNativePhasePressureLoad(const occa::memory& pressure)
{
  auto m=nrs->meshV;auto &s=nrs->scalar;const auto O=nrs->fieldOffset;
  launchKernel("core-wGradientVolumeHex3D",m->Nelements,m->o_vgeo,m->o_D,O,pressure,pressureGrad);
  for(int c=0;c<3;++c) {
    auto gp=pressureGrad.slice(c*O,m->Nlocal);
    auto gas=predictorBase.slice(s->fieldOffsetScan[1+c],m->Nlocal);
    auto liquid=predictorBase.slice(s->fieldOffsetScan[4+c],m->Nlocal);
    platform->linAlg->axpby(m->Nlocal,1/p.rhoGas,gp,1.0,gas);
    platform->linAlg->axpby(m->Nlocal,1/p.rhoLiquid,gp,1.0,liquid);
  }
}
inline void solveNativePressureVelocities(double time,const occa::memory& pressure,
                                          bool homogeneous=false,dfloat dragTolerance=0)
{
  auto &s=nrs->scalar;const auto O=nrs->fieldOffset,N=nrs->meshV->Nlocal;
  static deviceMemory<dfloat> pressureFreeBase;
  if(!pressureFreeBase.size()) pressureFreeBase.resize(6*O);
  for(int is=1;is<7;++is) {
    const auto offset=s->fieldOffsetScan[is];
    pressureFreeBase.copyFrom(predictorBase,N,(is-1)*O,offset);
    if(homogeneous) {
      platform->linAlg->fill(N,0.0,predictorBase.slice(offset,N));
      platform->linAlg->fill(N,0.0,s->o_solution(names[is]));
    }
  }
  addNativePhasePressureLoad(pressure);
  // Reuse frozen BDF/EXT RHS and drag coefficients, not makeForcing/lagSolution.
  // Both partner velocities converge with the SAME fixed, latest pressure.
  solvePhasePredictors(time,homogeneous,dragTolerance);
  for(int is=1;is<7;++is)
    predictorBase.copyFrom(pressureFreeBase,N,s->fieldOffsetScan[is],(is-1)*O);
  reconstructMixture();
}
// Native-only baseline: one variable-coefficient Poisson solve per timestep.
// This solves K p=b, not the stabilized composed equation S p=b. Keep the
// continuity diagnostics explicit: a small K residual is not zero nodal div(U).
inline void solveNativePhasePressure(double time,int step,bool log,double before,dfloat tau,bool incremental)
{
  auto m=nrs->meshV;const auto O=nrs->fieldOffset;
  auto pressure=nrs->fluid->o_solution("pressure");
  static bool initialized=false;
  occa::memory pressureUpdate=pressure;
  if(incremental) {
    const bool firstIncrement=!pressureIncrement.size();
    if(firstIncrement) pressureIncrement.resize(O);
    pressureUpdate=pressureIncrement;
    if(firstIncrement || !p.pressureWarmStart)
      platform->linAlg->fill(O,0.0,pressureUpdate);
  } else if(!initialized || !p.pressureWarmStart) {
    platform->linAlg->fill(O,0.0,pressureUpdate);
  }
  initialized=true;
  if(pressureSolver->Nmasked()) platform->linAlg->mask<dfloat>(
      pressureSolver->Nmasked(),pressureSolver->o_maskIds(),pressureUpdate);
  const char* pressureForm=incremental?"increment":"absolute";
  const double start=MPI_Wtime();
  pressureSolver->solve(pressureDiffusion,o_NULL,pressureRhs,pressureUpdate.slice(0,m->Nlocal));
  if(platform->comm.mpiRank()==0) {
    const double initial=pressureSolver->initialResidual(),final=pressureSolver->finalResidual();
    const auto tolerance=pressureSolver->options().getArgs("SOLVER TOLERANCE");
    const double target=std::stod(tolerance)*std::sqrt(m->volume)
        *(upperCase(tolerance).find("+RELATIVE")!=std::string::npos?initial:1.0);
    const double threshold=std::max(target,100*std::numeric_limits<dfloat>::epsilon()*initial);
    const bool converged=std::isfinite(initial)&&std::isfinite(final)&&final<=threshold;
    printf("method3Pressure step=%d time=%.8e mode=native pressureForm=%s iterations=%d initial=%.8e "
        "final=%.8e relative=%.8e target=%.8e converged=%s wallTime=%.6es\n",
        step,time,pressureForm,pressureSolver->Niter(),initial,final,initial>0?final/initial:final,
        threshold,converged?"yes":"no",MPI_Wtime()-start);
    fflush(stdout);
  }
  validateLinearSolve(pressureSolver.get(),"native phase pressure");
  std::vector<double> preWeak,boundary,b,k,degree;
  std::vector<dlong> outlet;
  deviceMemory<dfloat> work;
  if(log) {
    work.resize(O);work.copyFrom(pressureRhs,m->Nlocal);
    oogs::startFinish(work,1,0,ogsDfloat,ogsAdd,m->oogs);
    b=pressureHost(work);preWeak=assembledWeak(nrs->fluid->o_U);
    boundary.resize(b.size());
    for(size_t i=0;i<b.size();++i) boundary[i]=preWeak[i]-tau*b[i];
    nativePressureAction(pressureUpdate,work);k=pressureHost(work);
    outlet=pressureMask(pressureSolver.get());degree=pressureHost(m->ogs->o_invDegree);
  }
  if(incremental) platform->linAlg->axpby(m->Nlocal,1.0,pressureUpdate,1.0,pressure);
  solveNativePressureVelocities(time,pressure);
  std::vector<double> corrected;
  if(log) corrected=assembledWeak(nrs->fluid->o_U);
  nrs->scalar->applyDirichlet(time);reconstructMixture();
  if(platform->comm.mpiRank()==0) {
    printf("method3PressureCost step=%d mode=native velocityPressure=weakRhsHelmholtz nativeSolves=1 innerCalls=0 totalWallTime=%.6es\n",
        step,MPI_Wtime()-start);
    fflush(stdout);
  }
  if(log) {
    auto restored=assembledWeak(nrs->fluid->o_U);
    std::vector<double> r0=b,r1=b,r2=b,change=b,nativeResidual=b,defect=b;
    for(size_t i=0;i<b.size();++i) {
      r0[i]=tau*b[i];r1[i]=corrected[i]-boundary[i];r2[i]=restored[i]-boundary[i];
      change[i]=restored[i]-corrected[i];nativeResidual[i]=tau*(b[i]-k[i]);
      defect[i]=tau*k[i]-(preWeak[i]-corrected[i]);
    }
    const auto norm=[&](std::vector<double> v) {
      zeroPressureRows(v,outlet);double sum=0;
      for(size_t i=0;i<v.size();++i) sum+=degree[i]*v[i]*v[i];
      MPI_Allreduce(MPI_IN_PLACE,&sum,1,MPI_DOUBLE,MPI_SUM,platform->comm.mpiComm());
      return std::isfinite(sum)&&sum>=0?std::sqrt(sum):std::numeric_limits<double>::infinity();
    };
    const auto w0=norm(r0),w1=norm(r1),w2=norm(r2),bc=norm(change);
    const auto nr=norm(nativeResidual),nd=norm(defect);
    auto closure=r1;
    for(size_t i=0;i<b.size();++i) closure[i]-=nativeResidual[i]+defect[i];
    const auto balance=norm(closure),after=divergenceNorm();
    const auto cflG=phaseCFL(ug),cflL=phaseCFL(ul);
    const auto mm=platform->linAlg->minMax(m->Nlocal,{pressure,betaM},platform->comm.mpiComm());
    if(platform->comm.mpiRank()==0) {
      printf("method3Continuity step=%d mode=native pressureForm=%s weakL2(pre,corrected,BC)=(%.8e,%.8e,%.8e) "
          "boundaryChange=%.8e nativeEquationResidual=%.8e correctionDefect=%.8e residualBalanceError=%.8e\n",
          step,pressureForm,w0,w1,w2,bc,nr,nd,balance);
      printf("method3 step=%d mode=native divL2(pre,post)=(%.8e,%.8e) p=(%.8e,%.8e) "
          "mobility=(%.8e,%.8e) phaseCFL(g,l)=(%.8e,%.8e) dragIterations=%d dragChange=%.8e\n",
          step,before,after,mm[0].first,mm[0].second,mm[1].first,mm[1].second,
          cflG,cflL,dragIterations,dragChange);
      fflush(stdout);
    }
  }
}
// Assemble the same weak continuity residual for the predictor and all trials.
inline void formPressureRhs(dfloat tau)
{
  auto m=nrs->meshV;
  const auto O=nrs->fieldOffset;
  opSEM::divergence(m,O,nrs->fluid->o_U,pressureRhs);
  surfaceKernel(m->Nelements,O,m->o_sgeo,m->o_vmapM,m->o_EToB,
      nrs->fluid->o_U,pressureRhs);
  platform->linAlg->axpby(m->Nlocal,1/tau,pressureRhs,0.0,pressureRhs);
}

inline void solveConsistentPhasePressure(double time,int step,bool log,double before,
                                        dfloat tau,bool incremental,bool havePreviousPressure)
{
  auto m=nrs->meshV;
  const auto N=m->Nlocal,O=nrs->fieldOffset;
  auto pressure=nrs->fluid->o_solution("pressure");
  const auto& degree=m->ogs->o_invDegree;
  const auto comm=platform->comm.mpiComm();
  auto* linalg=platform->linAlg;
  const auto mask=[&](occa::memory& vector) {
    if(pressureSolver->Nmasked()) linalg->mask<dfloat>(
        pressureSolver->Nmasked(),pressureSolver->o_maskIds(),vector);
  };
  using Vector=deviceMemory<dfloat>;
  const auto axpby=[&](double a,const Vector& x,double b,Vector& y) {
    linalg->axpby(N,a,x,b,y);
  };
  const auto scale=[&](double a,Vector& x) {linalg->scale(N,a,x);};
  const auto dot=[&](const Vector& a,const Vector& b) {
    return linalg->weightedInnerProd(N,degree,a,b,comm);
  };
  const auto norm=[&](const occa::memory& v) {return linalg->weightedNorm2(N,degree,v,comm);};
  const auto continuity=[&]() {
    // Native elliptic::solve expects unassembled element RHSs. The outer
    // iteration uses assembled rows, counting each shared node only once.
    oogs::startFinish(pressureRhs,1,0,ogsDfloat,ogsAdd,m->oogs);
    mask(pressureRhs);
  };
  continuity();
  const double predictorNorm=norm(pressureRhs);
  const double target=std::max(p.pressureSchurAbsoluteTolerance/tau,
                               p.pressureSchurTolerance*predictorNorm);
  nekrsCheck(!std::isfinite(predictorNorm) || !std::isfinite(target),comm,EXIT_FAILURE,
      "%s","Nonfinite predictor continuity residual or Schur tolerance.\n");
  static Vector totalPressure,trialPressure,actualResidual;
  totalPressure.resize(O);trialPressure.resize(O);actualResidual.resize(O);
  const bool usePreviousPressure=incremental || (havePreviousPressure && p.pressureWarmStart);
  if(usePreviousPressure) totalPressure.copyFrom(pressure,N);
  else linalg->fill(N,0.0,totalPressure);
  mask(totalPressure);

  static Vector predictorGas,predictorLiquid;
  predictorGas.resize(3*O);predictorLiquid.resize(3*O);
  predictorGas.copyFrom(ug,3*O);predictorLiquid.copyFrom(ul,3*O);
  int phaseEvaluations=0,nativeSolves=0,iterations=0,passes=0;
  double linearResidual=0,phaseTime=0,preconditionerTime=0;
  const char* reason="postBcResidual";
  const double start=MPI_Wtime();
  const auto evaluate=[&](bool reusePredictor) {
    const double phaseStart=MPI_Wtime();
    if(!reusePredictor) {
      ug.copyFrom(predictorGas,3*O);ul.copyFrom(predictorLiquid,3*O);
      unpackVelocities();
      solveNativePressureVelocities(time,totalPressure,false,std::min(p.dragCouplingTolerance,dfloat(1e-12)));
      ++phaseEvaluations;
    }
    nrs->scalar->applyDirichlet(time);
    reconstructMixture();
    formPressureRhs(tau);
    continuity();
    actualResidual.copyFrom(pressureRhs,N);
    const double residualNorm=norm(actualResidual);
    phaseTime+=MPI_Wtime()-phaseStart;
    return residualNorm;
  };
  const auto apply=[&](const Vector& increment,Vector& action) {
    const double phaseStart=MPI_Wtime();
    // Scale tiny Krylov directions before the homogeneous Helmholtz solves.
    const double magnitude=norm(increment);
    nekrsCheck(!std::isfinite(magnitude),comm,EXIT_FAILURE,"%s","Nonfinite Schur pressure direction.\n");
    if(magnitude!=0) {
      axpby(1/magnitude,increment,0,trialPressure);
      mask(trialPressure);
      solveNativePressureVelocities(time,trialPressure,true);
      formPressureRhs(tau);
      continuity();
      axpby(-magnitude,pressureRhs,0,action);
      ++phaseEvaluations;
    } else {
      linalg->fill(N,0.0,action);
    }
    phaseTime+=MPI_Wtime()-phaseStart;
  };
  const auto nativeTolerance=pressureSolver->options().getArgs("SOLVER TOLERANCE");
  pressureSolver->options().setArgs("SOLVER TOLERANCE","1e-2+relative");
  const auto precondition=[&](const Vector& residual,Vector& direction) {
    const double preconditionerStart=MPI_Wtime();
    linalg->axmyz(N,1.0,degree,residual,pressureRhs);
    linalg->fill(N,0.0,direction);
    pressureSolver->solve(pressureDiffusion,o_NULL,pressureRhs,direction);
    validateLinearSolve(pressureSolver.get(),"Schur Poisson preconditioner");
    mask(direction);
    ++nativeSolves;
    preconditionerTime+=MPI_Wtime()-preconditionerStart;
  };
  // An incremental predictor already includes totalPressure. A pressure-free
  // predictor also matches the initial guess when it is zero.
  double final=evaluate(incremental || !usePreviousPressure);
  static Vector directions,images,correction,residual,action,combination,deviceCoefficients;
  static std::vector<dfloat> coefficients;
  const int restart=std::min(p.pressureSchurRestart,p.pressureSchurMaxIterations);
  if(final>target) {
    directions.resize(restart*O);images.resize(restart*O);
    correction.resize(O);residual.resize(O);action.resize(O);combination.resize(O);
    deviceCoefficients.resize(restart);coefficients.resize(restart);
  }
  const auto direction=[&](int k) {return directions.slice(k*O,N);};
  const auto image=[&](int k) {return images.slice(k*O,N);};
  const auto combinationKernel=std::string("gmres::")+
      (std::is_same<dfloat,double>::value?"double":"float")+"::1::PGMRESSolution";
  const auto orthogonalize=[&](int k,Vector& v,Vector& w) {
    // Two batched reductions per GCR step. The existing GMRES kernel forms
    // the corresponding basis combinations without transferring field data.
    linalg->weightedInnerProdMulti(N,k,1,O,degree,images,w,comm,coefficients.data());
    deviceCoefficients.copyFrom(coefficients,k);
    launchKernel(combinationKernel,N,O,k,deviceCoefficients,images,combination);
    axpby(-1,combination,1,w);
    launchKernel(combinationKernel,N,O,k,deviceCoefficients,directions,combination);
    axpby(-1,combination,1,v);
  };
  struct SchurSample { int pass,iteration; double norm; };
  std::vector<SchurSample> history;
  // A fresh physical residual corrects the small error from inexact phase
  // inverses. Refinement shares the configured outer iteration budget.
  while(std::isfinite(final) && final>target && passes<3 && iterations<p.pressureSchurMaxIterations) {
    linalg->fill(N,0.0,correction);
    const auto result=method3Projection::solveWithWorkspace(actualResidual,correction,residual,action,
        apply,precondition,dot,axpby,scale,direction,image,orthogonalize,
        p.pressureSchurMaxIterations-iterations,restart,0.0,0.25*target);
    for(const auto& sample : result.residualHistory)
      history.push_back({passes+1,iterations+sample.iteration,sample.norm});
    iterations+=result.iterations;
    linearResidual=result.residual;
    reason=result.reason;
    ++passes;
    if(!result.iterations || !std::isfinite(linearResidual)) break;
    axpby(1,correction,1,totalPressure);
    const double previous=final;
    final=evaluate(false);
    if(final>=previous) {reason="physicalResidualStalled";break;}
    if(result.converged && final>target) reason="postBcResidual";
  }
  pressureSolver->options().setArgs("SOLVER TOLERANCE",nativeTolerance);
  const bool converged=std::isfinite(final) && final<=target;
  if(platform->comm.mpiRank()==0) {
    if(log || !converged) {
      for(const auto& sample : history)
        printf("method3SchurRestart step=%d pass=%d iteration=%d weakLinearL2=%.8e\n",
            step,sample.pass,sample.iteration,tau*sample.norm);
    }
    printf("method3Schur step=%d time=%.8e pressureForm=%s iterations=%d restart=%d phaseEvaluations=%d "
        "nativePreconditionerSolves=%d passes=%d weakL2(predictor,final)=(%.8e,%.8e) target=%.8e "
        "linearResidual=%.8e reason=%s converged=%s phaseTime=%.6es preconditionerTime=%.6es wallTime=%.6es\n",step,time,
        incremental?"increment":"absolute",iterations,restart,phaseEvaluations,nativeSolves,passes,
        tau*predictorNorm,tau*final,tau*target,tau*linearResidual,
        converged?"converged":reason,converged?"yes":"no",phaseTime,preconditionerTime,MPI_Wtime()-start);
    fflush(stdout);
  }
  nekrsCheck(!converged,comm,EXIT_FAILURE,
      "Method3 Schur solve did not reach the post-BC continuity target: weakL2=%g target=%g. "
      "Inspect phase-solve accuracy, iteration history and pressure-space compatibility.\n",
      tau*final,tau*target);
  pressure.copyFrom(totalPressure,N);
  if(log) {
    const auto after=divergenceNorm();
    const auto cflG=phaseCFL(ug),cflL=phaseCFL(ul);
    const auto range=platform->linAlg->minMax(N,{pressure},comm);
    if(platform->comm.mpiRank()==0) {
      printf("method3 step=%d mode=consistent divL2(pre,post)=(%.8e,%.8e) p=(%.8e,%.8e) "
          "phaseCFL(g,l)=(%.8e,%.8e) dragIterations=%d dragChange=%.8e\n",step,before,after,
          range[0].first,range[0].second,cflG,cflL,dragIterations,dragChange);
      fflush(stdout);
    }
  }
}

inline void solvePressureAndCorrect(double time,int step)
{
  auto m=nrs->meshV; auto &s=nrs->scalar; const auto O=nrs->fieldOffset;
  if(!pressureDiffusion.size()) allocatePressureStorage();
  // Bootstrap absolute pressure once. Afterwards use the previous final
  // pressure in the predictor and solve only the increment. The predictor
  // and final solves reuse the same BDF/EXT base and frozen drag coefficients.
  const bool consistent=p.pressureSolveMode=="consistent";
  std::array<std::string,6> phaseTolerances;
  if(consistent) for(int is=1;is<7;++is) {
    auto &options=s->ellipticSolver[is]->options();
    phaseTolerances[is-1]=options.getArgs("SOLVER TOLERANCE");
    options.setArgs("SOLVER TOLERANCE","1e-11+relative");
  }
  const dfloat dragTolerance=consistent?std::min(p.dragCouplingTolerance,dfloat(1e-12)):0;
  const bool havePreviousPressure=bool(pressureSolver);
  const bool incremental=p.pressureIncremental && havePreviousPressure;
  if(incremental) {
    solveNativePressureVelocities(time,nrs->fluid->o_solution("pressure"),false,dragTolerance);
  } else {
    solvePhasePredictors(time,false,dragTolerance);
    reconstructMixture();
  }
  const bool log=p.monitorInterval>0 && (step==1 || step%p.monitorInterval==0);
  const double before=log?divergenceNorm():0;
  // Use the inverse local coupled inertia/drag block of both phase predictors.
  const dfloat tau=nrs->dt[0]/nrs->g0;
  nekrsCheck(!std::isfinite(tau) || tau<=0,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","Invalid phase pressure timestep coefficient.\n");
  mobilityKernel(m->Nlocal,tau,p.rhoGas,p.rhoLiquid,
      s->o_solution("alpha"),lambdaG,lambdaL,betaG,betaL,betaM);
  formPressureRhs(tau);
  platform->linAlg->axpby(m->Nlocal,1/tau,betaM,0.0,pressureDiffusion);
  if(!pressureSolver) setupPressure(); // normalized coefficient is now ready
  if(consistent) {
    solveConsistentPhasePressure(time,step,log,before,tau,incremental,havePreviousPressure);
  } else {
    solveNativePhasePressure(time,step,log,before,tau,incremental);
  }
  if(consistent) for(int is=1;is<7;++is)
    s->ellipticSolver[is]->options().setArgs("SOLVER TOLERANCE",phaseTolerances[is-1]);
}
} // namespace bubbleColumn3
