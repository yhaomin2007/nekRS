#pragma once

namespace bubbleColumn3 {
// Case-local extraction of native solvePressure's weak divergence + boundary
// flux + variable-coefficient Poisson solve. Its native stress/BDF construction
// is replaced by the six scalar Helmholtz predictors, which already include
// BDF, convection, viscosity, gravity and diagonal drag.
// Own this solver here: assigning it to fluid->ellipticSolverP would cause the
// native fluid->solve() call to solve pressure again.
static std::unique_ptr<elliptic> pressureSolver;
static deviceMemory<dfloat> betaG,betaL,betaM,pressureDiffusion,pressureRhs,pressureGrad;
static occa::kernel mobilityKernel,correctKernel,surfaceKernel;

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

// Host Krylov vectors keep this diagnostic implementation independent of
// private elliptic callbacks (which operate on partial element lists). All SEM
// derivatives and gather/scatters still run on the device. This costs host/device
// transfers and is a correctness baseline, not a performance implementation.
inline std::vector<double> pressureHost(const occa::memory& a)
{
  std::vector<dfloat> v(nrs->meshV->Nlocal);a.copyTo(v.data(),v.size());
  return std::vector<double>(v.begin(),v.end());
}
inline void pressureDevice(const std::vector<double>& a,occa::memory& b)
{
  std::vector<dfloat> v(a.begin(),a.end());b.copyFrom(v.data(),v.size());
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
// Same mask is used in the operator AND in the final phase corrections.
// The supported case has identical Dirichlet sets for all six components.
inline std::vector<dlong> phasePressureMask()
{
  auto ids=pressureMask(nrs->scalar->ellipticSolver[1]);bool mismatch=false;
  for(int k=2;k<7;++k) mismatch|=(ids!=pressureMask(nrs->scalar->ellipticSolver[k]));
  nekrsCheck(mismatch,platform->comm.mpiComm(),EXIT_FAILURE,"%s",
      "Method3 projection requires identical Dirichlet masks for all six phase components.\n");
  return ids;
}
inline void maskPressureGradient(const std::vector<dlong>& ids)
{
  const auto O=nrs->fieldOffset;
  for(int c=0;c<3;++c) {
    auto component=pressureGrad.slice(c*O,nrs->meshV->Nlocal);
    auto v=pressureHost(component);zeroPressureRows(v,ids);pressureDevice(v,component);
  }
}
inline void solvePressureAndCorrect(double time,int step)
{
  auto m=nrs->meshV; auto &s=nrs->scalar; const auto O=nrs->fieldOffset;
  if(!pressureDiffusion.size()) allocatePressureStorage();
  const bool log=p.monitorInterval>0 && (step==1 || step%p.monitorInterval==0);
  const double before=log?divergenceNorm():0;
  // Use the inverse local coupled inertia/drag block of both phase predictors.
  const dfloat tau=nrs->dt[0]/nrs->g0;
  nekrsCheck(!std::isfinite(tau) || tau<=0,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","Invalid phase pressure timestep coefficient.\n");
  mobilityKernel(m->Nlocal,tau,p.rhoGas,p.rhoLiquid,
      s->o_solution("alpha"),lambdaG,lambdaL,betaG,betaL,betaM);
  // Positive stiffness A=-div(betaM grad), rhs=-div(UM*), represented as
  // integral grad(test).UM* - integral_N test*(UM_prescribed.n).
  opSEM::divergence(m,O,nrs->fluid->o_U,pressureRhs);
  surfaceKernel(m->Nelements,O,m->o_sgeo,m->o_vmapM,m->o_EToB,
      nrs->fluid->o_U,pressureRhs);
  // Divide BOTH sides by tau. This preserves pressure and all velocity
  // corrections, while removing the small BDF timestep scale from the
  // elliptic coefficient and RHS. Keep physical betaG/L/M unchanged.
  platform->linAlg->axpby(m->Nlocal,1/tau,betaM,0.0,pressureDiffusion);
  platform->linAlg->axpby(m->Nlocal,1/tau,pressureRhs,0.0,pressureRhs);
  if(!pressureSolver) setupPressure(); // normalized coefficient is now ready
  auto pressure=nrs->fluid->o_solution("pressure");
  // Homogeneous pressure outlet. Elliptic::solve masks the RHS internally;
  // initialize the full field to prevent stale outlet values on restart.
  platform->linAlg->fill(O,0.0,pressure);
  const double pressureStart=MPI_Wtime();
  pressureSolver->solve(pressureDiffusion,o_NULL,pressureRhs,pressure.slice(0,m->Nlocal));
  // The user-owned solver is absent from native fluid-pressure step output.
  // Print every solve, independently of monitorInterval, before validation so
  // an unconverged solve also leaves a complete, flushed log record.
  if(platform->comm.mpiRank()==0) {
    const dfloat initial=pressureSolver->initialResidual();
    const dfloat final=pressureSolver->finalResidual();
    const auto tolerance=pressureSolver->options().getArgs("SOLVER TOLERANCE");
    const dfloat target=std::stod(tolerance)*std::sqrt(m->volume)
        *(upperCase(tolerance).find("+RELATIVE")!=std::string::npos?initial:1.0);
    const dfloat threshold=std::max(target,100*std::numeric_limits<dfloat>::epsilon()*initial);
    const dfloat relative=initial>0?final/initial
        :(final==0?0:std::numeric_limits<dfloat>::infinity());
    const bool converged=std::isfinite(initial)&&std::isfinite(final)&&final<=threshold;
    printf("method3Pressure step=%d time=%.8e iterations=%d initial=%.8e final=%.8e "
        "relative=%.8e target=%.8e converged=%s wallTime=%.6es\n",
        step,time,pressureSolver->Niter(),initial,final,relative,threshold,
        converged?"yes":"no",MPI_Wtime()-pressureStart);
    fflush(stdout);
  }
  validateLinearSolve(pressureSolver.get(),"phase pressure");
  // The native stiffness solve above is an initial guess / preconditioner.
  // Solve the actual composed operator W [betaM Mv^-1 assembled(grad)] / tau.
  // W is assembled integral grad(test).u; prescribed boundary flux is fixed
  // from the predictor. Homogeneous phase masks prevent a later BC overwrite
  // from destroying the projection.
  const auto outlet=pressureMask(pressureSolver.get());
  const auto phaseMask=phasePressureMask();
  const auto multiplicity=pressureHost(m->ogs->o_invDegree);
  const auto dot=[&](const std::vector<double>& a,const std::vector<double>& b) {
    double v=0;for(size_t i=0;i<a.size();++i) v+=multiplicity[i]*a[i]*b[i];
    MPI_Allreduce(MPI_IN_PLACE,&v,1,MPI_DOUBLE,MPI_SUM,platform->comm.mpiComm());return v;
  };
  const auto norm=[&](const std::vector<double>& a){return std::sqrt(std::max(0.0,dot(a,a)));};
  deviceMemory<dfloat> trial,flux,assembledRhs;
  trial.resize(O);flux.resize(3*O);assembledRhs.resize(O);
  assembledRhs.copyFrom(pressureRhs,m->Nlocal);
  oogs::startFinish(assembledRhs,1,0,ogsDfloat,ogsAdd,m->oogs);
  auto b=pressureHost(assembledRhs);zeroPressureRows(b,outlet);
  auto preWeak=assembledWeak(nrs->fluid->o_U);
  // Boundary lifting stays fixed for all three reported residuals.
  std::vector<double> boundary(preWeak.size());
  for(size_t i=0;i<b.size();++i) boundary[i]=preWeak[i]-tau*b[i];
  const auto apply=[&](const std::vector<double>& x,std::vector<double>& y) {
    auto q=x;zeroPressureRows(q,outlet);pressureDevice(q,trial);
    opSEM::strongGrad(m,O,trial,pressureGrad);maskPressureGradient(phaseMask);
    platform->linAlg->fill(3*O,0.0,flux);
    // correct(0)=-beta*grad; invert the sign and normalize by tau.
    correctKernel(m->Nlocal,O,betaM,pressureGrad,flux);
    platform->linAlg->axpby(3*O,-1/tau,flux,0.0,flux);
    y=assembledWeak(flux);zeroPressureRows(y,outlet);
  };
  const auto precondition=[&](const std::vector<double>& r,std::vector<double>& z) {
    // Native solve assembles its RHS internally: partition an assembled vector
    // by inverse multiplicity before passing it to that interface.
    auto local=r;for(size_t i=0;i<local.size();++i) local[i]*=multiplicity[i];
    pressureDevice(local,assembledRhs);platform->linAlg->fill(O,0.0,trial);
    pressureSolver->solve(pressureDiffusion,o_NULL,assembledRhs,trial.slice(0,m->Nlocal));
    validateLinearSolve(pressureSolver.get(),"projection preconditioner");
    z=pressureHost(trial);zeroPressureRows(z,outlet);
  };
  auto solution=pressureHost(pressure);
  const auto projected=method3Projection::solve(b,solution,apply,precondition,dot,
      p.pressureProjectionMaxIterations,p.pressureProjectionRestart,p.pressureProjectionTolerance);
  pressureDevice(solution,pressure);
  if(platform->comm.mpiRank()==0) {
    printf("method3Projection step=%d iterations=%d initial=%.8e trueResidual=%.8e converged=%s\n",
        step,projected.iterations,projected.initial,projected.residual,projected.converged?"yes":"no");
    fflush(stdout);
  }
  nekrsCheck(!projected.converged,platform->comm.mpiComm(),EXIT_FAILURE,"%s",
      "Composed phase-pressure projection did not converge; inspect equal-order null modes / boundary compatibility.\n");
  // Compare the old stiffness to the actual correction on identical free rows.
  std::vector<double> actual,old;
  apply(solution,actual);
  pressureSolver->op(pressure,assembledRhs);old=pressureHost(assembledRhs);zeroPressureRows(old,outlet);
  auto difference=old;for(size_t i=0;i<old.size();++i) difference[i]-=actual[i];
  const double mismatch=norm(difference)/std::max(norm(actual),std::numeric_limits<double>::min());
  opSEM::strongGrad(m,O,pressure,pressureGrad);maskPressureGradient(phaseMask);

  packVelocities();
  correctKernel(m->Nlocal,O,betaG,pressureGrad,ug);
  correctKernel(m->Nlocal,O,betaL,pressureGrad,ul);
  unpackVelocities();
  reconstructMixture();
  auto correctedWeak=assembledWeak(nrs->fluid->o_U);
  // Restore phase Dirichlet values, including shared wall/inlet corners.
  s->applyDirichlet(time);
  reconstructMixture();
  if(log) {
    auto restoredWeak=assembledWeak(nrs->fluid->o_U);
    auto bcChange=restoredWeak;
    auto identity=preWeak;
    for(size_t i=0;i<b.size();++i) {
      bcChange[i]-=correctedWeak[i];
      identity[i]=(preWeak[i]-correctedWeak[i])/tau-actual[i];
      preWeak[i]-=boundary[i];correctedWeak[i]-=boundary[i];restoredWeak[i]-=boundary[i];
    }
    zeroPressureRows(preWeak,outlet);zeroPressureRows(correctedWeak,outlet);
    zeroPressureRows(restoredWeak,outlet);zeroPressureRows(bcChange,outlet);zeroPressureRows(identity,outlet);
    const double w0=norm(preWeak),w1=norm(correctedWeak),w2=norm(restoredWeak);
    const double bc=norm(bcChange),id=norm(identity);
    if(platform->comm.mpiRank()==0)
      printf("method3Continuity step=%d weakL2(pre,corrected,BC)=(%.8e,%.8e,%.8e) "
          "boundaryChange=%.8e nativeVsComposedRel=%.8e correctionIdentity=%.8e\n",
          step,w0,w1,w2,bc,mismatch,id);
    const auto after=divergenceNorm();
    const auto cflG=phaseCFL(ug);
    const auto cflL=phaseCFL(ul);
    const auto mm=platform->linAlg->minMax(m->Nlocal,{pressure,betaM},platform->comm.mpiComm());
    if(platform->comm.mpiRank()==0)
      printf("method3 step=%d alpha->UG/UL_predictors->P->UG/UL_corrected->UM "
          "divL2(pre,post)=(%.8e,%.8e) p=(%.8e,%.8e) mobility=(%.8e,%.8e) "
          "phaseCFL(g,l)=(%.8e,%.8e) dragIterations=%d dragChange=%.8e nativePrecoIters=%d nativePrecoResidual=%.8e\n",step,before,after,mm[0].first,mm[0].second,
          mm[1].first,mm[1].second,cflG,cflL,dragIterations,dragChange,pressureSolver->Niter(),pressureSolver->finalResidual());
  }
}
} // namespace bubbleColumn3
