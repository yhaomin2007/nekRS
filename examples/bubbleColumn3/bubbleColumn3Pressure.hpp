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
  opSEM::strongGrad(m,O,pressure,pressureGrad);
  packVelocities();
  correctKernel(m->Nlocal,O,betaG,pressureGrad,ug);
  correctKernel(m->Nlocal,O,betaL,pressureGrad,ul);
  unpackVelocities();
  // Restore phase Dirichlet values, including shared wall/inlet corners.
  s->applyDirichlet(time);
  reconstructMixture();
  if(log) {
    const auto after=divergenceNorm();
    const auto cflG=phaseCFL(ug);
    const auto cflL=phaseCFL(ul);
    const auto mm=platform->linAlg->minMax(m->Nlocal,{pressure,betaM},platform->comm.mpiComm());
    if(platform->comm.mpiRank()==0)
      printf("method3 step=%d alpha->UG/UL_predictors->P->UG/UL_corrected->UM "
          "divL2(pre,post)=(%.8e,%.8e) p=(%.8e,%.8e) mobility=(%.8e,%.8e) "
          "phaseCFL(g,l)=(%.8e,%.8e) dragIterations=%d dragChange=%.8e pIters=%d pResidual=%.8e\n",step,before,after,mm[0].first,mm[0].second,
          mm[1].first,mm[1].second,cflG,cflL,dragIterations,dragChange,pressureSolver->Niter(),pressureSolver->finalResidual());
  }
}
} // namespace bubbleColumn3
