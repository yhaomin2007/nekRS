#pragma once
namespace bubbleColumn4 {
static std::unique_ptr<elliptic> pressureSolver;
static deviceMemory<dfloat> beta,pressureRhs,pressureGrad,flux,boundaryFlux;
static deviceMemory<dfloat> gradientTensor,crossG,crossL,viscG,viscL,fixedLoad;
static deviceMemory<dfloat> previous,defect,zero,lap;
static occa::kernel transposeKernel,pressureFluxKernel,momentumLoadKernel,surfaceKernel;
static int pressureStepDone=-1;

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
inline void allocateFluidStorage()
{
  if(beta.size()) return;
  const auto O=nrs->fieldOffset;
  beta.resize(O);pressureRhs.resize(O);pressureGrad.resize(3*O);
  flux.resize(3*O);boundaryFlux.resize(3*O);gradientTensor.resize(9*O);
  crossG.resize(3*O);crossL.resize(3*O);viscG.resize(3*O);viscL.resize(3*O);
  fixedLoad.resize(3*O);previous.resize(3*O);defect.resize(3*O);zero.resize(O);lap.resize(O);
  platform->linAlg->fill(O,0.0,zero);
}
inline void assembleVector(occa::memory& v)
{
  auto m=nrs->meshV;
  oogs::startFinish(v,3,nrs->fieldOffset,ogsDfloat,ogsAdd,m->oogs);
}
inline dfloat velocityNorm(const occa::memory& v)
{
  return platform->linAlg->weightedNorm2Many(nrs->meshV->Nlocal,3,nrs->fieldOffset,
      nrs->meshV->o_LMM,v,platform->comm.mpiComm());
}
inline void zeroRows(elliptic* solver,occa::memory& v)
{
  if(solver->Nmasked()) launchKernel("core-maskCopy",solver->Nmasked(),0,0,
      solver->o_maskIds(),zero,v);
}
// Kcross_c = integral nu * (d_c u_j) * d_j(test_c), element-local weak load.
// All derivatives here are element-local: no smoothing before forming stress.
inline void crossStress(const deviceMemory<dfloat>& velocity,int first,deviceMemory<dfloat>& out)
{
  auto m=nrs->meshV;const auto O=nrs->fieldOffset;
  opSEM::strongGradVec(m,O,velocity,gradientTensor,false);
  for(int c=0;c<3;++c) {
    dfloat nu=0;
    platform->options.getArgs("SCALAR"+scalarDigitStr(first+c)+" DIFFUSIONCOEFF",nu);
    transposeKernel(m->Nlocal,O,c,nu,m->o_LMM,gradientTensor,flux);
    auto component=out.slice(c*O,m->Nlocal);
    opSEM::divergence(m,O,flux,component);
  }
}
// Full extrapolated viscous acceleration for pressure: -M^-1(Klap+Kcross)ue.
inline void viscousAcceleration(const deviceMemory<dfloat>& velocity,int first,
    deviceMemory<dfloat>& cross,deviceMemory<dfloat>& acceleration)
{
  auto m=nrs->meshV;auto &s=nrs->scalar;const auto O=nrs->fieldOffset;
  crossStress(velocity,first,cross);
  for(int c=0;c<3;++c) {
    auto u=velocity.slice(c*O,m->Nlocal);
    s->ellipticSolver[first+c]->Ax(s->o_diffusionCoeff(names[first+c]),zero,u,lap);
    auto out=acceleration.slice(c*O,m->Nlocal);
    platform->linAlg->axpbyz(m->Nlocal,-1.0,lap,-1.0,cross.slice(c*O,m->Nlocal),out);
  }
  assembleVector(acceleration);
  platform->linAlg->axmyMany(m->Nlocal,3,O,0,1.0,m->o_invLMM,acceleration);
}
inline void pressureStep(double time,int stage)
{
  nekrsCheck(stage!=1 || alphaStepDone!=nrs->tstep || preparedStep!=nrs->tstep || completedStep==nrs->tstep
      || pressureStepDone==nrs->tstep,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","Method4 pressure called out of order.\n");
  allocateFluidStorage();
  auto m=nrs->meshV;auto &s=nrs->scalar;const auto O=nrs->fieldOffset;
  const dfloat c=nrs->g0/nrs->dt[0];
  viscousAcceleration(extrapG,1,crossG,viscG);
  viscousAcceleration(extrapL,4,crossL,viscL);
  auto bg=momentumBase.slice(s->fieldOffsetScan[1],3*O);
  auto bl=momentumBase.slice(s->fieldOffsetScan[4],3*O);
  pressureFluxKernel(m->Nlocal,O,p.rhoGas,p.rhoLiquid,s->o_solution("alpha"),
      m->o_LMM,bg,bl,viscG,viscL,extrapG,extrapL,lambdaG,lambdaL,beta,flux);
  // F contains phase history weighted by NEW alpha. beta has no BDF/drag scale.
  opSEM::divergence(m,O,flux,pressureRhs);
  reconstructMixture();
  platform->linAlg->axpby(3*O,c,nrs->fluid->o_U,0.0,boundaryFlux);
  surfaceKernel(m->Nelements,O,m->o_sgeo,m->o_vmapM,m->o_EToB,boundaryFlux,pressureRhs);
  if(!pressureSolver) {
    auto EToB=m->createEToB([](int id)->int {
      if(id<1) return ellipticBcType::NO_OP;
      return id==2?ellipticBcType::DIRICHLET:ellipticBcType::NEUMANN;
    });
    pressureSolver=std::make_unique<elliptic>("elliptic phasepressure",m,0,EToB,beta,o_NULL);
  }
  auto pressure=nrs->fluid->o_solution("pressure");
  platform->linAlg->fill(O,0.0,pressure); // homogeneous pressure outlet
  pressureSolver->solve(beta,o_NULL,pressureRhs,pressure.slice(0,m->Nlocal));
  if(platform->comm.mpiRank()==0) {
    printf("method4Pressure step=%d time=%.8e iterations=%d initial=%.8e final=%.8e order=alpha,P,UG,UL\n",
        nrs->tstep,time,pressureSolver->Niter(),pressureSolver->initialResidual(),pressureSolver->finalResidual());
    fflush(stdout);
  }
  validateLinearSolve(pressureSolver.get(),"method4 pressure");
  opSEM::strongGrad(m,O,pressure,pressureGrad);
  pressureStepDone=nrs->tstep;
}
// Solve all three components of ONE phase to the full stress residual. These
// are viscosity iterations at FIXED p, not pressure predictors/correctors.
inline void solvePhase(double time,int first,deviceMemory<dfloat>& velocity,
    deviceMemory<dfloat>& cross,dfloat rho,deviceMemory<dfloat>& lambda,dfloat sign)
{
  auto m=nrs->meshV;auto &s=nrs->scalar;const auto O=nrs->fieldOffset;
  const dfloat c=nrs->g0/nrs->dt[0];
  // fixedLoad is independent of stress iterations and of the other phase solve.
  platform->linAlg->fill(3*O,0.0,cross);
  for(int j=0;j<3;++j) {
    auto base=momentumBase.slice(s->fieldOffsetScan[first+j],m->Nlocal);
    auto load=fixedLoad.slice(j*O,m->Nlocal);
    momentumLoadKernel(m->Nlocal,O,j,1/rho,sign,m->o_LMM,base,pressureGrad,
        lambda,extrapG,extrapL,cross.slice(j*O,m->Nlocal),load);
  }
  s->compute.assign(7,0);
  for(int j=0;j<3;++j) s->compute[first+j]=1;
  dfloat residual=0;int iteration=0;
  for(iteration=1;iteration<=p.stressMaxIterations;++iteration) {
    previous.copyFrom(velocity);
    crossStress(velocity,first,cross);
    for(int j=0;j<3;++j) {
      auto rhs=s->o_JwF.slice(s->fieldOffsetScan[first+j],m->Nlocal);
      platform->linAlg->axpbyz(m->Nlocal,1.0,fixedLoad.slice(j*O,m->Nlocal),
          -1.0,cross.slice(j*O,m->Nlocal),rhs);
    }
    s->applyDirichlet(time);s->solve(time,iteration);
    for(int j=0;j<3;++j) {
      validateLinearSolve(s->ellipticSolver[first+j],names[first+j]);
      velocity.copyFrom(s->o_solution(names[first+j]),m->Nlocal,j*O,0);
    }
    // True free-row residual (c M + Klap + Kcross)u - fixedLoad.
    crossStress(velocity,first,cross);
    for(int j=0;j<3;++j) {
      const auto is=first+j;
      auto u=velocity.slice(j*O,m->Nlocal);
      auto out=defect.slice(j*O,m->Nlocal);
      s->ellipticSolver[is]->Ax(s->o_diffusionCoeff(names[is]),zero,u,out);
      platform->linAlg->axpby(m->Nlocal,1.0,cross.slice(j*O,m->Nlocal),1.0,out);
      platform->linAlg->axpby(m->Nlocal,-1.0,fixedLoad.slice(j*O,m->Nlocal),1.0,out);
      platform->linAlg->axmyz(m->Nlocal,c,m->o_LMM,u,lap);
      platform->linAlg->axpby(m->Nlocal,1.0,lap,1.0,out);
    }
    assembleVector(defect);
    platform->linAlg->axmyMany(m->Nlocal,3,O,0,1/c,m->o_invLMM,defect);
    for(int j=0;j<3;++j) {auto out=defect.slice(j*O,m->Nlocal);zeroRows(s->ellipticSolver[first+j],out);}
    residual=velocityNorm(defect)/std::max(std::sqrt(m->volume),velocityNorm(velocity));
    nekrsCheck(!std::isfinite(residual),platform->comm.mpiComm(),EXIT_FAILURE,
        "%s","Nonfinite Method4 momentum residual.\n");
    if(residual<=p.stressTolerance) break;
    // Damped block Jacobi for transpose-gradient coupling.
    platform->linAlg->axpby(3*O,0.5,previous,0.5,velocity);
    for(int j=0;j<3;++j) s->o_solution(names[first+j]).copyFrom(velocity,m->Nlocal,0,j*O);
  }
  if(platform->comm.mpiRank()==0) {
    printf("method4Momentum step=%d phase=%s stressIterations=%d scaledResidual=%.8e target=%.8e\n",
        nrs->tstep,first==1?"gas":"liquid",iteration,residual,p.stressTolerance);
    fflush(stdout);
  }
  nekrsCheck(iteration>p.stressMaxIterations,platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","Method4 full-stress momentum did not converge.\n");
}
inline void velocityStep(double time,int stage)
{
  nekrsCheck(stage!=1 || pressureStepDone!=nrs->tstep || completedStep==nrs->tstep,
      platform->comm.mpiComm(),EXIT_FAILURE,"%s","Method4 velocity called before latest pressure.\n");
  packVelocities();
  solvePhase(time,1,ug,crossG,p.rhoGas,lambdaG,1.0);
  solvePhase(time,4,ul,crossL,p.rhoLiquid,lambdaL,-1.0);
  nrs->scalar->compute.assign(7,1);nrs->scalar->applyDirichlet(time);
  reconstructMixture();
  // Report continuity of FINAL momenta, with the same prescribed boundary
  // lifting as pressure. Never apply a post-solve velocity projection.
  auto m=nrs->meshV;const auto O=nrs->fieldOffset;
  if(p.monitorInterval>0 && (nrs->tstep==1 || nrs->tstep%p.monitorInterval==0)) {
    opSEM::divergence(m,O,nrs->fluid->o_U,pressureRhs);
    surfaceKernel(m->Nelements,O,m->o_sgeo,m->o_vmapM,m->o_EToB,nrs->fluid->o_U,pressureRhs);
    oogs::startFinish(pressureRhs,1,0,ogsDfloat,ogsAdd,m->oogs);
    zeroRows(pressureSolver.get(),pressureRhs);
    const dfloat weak=platform->linAlg->weightedNorm2(m->Nlocal,m->ogs->o_invDegree,
        pressureRhs,platform->comm.mpiComm());
    auto div=opSEM::strongDivergence(m,O,nrs->fluid->o_U);
    const dfloat strong=platform->linAlg->weightedNorm2(m->Nlocal,m->o_LMM,div,platform->comm.mpiComm());
    const auto range=platform->linAlg->minMax(m->Nlocal,{nrs->scalar->o_solution("alpha"),
        nrs->fluid->o_solution("pressure"),lambdaG,lambdaL},platform->comm.mpiComm());
    const dfloat normG=velocityNorm(ug),normL=velocityNorm(ul);
    const dfloat cflG=phaseCFL(ug),cflL=phaseCFL(ul);
    if(platform->comm.mpiRank()==0)
      printf("method4Continuity step=%d time=%.8e weakFreeL2=%.8e strongL2=%.8e "
          "alpha=(%.8e,%.8e) pressure=(%.8e,%.8e) explicitDragDt=%.8e alphaClipMass=%.8e velocityL2=(%.8e,%.8e) phaseCFL=(%.8e,%.8e) order=alpha,P,UG,UL\n",
          nrs->tstep,time,weak,strong,range[0].first,range[0].second,range[1].first,range[1].second,
          nrs->dt[0]*(range[2].second+range[3].second),alphaClipMass,normG,normL,cflG,cflL);
    nekrsCheck(!std::isfinite(weak)||!std::isfinite(strong)||!std::isfinite(normG)||!std::isfinite(normL),platform->comm.mpiComm(),EXIT_FAILURE,
        "%s","Nonfinite Method4 continuity.\n");
  }
  completedStep=nrs->tstep;
}
} // namespace bubbleColumn4
