#pragma once

namespace bubbleColumn3 {
// Case-local extraction of native solvePressure's weak divergence + boundary
// flux + variable-coefficient Poisson solve. Its native stress/BDF construction
// is replaced by the six scalar Helmholtz predictors, which already include
// BDF, convection, viscosity, gravity and diagonal drag.
// Own this solver here: assigning it to fluid->ellipticSolverP would cause the
// native fluid->solve() call to solve pressure again.
static std::unique_ptr<elliptic> pressureSolver;
static deviceMemory<dfloat> betaG,betaL,betaM,pressureRhs,pressureGrad;
static occa::kernel mobilityKernel,correctKernel,surfaceKernel;

inline void setupPressure()
{
  auto m=nrs->meshV; const auto O=nrs->fieldOffset;
  betaG.resize(O);betaL.resize(O);betaM.resize(O);
  pressureRhs.resize(O);pressureGrad.resize(3*O);
  platform->linAlg->fill(O,1.0/p.rhoLiquid,betaM);
  auto EToB=m->createEToB([](int id)->int {
    if(id<1) return ellipticBcType::NO_OP;
    return id==2 ? ellipticBcType::DIRICHLET : ellipticBcType::NEUMANN;
  });
  pressureSolver=std::make_unique<elliptic>("elliptic phasepressure",m,0,EToB,betaM,o_NULL);
  nekrsCheck(pressureSolver->nullSpace(),platform->comm.mpiComm(),EXIT_FAILURE,
      "%s","bubbleColumn3 requires a pressure outlet with boundary ID 2.\n");
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
  const bool log=p.monitorInterval>0 && (step==1 || step%p.monitorInterval==0);
  const double before=log?divergenceNorm():0;
  // Include precisely the diagonal inertia/drag used by both predictors.
  mobilityKernel(m->Nlocal,nrs->dt[0]/nrs->g0,p.rhoGas,p.rhoLiquid,
      s->o_solution("alpha"),lambdaG,lambdaL,betaG,betaL,betaM);
  // Positive stiffness A=-div(betaM grad), rhs=-div(UM*), represented as
  // integral grad(test).UM* - integral_N test*(UM_prescribed.n).
  opSEM::divergence(m,O,nrs->fluid->o_U,pressureRhs);
  surfaceKernel(m->Nelements,O,m->o_sgeo,m->o_vmapM,m->o_EToB,
      nrs->fluid->o_U,pressureRhs);
  auto pressure=nrs->fluid->o_solution("pressure");
  // Homogeneous pressure outlet. Elliptic::solve masks the RHS internally;
  // initialize the full field to prevent stale outlet values on restart.
  platform->linAlg->fill(O,0.0,pressure);
  pressureSolver->solve(betaM,o_NULL,pressureRhs,pressure);
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
    const auto mm=platform->linAlg->minMax(m->Nlocal,{pressure,betaM},platform->comm.mpiComm());
    if(platform->comm.mpiRank()==0)
      printf("method3 step=%d alpha->UG/UL_predictors->P->UG/UL_corrected->UM "
          "divL2(pre,post)=(%.8e,%.8e) p=(%.8e,%.8e) mobility=(%.8e,%.8e) "
          "pIters=%d pResidual=%.8e\n",step,before,after,mm[0].first,mm[0].second,
          mm[1].first,mm[1].second,pressureSolver->Niter(),pressureSolver->finalResidual());
  }
}
} // namespace bubbleColumn3
