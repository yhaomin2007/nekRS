#if !defined(nrs_hydro_hpp_)
#define nrs_hydro_hpp_

#include "solver.hpp"
#include "geomSolver.hpp"

class fluidSolverCfg_t : public solverCfg_t
{
public:
  std::string velocityName;
  std::string pressureName;
  dlong fieldOffset;
  dlong cubatureOffset;
  bool createPressureSolver = true;
};

class twoFluid_t;

class fluidSolver_t : public solver_t
{
private:
  void advectionSubcycling(int nEXT, double time);

  occa::memory o_zeroNormalMask;
  occa::memory o_filterRT;
  int Nsubsteps;

  occa::memory o_ADV;

  occa::memory o_U0;
  occa::memory o_P0;
  occa::memory o_relUrst0;
  occa::memory o_prop0;
  occa::memory o_EXT0;

  occa::memory o_coeffEXTP;

public:
  fluidSolver_t(const fluidSolverCfg_t &cfg, const std::unique_ptr<geomSolver_t> &geom);
  void solvePressure(double time, int stage);
  void solveVelocity(double time, int stage);

  deviceMemory<dfloat> o_solution(std::string key = "") override
  {
    if (key.empty()) {
      return deviceMemory<dfloat>(o_U);
    }

    if (lowerCase(key) == "p" || lowerCase(key) == "pressure") {
      return deviceMemory<dfloat>(o_P);
    }

    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_U.slice(idx * fieldOffset, fieldOffset))
                      : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_explicitTerms(std::string key = "") override
  {
    if (key.empty()) {
      return deviceMemory<dfloat>(o_EXT);
    }
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_EXT.slice(idx * fieldOffset, fieldOffset))
                      : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_diffusionCoeff(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_mue);
  };

  deviceMemory<dfloat> o_transportCoeff(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_rho);
  }

  void setTimeIntegrationCoeffs(int tstep) override;

  void lagSolution() override;
  void extrapolateSolution() override;

  void saveSolutionState() override;
  void restoreSolutionState() override;

  void applyDirichlet(double time) override;
  void setupEllipticSolver() override;

  void makeAdvection(double time, int tstep);
  void makeExplicit(double time, int tstep);

  void solve(double time, int stage) override
  {
    // In two-fluid mode the coordinator owns the shared-pressure correction.
    // Each phase solve here is therefore momentum-only; this avoids running an
    // independent NekRS pressure predictor in addition to the mixture-pressure
    // correction performed later in the coupling iteration.
    if (createPressureSolver && !twoFluid) solvePressure(time, stage);
    solveVelocity(time, stage);
  };

  void makeForcing(bool shiftHistory = true);

  void updateZeroNormalMask()
  {
    if (platform->app->bc->hasUnalignedMixed(name)) {
      o_zeroNormalMask = mesh->createZeroNormalMask(fieldOffset, ellipticSolver[0]->o_EToB());
    }
  };

  mesh_t *mesh = nullptr;

  elliptic *ellipticSolverP = nullptr;

  const std::unique_ptr<geomSolver_t> &geom;

  dlong fieldOffset = -1;
  dlong cubatureOffset = -1;

  std::string velocityName;
  occa::memory o_velocityName;

  std::string pressureName;
  occa::memory o_pressureName;

  std::function<occa::memory(double)> userImplicitLinearTerm = nullptr;
  std::function<occa::memory(double, int)> userAdvectionTerm = nullptr;

  // Set by the experimental Eulerian--Eulerian coordinator.  The primary
  // phase owns the shared pressure storage/elliptic solver, but in two-fluid
  // mode pressure advancement is controlled by twoFluid_t rather than by the
  // standard single-fluid solve() sequence.
  twoFluid_t *twoFluid = nullptr;
  bool createPressureSolver = true;

  occa::memory o_U;
  occa::memory o_Ue;
  occa::memory o_div;

  occa::memory o_P;
  occa::memory o_Pe;

  dfloat rho0 = NAN;
  occa::memory o_rho;
  occa::memory o_mue;

  occa::memory o_relUrst;

  void finalize() override
  {
    for (auto &entry : ellipticSolver) {
      delete entry;
      entry = nullptr;
    }
    delete ellipticSolverP;
    ellipticSolverP = nullptr;
  };
};

#endif
