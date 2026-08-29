#pragma once

#include "platform.hpp"

class fluidSolver_t;
class geomSolver_t;

// Experimental, constant-volume-fraction two-fluid coordinator.
//
// Scope of the first branch:
//   * constant rho and mu for each phase
//   * prescribed, spatially uniform gas volume fraction
//   * a common mechanical pressure
//   * gravity and Schiller--Naumann drag
//   * diagonal-implicit drag with outer correction iterations
//   * periodic or impermeable boundaries only
class twoFluid_t
{
public:
  twoFluid_t(fluidSolver_t *liquid,
             fluidSolver_t *gas,
             const std::unique_ptr<geomSolver_t> &geom);

  void setup();
  void updateAdvectionCoordinates();
  void makeExplicit(double time);
  void refreshCouplingForcing(double time, int tstep);
  void finalizeCouplingForcing();
  void solvePressure(double time, int stage);
  void correctMixtureContinuity(double time);
  void updateDiagnostics();
  void reportContinuity(const char *label = nullptr);

  fluidSolver_t *liquid = nullptr;
  fluidSolver_t *gas = nullptr;

  dfloat alphaG = 0;
  dfloat alphaL = 1;
  dfloat bubbleDiameter = 1e-3;
  dfloat gravity[3] = {0, 0, 0};
  dfloat alphaFloor = 1e-8;
  dfloat dragMultiplier = 1;
  int couplingIterations = 2;
  int pressureCorrectors = 2;
  bool projectionOnly = false;

  occa::memory o_alphaG;
  occa::memory o_alphaL;
  occa::memory o_drag;
  occa::memory o_implicitLiquid;
  occa::memory o_implicitGas;
  occa::memory o_slipVelocity;
  occa::memory o_mixtureVelocity;
  occa::memory o_interphaseForce;
  occa::memory o_continuityResidual;
  occa::memory o_pressureResponseLiquid;
  occa::memory o_pressureResponseGas;

private:
  const std::unique_ptr<geomSolver_t> &geom;
  occa::memory o_pressureCoeff;
  occa::memory o_mixtureFlux;
  occa::memory o_baseExtLiquid;
  occa::memory o_baseExtGas;
  occa::memory o_divergenceLiquid;
  occa::memory o_divergenceGas;

  void phasePressureFlux(fluidSolver_t *phase, occa::memory o_flux);
  void weakDivergence(const occa::memory &o_velocity, occa::memory o_divergence);
  void updatePressureResponse(bool scaleForMomentumPredictor);
};

void registerTwoFluidKernels();
