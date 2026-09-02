#pragma once

#include "platform.hpp"

class fluidSolver_t;
class geomSolver_t;
class linearSolver;
class elliptic;
class scalar_t;

// Experimental two-fluid coordinator with conservative gas-volume transport.
//
// Scope of the first branch:
//   * constant rho and mu for each phase
//   * conservative gas-volume-fraction transport and alpha_l = 1 - alpha_g
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
  ~twoFluid_t();

  void setup();
  void beginTimeStep();
  void advanceVolumeFraction(int couplingIteration);
  void updateAdvectionCoordinates();
  void makeExplicit(double time);
  void refreshCouplingForcing(double time, int tstep);
  void finalizeCouplingForcing();
  void solvePressure(double time, int stage);
  void correctMixtureContinuity(double time, const char *stageLabel = nullptr);
  void updateDiagnostics();
  void reportContinuity(const char *label = nullptr);

  fluidSolver_t *liquid = nullptr;
  fluidSolver_t *gas = nullptr;

  dfloat alphaG = 0;
  dfloat alphaL = 1;
  dfloat bubbleDiameter = 1e-3;
  dfloat gravity[3] = {0, 0, 0};
  dfloat alphaFloor = 1e-8;
  dfloat alphaDiffusivity = 0;
  dfloat dragMultiplier = 1;
  dfloat mixtureContinuityTolerance = 1e-9;
  int couplingIterations = 2;
  int pressureCorrectors = 2;
  bool projectionOnly = false;
  bool nativeAlphaScalar = false;
  scalar_t *alphaScalar = nullptr;

  occa::memory o_alphaG;
  occa::memory o_alphaL;
  occa::memory o_drag;
  occa::memory o_implicitLiquid;
  occa::memory o_implicitGas;
  occa::memory o_slipVelocity;
  occa::memory o_mixtureVelocity;
  occa::memory o_interphaseForce;
  occa::memory o_continuityResidual;
  occa::memory o_gasContinuityResidual;
  occa::memory o_liquidContinuityResidual;
  occa::memory o_pressureResponseLiquid;
  occa::memory o_pressureResponseGas;

private:
  const std::unique_ptr<geomSolver_t> &geom;
  occa::memory o_pressureCoeff;
  linearSolver *pressureCorrectionSolver = nullptr;
  elliptic *alphaSolver = nullptr;
  occa::memory o_alphaDiffusionCoeff;
  occa::memory o_alphaTransportCoeff;
  occa::memory o_mixtureFlux;
  occa::memory o_baseExtLiquid;
  occa::memory o_baseExtGas;
  occa::memory o_liquidVelocityTimePrevious;
  occa::memory o_gasVelocityTimePrevious;
  occa::memory o_divergenceLiquid;
  occa::memory o_divergenceGas;
  occa::memory o_alphaGPrevious;
  occa::memory o_alphaGRaw;
  occa::memory o_boundCapacity;
  occa::memory o_alphaGradient;
  occa::memory o_phaseFluxLiquid;
  occa::memory o_phaseFluxGas;
  occa::memory o_fluxDivergenceEToB;
  occa::memory o_divergencePhaseFluxLiquid;
  occa::memory o_divergencePhaseFluxGas;

  // Read-only snapshots used only for reporting nonlinear coupling progress.
  // They do not participate in any solver update.
  occa::memory o_alphaGCouplingPrevious;
  occa::memory o_liquidVelocityCouplingPrevious;
  occa::memory o_gasVelocityCouplingPrevious;
  occa::memory o_pressureCouplingPrevious;
  dfloat alphaCouplingRelativeL2 = 0;
  int currentCouplingIteration = -1;

  void phasePressureFlux(fluidSolver_t *phase, occa::memory o_flux);
  void applyHomogeneousVelocityMask(fluidSolver_t *phase,
                                    occa::memory o_velocityCorrection);
  void pressureCorrectionFlux(const occa::memory &o_phi,
                              occa::memory o_deltaLiquid,
                              occa::memory o_deltaGas,
                              occa::memory o_deltaMixture);
  void pressureCorrectionOperator(const occa::memory &o_phi,
                                  occa::memory o_Aphi);
  void updatePhaseFluxes();
  void weakDivergence(const occa::memory &o_velocity, occa::memory o_divergence);
  void updatePressureResponse(bool scaleForMomentumPredictor);
  void reportCouplingIteration(int couplingIteration);
};

void registerTwoFluidKernels();
