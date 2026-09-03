#if !defined(nekrs_eulerEuler_hpp_)
#define nekrs_eulerEuler_hpp_

#include "nrs.hpp"

// Two-fluid (Eulerian-Eulerian) dispersed-phase plugin.
//
// Scope (see src/app/nrs/plugins/EULER_EULER_DESIGN.md for the full design note):
//   - exactly 2 phases: nekRS's native fluid solver is phase 1 (continuous), this
//     plugin adds phase 2 (dispersed) as a plain velocity field it owns directly.
//   - isothermal, incompressible, constant particle/bubble diameter.
//   - interfacial coupling is drag-only (no lift, virtual mass, wall lubrication).
//   - the phase fraction alpha (dispersed-phase volume fraction) is transported by
//     nekRS's own scalar_t solver (registered via `scalars = ...` in .par) using its
//     native CG discretization plus a user-supplied numerical diffusivity, instead of
//     a bound-preserving (FCT/limited) scheme -- this is the accepted Phase 0
//     simplification. alpha is clipped to [alphaMin, alphaMax] every step.
//   - the phase-drag coupling is applied via an operator-split correction, i.e.
//     *after* nekRS's own fluid+scalar step has converged for the timestep, not
//     simultaneously with the pressure solve as in OpenFOAM's pEqn.H. This is a
//     known approximation; removing it requires core-level changes (see design doc).
namespace eulerEuler
{

enum struct DragModel { SchillerNaumann, WenYu, Ergun, Gidaspow };

struct config_t {
  // Name of the scalar (declared via `scalars = <NAME>` in .par) that holds the
  // dispersed-phase volume fraction alpha. Its [SCALAR <NAME>] diffusivity entry is
  // the artificial numerical diffusivity used to stabilize the CG transport (Phase 0).
  std::string alphaScalarName = "alpha";

  dfloat rhoContinuous = 1.0;
  dfloat rhoDispersed = 1.0;

  // kinematic viscosity of the continuous phase; must match [FLUID VELOCITY]
  // viscosity so the drag Reynolds number is computed consistently.
  dfloat nuContinuous = 1.0e-3;

  dfloat particleDiameter = 1.0e-3;
  dfloat gravity[3] = {0.0, 0.0, 0.0};

  DragModel dragModel = DragModel::SchillerNaumann;

  bool clipAlpha = true;
  dfloat alphaMin = 0.0;
  dfloat alphaMax = 0.999; // keep continuous-phase fraction bounded away from 0
};

// Registers this plugin's OKL kernels. Call unconditionally from UDF_LoadKernels,
// mirroring lowMach::buildKernel / RANSktau::buildKernel.
void buildKernel(occa::properties kernelInfo);

// One-time setup. Call from UDF_Setup, after nrs->scalar has been constructed (i.e.
// after nrsFinalizeSetup / the point where `scalars = ...` fields exist).
void setup(const config_t &cfg);

// Overwrites the dispersed-phase velocity initial condition (defaults to zero).
void setIC(const occa::memory &o_U2);

// Advances the interfacial coupling by one step. Call from UDF_ExecuteStep, i.e.
// after nekRS's own step() has completed for this timestep.
void step(double time, int tstep);

occa::memory o_U2();     // dispersed-phase velocity; layout matches nrs->fluid->o_U
occa::memory o_alphaD(); // = nrs->scalar->o_solution(alphaScalarName)

} // namespace eulerEuler

#endif
