#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nrs.hpp"
#include "udf.hpp"

#include "eulerEuler.hpp"
#include "linAlg.hpp"

namespace
{

nrs_t *_nrs = nullptr;
eulerEuler::config_t _cfg;

occa::memory _o_U2;

occa::kernel clipAlphaKernel;
occa::kernel partialEliminationKernel;

bool buildKernelCalled = false;
bool setupCalled = false;

} // namespace

void eulerEuler::buildKernel(occa::properties kernelInfo)
{
  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/app/nrs/plugins/");
    const auto fileName = path + "eulerEuler.okl";
    const auto reqName = "eulerEuler::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    } else {
      buildKernelCalled = true;
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  clipAlphaKernel = buildKernel("clipAlpha");
  partialEliminationKernel = buildKernel("partialElimination");
}

void eulerEuler::setup(const config_t &cfg)
{
  static bool isInitialized = false;
  if (isInitialized) {
    return;
  }
  isInitialized = true;

  _nrs = dynamic_cast<nrs_t *>(platform->app);
  _cfg = cfg;

  nekrsCheck(_nrs->scalar == nullptr,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "eulerEuler::setup requires at least one scalar (the phase fraction) to be configured!");

  nekrsCheck(_nrs->scalar->nameToIndex.find(lowerCase(_cfg.alphaScalarName)) == _nrs->scalar->nameToIndex.end(),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             ("eulerEuler::setup requires scalar `" + _cfg.alphaScalarName +
              "` -- add it via `scalars = " + _cfg.alphaScalarName + "` in .par")
                 .c_str());

  _o_U2 = platform->device.malloc<dfloat>(3 * _nrs->fieldOffset);
  platform->linAlg->fill(3 * _nrs->fieldOffset, 0.0, _o_U2);

  setupCalled = true;
}

void eulerEuler::setIC(const occa::memory &o_U2in)
{
  o_U2in.copyTo(_o_U2);
}

occa::memory eulerEuler::o_U2()
{
  return _o_U2;
}

occa::memory eulerEuler::o_alphaD()
{
  return _nrs->scalar->o_solution(_cfg.alphaScalarName);
}

void eulerEuler::step(double time, int tstep)
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "eulerEuler::step called prior to eulerEuler::setup()/buildKernel()!");

  auto nrs = _nrs;
  auto mesh = nrs->fluid->mesh;
  occa::memory o_alpha = nrs->scalar->o_solution(_cfg.alphaScalarName);

  if (_cfg.clipAlpha) {
    clipAlphaKernel(mesh->Nlocal, _cfg.alphaMin, _cfg.alphaMax, o_alpha);
    // NOTE(Phase 0 simplification): a pointwise clip is not the same as the
    // MULES-style bounded transport OpenFOAM uses -- it can introduce a small,
    // local mass-conservation error in alpha near the clip bound. Acceptable for
    // the v0 framework per the accepted CG + artificial-diffusivity simplification;
    // revisit if drift becomes visible (e.g. track sum(alpha) over time).
  }

  partialEliminationKernel(mesh->Nlocal,
                            nrs->fieldOffset,
                            static_cast<int>(_cfg.dragModel),
                            _cfg.rhoContinuous,
                            _cfg.rhoDispersed,
                            _cfg.nuContinuous,
                            _cfg.particleDiameter,
                            _cfg.gravity[0],
                            _cfg.gravity[1],
                            _cfg.gravity[2],
                            nrs->g0,
                            nrs->dt[0],
                            o_alpha,
                            nrs->fluid->o_U,
                            _o_U2);

  // TODO(Phase 5 -- true pressure-drag coupling): the call above overwrites
  // nrs->fluid->o_U *after* nekRS's own pressure-velocity solve has already
  // completed for this timestep. Drag is therefore coupled implicitly to
  // velocity, but not to the pressure equation within the same step -- this is
  // an operator-split (fractional-step) approximation of OpenFOAM's
  // simultaneous pEqn.H + partialElimination() coupling. Closing this gap
  // requires blending alphaC/rhoC and alphaD/rhoD into nekRS's shared pressure
  // mobility field (the `o_lambda` built in fluidSolver_t::solvePressure(),
  // src/solver/fluid/fluidSolver.cpp), which is core-solver work -- see
  // EULER_EULER_DESIGN.md.
  //
  // TODO(Phase 6 -- dispersed-phase transport): only drag + buoyancy are
  // applied to the dispersed phase above; self-advection and diffusion of U2
  // (the twoPhaseEulerFoam UEqns.H analogue for phase 2) are not yet
  // implemented.
}
