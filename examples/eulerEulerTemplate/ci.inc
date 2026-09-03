// Smoke-test CI harness for the eulerEuler plugin template.
//
// This is intentionally NOT a physics-accuracy regression test (there is no
// verified reference solution for this new plugin yet -- see
// ../../src/app/nrs/plugins/EULER_EULER_DESIGN.md limitation #4). It only
// checks that a short run (a) completes without the solver diverging and
// (b) keeps the plugin's basic invariants intact: both phase velocities stay
// finite and the phase fraction alpha stays within its clipped bounds.
// Tighten/replace this once a real reference case exists.

#include <algorithm>
#include <cmath>

static int ciMode = 0;

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("POLYNOMIAL DEGREE", std::string("7"));
  options.setArgs("SOLUTION OUTPUT INTERVAL", "0");
  options.setArgs("NUMBER TIMESTEPS", std::string("20"));
  options.setArgs("DT", std::string("1e-03"));
  options.setArgs("BDF ORDER", "2");
  options.setArgs("FLUID VELOCITY SOLVER TOLERANCE", std::string("1e-12"));
  options.setArgs("FLUID PRESSURE SOLVER TOLERANCE", std::string("1e-08"));
  options.setArgs("SCALAR00 SOLVER TOLERANCE", std::string("1e-12"));
}

static bool fieldIsFinite(occa::memory o_fld, dlong N)
{
  std::vector<dfloat> hostFld(N);
  o_fld.copyTo(hostFld.data(), N);

  for (auto v : hostFld) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

void ciTestErrors(nrs_t *nrs, double time, int tstep)
{
  if (!nrs->lastStep) {
    return;
  }

  const bool uFinite = fieldIsFinite(nrs->fluid->o_U, 3 * nrs->fieldOffset);
  CiEvalTest("continuous-phase velocity finite", uFinite);

  const bool u2Finite = fieldIsFinite(eulerEuler::o_U2(), 3 * nrs->fieldOffset);
  CiEvalTest("dispersed-phase velocity finite", u2Finite);

  occa::memory o_alpha = eulerEuler::o_alphaD();
  const auto alphaOffset = nrs->scalar->fieldOffset();
  const bool alphaFinite = fieldIsFinite(o_alpha, alphaOffset);
  CiEvalTest("alpha finite", alphaFinite);

  auto mesh = nrs->fluid->mesh;
  std::vector<dfloat> alphaHost(mesh->Nlocal);
  o_alpha.copyTo(alphaHost.data(), mesh->Nlocal);

  dfloat alphaMin = *std::min_element(alphaHost.begin(), alphaHost.end());
  dfloat alphaMax = *std::max_element(alphaHost.begin(), alphaHost.end());
  MPI_Allreduce(MPI_IN_PLACE, &alphaMin, 1, MPI_DFLOAT, MPI_MIN, platform->comm.mpiComm());
  MPI_Allreduce(MPI_IN_PLACE, &alphaMax, 1, MPI_DFLOAT, MPI_MAX, platform->comm.mpiComm());

  const dfloat tol = 1e-6;
  CiEvalTest("alpha bounded in [0,1] (min=" + to_string_f(alphaMin) + ", max=" + to_string_f(alphaMax) + ")",
             alphaMin > -tol && alphaMax < 1.0 + tol);
}
