# bubbleColumn3: independently transported gas and liquid velocities

This example advances **ALPHA, UGX, UGY, UGZ, ULX, ULY, ULZ** with NekRS's
native scalar BDF/EXT and elliptic machinery. `fluid->o_U` stores only the
reconstructed **volume average**

    um = alpha*ug + (1-alpha)*ul.

No mixture momentum equation is solved. All implementation changes are local
to this directory; `bubbleColumn` and `bubbleColumn2` are unchanged.

## Run

Use the same column `.re2` mesh as `bubbleColumn2`. The supplied `[MESH] file`
points to `../bubbleColumn2/bubbleColumn2.re2`; change that path if your existing
column mesh has another filename. The mesh is not committed in this repository.
Boundary IDs must be **1 inlet, 2 outlet, 3 no-slip wall**.

From this directory, using the installed v26 NekRS executable:

```sh
nekrs --setup bubbleColumn3 --build-only
nekrs --setup bubbleColumn3
```

Use the usual MPI/GPU launcher for production runs. Start fresh for the first
comparison. A Method2 checkpoint has only four scalars and is **not** a valid
Method3 restart; only seven-scalar Method3 checkpoints have the correct layout.
The default output/checkpoint contains `um`, the shared pressure, and all seven
scalars. Separate `ug`/`ul` vector files are also written at checkpoints.

## Timestep and equations

1. At source time, evaluate conservative `div(alpha*ug)` and material
   `ug.grad(ug)` / `ul.grad(ul)` on the existing GL cubature grid. Retain native
   EXT histories. Freeze the diagonal drag coefficients and partner-velocity
   loads for the timestep; form all six phase RHSs before native history lagging.
2. The host advances alpha once and clips it to the configured range. Inlet
   alpha is then restored. Native history arrays shift exactly once.
3. In `UDF_ExecuteStep`, solve the six pressure-free scalar Helmholtz predictors
   with native BDF, implicit scalar Laplacian diffusion, HPFRT if configured,
   gravity, and semi-implicit diagonal drag. Reconstruct `um*`.
4. Solve the user-owned variable-coefficient pressure Poisson equation.
5. Correct **both phases with this newly solved pressure**, restore their
   scalar Dirichlet boundaries, and reconstruct final `um`.

Writing `tau=dt/g0`, `lambda_k` for each normalized diagonal drag coefficient,
and `nu_k` for each scalar's diffusion coefficient, the phase predictor is

    (g0/dt + lambda_k - nu_k Laplacian) uk*
        = BDF_history/dt + EXT[-uk.grad(uk) + gravity + lambda_k*u_other].

Both phase equations use transportCoeff=1. The diagonal drag is lagged; the
explicit partner load follows native EXT. This is a segregated drag treatment,
not an exact coupled phase inversion. The pressure response used by the
projection is

    beta_g = tau/[rho_g*(1+tau*lambda_g)]
    beta_l = tau/[rho_l*(1+tau*lambda_l)]
    beta_m = alpha_new*beta_g + (1-alpha_new)*beta_l
    -div(beta_m grad(p_new)) = -div(um*)
    ug_new = ug* - beta_g grad(p_new)
    ul_new = ul* - beta_l grad(p_new).

`bubbleColumn3Pressure.hpp` adapts the weak-divergence assembly, pressure-Neumann
surface flux, and elliptic solve from `fluidSolver_t::solvePressure`. The six
scalar predictors replace the native mixture BDF/forcing/stress construction.
The pressure is an **absolute projection pressure**: predictors contain no
old-pressure load, so the correction uses `p_new`, not `p_new-p_old`.

This is a pressure/viscosity projection split. The local correction uses
inertia plus diagonal drag rather than the full inverse Helmholtz operator.
The continuous Poisson identity does not imply zero discrete strong divergence:
native CG stiffness, assembled nodal gradients, and boundary reimposition can
leave a splitting/discretization residual. The log reports actual pre/post
strong-divergence L2 norms along with pressure range, mobility range, iterations,
and pressure residual. Check those on the target mesh before judging stability.

## Boundaries and controls

- Both phase velocities: prescribed inlet, zero-gradient outlet, no-slip wall.
  Liquid inlet velocity defaults to zero; gas inlet velocity defaults to 0.3 m/s.
  Shared wall/inlet nodes use native scalar boundary precedence.
- Alpha: prescribed inlet, zero-gradient outlet and wall.
- Pressure: zero gauge at outlet; inlet/wall pressure fluxes come from the
  prescribed reconstructed mixture normal velocity. Pressure correction does
  not override phase Dirichlet values.
- `solver=none` in `[FLUID VELOCITY]` prevents construction of native velocity
  and pressure elliptic solvers. The user pressure solver has the independent
  `[ELLIPTIC phasePressure]` configuration and is deliberately **not** assigned
  to `fluid->ellipticSolverP`, which would trigger a second native pressure solve.
- `equation=stokes` selects **no native advection only**. Full phase convection
  remains active in user sources. This avoids native `um` advection and fragile
  cancellation of a second convection operator.
- The first baseline includes Schiller–Naumann drag and scalar Laplacian phase
  viscosity. Virtual mass is explicitly disabled and nonzero settings abort.
  Method2 drift-stress/mixture forces are not appropriate when both phase
  momenta are advanced directly. Lift, turbulent dispersion, wall lubrication,
  full phase-fraction-weighted symmetric stresses, and phase change are not
  implemented in this initial example.
- `alphaMaximum<1` is required because the normalized liquid drag contains
  `alpha/(1-alpha)`. Alpha diffusion, clipping, and regularization are configurable
  numerical stabilizations; clipping may change integrated gas volume.
- Fixed mesh, seven native elliptic scalars, a single native stage, and no
  subcycling/NekNek/CVODE are required and checked at setup.

## Validation

```sh
python3 tests/run.py
```

This compiles and executes the production pointwise OKL equations as serial
C++ and checks reconstruction, equal/opposite drag force coefficients, finite
zero-slip behavior, pressure mobility/correction consistency, the single-liquid
hydrostatic limit, retained convection, and pressure boundary-flux sign.
It does not replace a coupled NekRS run on the column mesh.

For the optional real OCCA cubature test (using your installed OCCA paths):

```sh
c++ -std=c++17 -I"$NEKRS_HOME/occa/include" tests/transport.cpp \
  -L"$NEKRS_HOME/occa/lib" -Wl,-rpath,"$NEKRS_HOME/occa/lib" -locca -o /tmp/method3-transport
/tmp/method3-transport bubbleColumn3Transport.okl
```

During development, the host UDF passed a C++17 syntax check against this
branch's headers; both OKL files passed OCCA CUDA translation. The real OCCA
Serial cubature test reproduced `div(alpha*u)` and all three material convection
components with maximum absolute error `1.60e-14` on its manufactured polynomial
fields. The full NekRS build/run has not been performed here: no installed
NekRS/Fortran toolchain or column mesh is available in this workspace.
