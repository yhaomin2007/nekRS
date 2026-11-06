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
   EXT histories for convection/regularization, and form all six base phase
   RHSs before native history lagging. Drag partner loads are excluded from EXT.
2. The host advances alpha once and clips it to the configured range. Inlet
   alpha is then restored. Native history arrays shift exactly once. Freeze
   drag coefficients using this updated alpha and the lagged phase slip.
3. In `UDF_ExecuteStep`, solve the six pressure-free scalar Helmholtz predictors
   with native BDF, implicit scalar Laplacian diffusion, HPFRT if configured,
   gravity, and coupled implicit drag. Iterate the two phase partner loads to
   `dragCouplingTolerance` with unchanged BDF/EXT histories and coefficients;
   abort if `dragCouplingMaxIterations` is reached. Reconstruct `um*`. These are
   drag iterations only: pressure is still solved once per timestep.
4. Solve the user-owned variable-coefficient pressure Poisson equation.
5. Correct **both phases with this newly solved pressure**, restore their
   scalar Dirichlet boundaries, and reconstruct final `um`.

Writing `tau=dt/g0`, `lambda_k` for each normalized diagonal drag coefficient,
and `nu_k` for each scalar's diffusion coefficient, the phase predictor is

    (g0/dt + lambda_k - nu_k Laplacian) uk*
        = BDF_history/dt + EXT[-uk.grad(uk) + gravity] + lambda_k*u_other*.

Both phase equations use transportCoeff=1. Drag coefficients are lagged in
slip but use current alpha. The partner load uses the latest predictor iterate
and the same coefficient as its implicit diagonal; it is never extrapolated
independently. The converged predictor solves the coupled frozen-coefficient
drag/Helmholtz equations. The pressure response uses the inverse local 2x2
inertia/drag block:

    tg = tau*lambda_g; tl = tau*lambda_l; det = 1+tg+tl
    beta_g = tau*((1+tl)/rho_g + tg/rho_l)/det
    beta_l = tau*(tl/rho_g + (1+tg)/rho_l)/det
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
the full two-phase inertia/drag block rather than the full inverse spatial
Helmholtz operator. The continuous Poisson identity does not imply zero discrete strong divergence:
native CG stiffness, assembled nodal gradients, and boundary reimposition can
leave a splitting/discretization residual. The log reports actual pre/post
strong-divergence L2 norms along with pressure range, mobility range, iterations,
pressure residual, drag iterations/change, and separate gas/liquid CFL values.
Check those on the target mesh before judging stability. A small elliptic
residual alone does not establish a small continuity residual.

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
  `[ELLIPTIC phasePressure]` configuration. The Poisson type is set internally
  in `UDF_Setup0`, before kernel registration, because some installed v26
  parsers (including sha `96b3cf9`) reject `poisson` as a PAR key. It is
  deliberately **not** assigned
  to `fluid->ellipticSolverP`, which would trigger a second native pressure solve.
  The user pressure solver is built lazily at the first completed predictor,
  after native mesh/scalar initialization and after the actual pressure
  coefficient is available. Initial scalar boundary application
  is similarly deferred until native scalar elliptic masks exist.
- `equation=stokes` selects **no native advection only**. Full phase convection
  remains active in user sources. This avoids native `um` advection and fragile
  cancellation of a second convection operator. Core advection kernels are
  nevertheless registered unconditionally by v26; `bubbleColumn3Cubature.hpp`
  supplies missing `c_I` constants for requested polynomial/cubature pairs.
- The first baseline includes Schiller–Naumann drag and scalar Laplacian phase
  viscosity. Virtual mass is explicitly disabled and nonzero settings abort.
  Method2 drift-stress/mixture forces are not appropriate when both phase
  momenta are advanced directly. Lift, turbulent dispersion, wall lubrication,
  full phase-fraction-weighted symmetric stresses, and phase change are not
  implemented in this initial example.
- `alphaMaximum<1` is required because the normalized liquid drag contains
  `alpha/(1-alpha)`. Alpha diffusion, clipping, and regularization are configurable
  numerical stabilizations; clipping may change integrated gas volume.
- Fixed mesh, fixed dt, seven native elliptic scalars on the same mesh, a single
  native stage, and no subcycling/NekNek/CVODE are required and checked. Native
  adaptive dt uses UM-based CFL and could miss phase slip, so it is rejected.
- The supplied scalar inlet/outlet/wall maps are checked against the pressure
  boundary assumptions. Both phase diffusion coefficients remain independent
  `.par` settings; adjust them when changing the desired kinematic viscosities.
- The pressure solver uses p-multigrid, a degree-3 Chebyshev/ASM smoother,
  and a CPU BoomerAMG coarse solve, following the native pressure strategy.
  A standalone Jacobi preconditioner is insufficient for the demonstrated
  column run. BoomerAMG requires the normal NekRS HYPRE dependency.
  The outer Krylov basis stays at 15 vectors to limit memory use.
- Both pressure-operator coefficient and RHS are divided by `tau=dt/g0` before
  solving; this is an equivalent equation, not a change to pressure or phase
  correction. The hierarchy is constructed using the actual first-step
  normalized coefficient. `beta_g/l/m` retain their physical timestep factors.
- Every phase linear solve and the pressure linear solve must meet its native
  configured tolerance; a nonfinite or unconverged result aborts. Pressure uses
  a relative tolerance of `1e-8` (with native volume normalization), so a
  small dimensional RHS cannot pass trivially. The convergence guard remains
  enabled; the new settings still need verification on the target column mesh.

The custom pressure solver prints a `method3Pressure` summary **every timestep**,
independently of `monitorInterval`: step/time, iteration count, initial/final
residuals, final-to-initial ratio, effective convergence target, convergence
status and solve wall time. It prints and flushes before the failure check, so
an unsuccessful solve is recorded as well. Residuals refer to the normalized
elliptic equation; they are not mixture continuity residuals.

## Validation

```sh
python3 tests/run.py
python3 tests/lifecycle.py
```

This compiles and executes the production pointwise OKL equations as serial
C++ and checks reconstruction, equal/opposite drag force coefficients, finite
zero-slip behavior, pressure mobility/correction consistency, the single-liquid
hydrostatic limit, retained convection, and pressure boundary-flux sign.
It also checks the coupled drag block against common-gravity and drag-only
momentum-conservation regressions. `lifecycle.py` compiles the production host
callbacks against a uniform-mode native-style adapter, exercising BDF2 with EXT2 and EXT3,
one-time history lagging, alpha-first/deferred-phase scheduling, coupled drag
convergence, and the missing-completion guard. This adapter does not emulate
distributed spatial solves. These tests do not replace a coupled NekRS run on
the column mesh.

To check the actual CUDA lowering (without needing nvcc):

```sh
OCCA_BIN="$NEKRS_HOME/occa/bin/occa" python3 tests/cuda_translation.py
```

The surface flux uses atomic addition of a negative value: CUDA has FP64
`atomicAdd` on the target architecture but no floating-point `atomicSub`.
The regression checks the generated CUDA, while the numerical boundary-flux
test checks the unchanged sign. CUDA translation is not an nvcc/GPU build.

For the optional real OCCA cubature test (using your installed OCCA paths):

```sh
c++ -std=c++17 -I"$NEKRS_HOME/occa/include" tests/transport.cpp \
  -L"$NEKRS_HOME/occa/lib" -Wl,-rpath,"$NEKRS_HOME/occa/lib" -locca -o /tmp/method3-transport
/tmp/method3-transport bubbleColumn3Transport.okl bubbleColumn3Equations.okl
```

The actual native PAR parser accepts the multigrid/Chebyshev-ASM/CPU-BoomerAMG
configuration and bounded Krylov settings; that check does not run the solver.

During development, the host UDF passed a C++17 syntax check against this
branch's headers; both OKL files passed OCCA CUDA translation. All phase,
drag and pressure kernels built with real OCCA Serial. The real OCCA cubature
test reproduced `div(alpha*u)` and all three material convection
components with maximum absolute error `1.60e-14` on its manufactured polynomial
fields. The full NekRS build/run has not been performed here: no installed
NekRS runtime or column mesh is available. A local CPU build attempt stopped
in the dependency's Fortran/C interoperability configuration, before a NekRS
executable could be produced; no full-solver validation is claimed.

## Stabilized composed pressure projection and continuity diagnostics

The unregularized composed operator introduced in 4830c8c is not a robust
pressure equation for the complete equal-order 3D pressure space. Masking
inlet/wall phase velocities creates pressure modes with zero allowed nodal
velocity response. A single order-3 hex has 48 free pressure unknowns but only
rank 28 in the masked gradient (20 null modes); at order 5 the counts are 180,
144, and 36. The earlier 1D regression did not reveal this structural issue.
A compatible RHS can still converge with a singular operator; null modes alone
do not prove the cause of a particular user's iteration failure.

Define `W u = assembled integral grad(q).u`, `G p = assembled mass-inverse
weighted gradient`, `F` as the homogeneous six-phase Dirichlet mask,
`A p = W betaM F G p / tau`, and `K` as the native variable-coefficient
stiffness with coefficient betaM/tau. The final pressure now solves

```
S p = b,   S = A + epsilon*(K-A) = (1-epsilon)*A + epsilon*K
```

The RHS `b` is the same predictor weak continuity residual with fixed prescribed
boundary flux, divided by tau. `pressureProjectionStabilization=0.1` (CASEDATA)
is explicit numerical pressure stabilization, required to be in `(0,1]`.
`epsilon=1` recovers the native stiffness equation. Smaller positive values
approach the nodal composed operator but weaken control of its near-null modes.
This is NOT exact zero weak divergence: after nodal phase correction,

```
weakResidualCorrected = tau * [ (b-S*p) + epsilon*(K-A)*p ]
```

Report both contributions; never interpret convergence of S alone as vanishing
continuity error. Stability and velocity-band behavior still require validation
on the target column mesh. The default epsilon is a correctness/development
choice, not a mesh-converged physical parameter.

The native Poisson solve supplies the initial pressure and preconditions
restarted flexible GCR for S. All changes remain in the example; native UM and
its default pressure solver remain disabled. GCR vectors/reductions are on the
host, while SEM derivatives and gather/scatters run on the device. This costs
host/device transfers, extra native solves, and roughly two restart-length sets
of host pressure vectors. Performance optimization must follow integration
validation.

CASEDATA also controls `pressureProjectionMaxIterations=200`,
`pressureProjectionRestart=15`, `pressureProjectionTolerance=1e-8`. The target
is tolerance times the larger of RHS norm and initial residual norm. The true
residual is recomputed before accepting convergence; a failed solve still
aborts, with diagnostics printed/flushed first.

Log records:

- `method3Pressure`: initial native stiffness solve.
- `method3Projection`: final stabilized solve, iterations, initial/true residual,
  RHS norm, target, relative residual, epsilon, stopping reason, convergence.
- `method3Continuity`: `weakL2(pre,corrected,BC)` uses one assembled weak
  operator and fixed boundary lifting. Pressure-outlet rows are excluded and
  duplicate copies weighted by inverse multiplicity. These are algebraic weak
  norms, not strong divergence or percentage mass imbalance.
- `boundaryChange`: residual change from restoring phase boundary values.
- `nativeVsComposedRel`: norm of `(K-A)p` relative to `A p`.
- `correctionIdentity`: norm of `(W UMpre-W UMcorrected)/tau-A p`.
- `stabilizationContinuity`: `tau*epsilon*norm((K-A)p)`; the continuity error
  explicitly permitted by pressure stabilization.
- `residualBalanceError`: discrepancy from the residual identity above; it
  should be roundoff relative to its terms.

The old strong `divL2` is separate. The `nativePrecoIters` and
`nativePrecoResidual` belong to the latest preconditioning solve, not S.

Spatial regressions (NumPy and C++17 required):

```sh
python3 tests/projection.py
python3 tests/projection3d.py
```

The first exercises a two-element 1D GLL correction. The second demonstrates
3D masked-pressure null modes. Its ideal inflow RHS is compatible; it then adds
an explicitly manufactured 1% null-space RHS component, verifies rejection by
the unregularized production GCR, and convergence of the stabilized production
GCR below 1e-8. It also checks the stabilization-continuity identity and the
native epsilon=1 limit. This is a spatial algebra regression, not a measured
failure reproduction on the user's mesh or a full 3D MPI/GPU run.

### Native operator coefficient lifetime

`elliptic::_solve()` clears its internal coefficient pointers before returning.
Calling `elliptic::op()` afterwards is invalid: the variable-coefficient CUDA
kernel can dereference a null coefficient pointer, with the asynchronous error
reported by a later `pressureHost()` copy. Both native-stiffness actions now use
`Ax(pressureDiffusion, o_NULL, q, result)`, which explicitly binds coefficients
for the call. Since Ax returns element-local contributions, the helper performs
one gather-add before callers mask outlet rows. The UDF body changes too, to
invalidate the host UDF cache when updating this included header.

`python3 tests/native_action.py` compiles the production helper against an
adapter that models coefficient unbinding after solve and element-local Ax
output, checking explicit coefficient binding and single assembly. It also
checks the lifetime assumption against the native source. This contract test
and the native-header syntax check do not replace a CUDA integration run.
