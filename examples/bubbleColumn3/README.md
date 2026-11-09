# bubbleColumn3: native pressure and phase Helmholtz velocity solves

This example advances ALPHA, UGX, UGY, UGZ, ULX, ULY, ULZ with native scalar
BDF/EXT/elliptic machinery. Native velocity storage contains only the reconstructed
volume velocity `UM=alpha*UG+(1-alpha)*UL`. Native UM momentum and its default
pressure solver stay disabled. No NekRS core files are changed.

## Native pressure/velocity audit

| Operation | Native fluid solver | Method3 |
|---|---|---|
| Pressure linear solver | `elliptic::solve` | Same class and solve method, separate `phasePressure` instance |
| Pressure linear algebra | native PGMRES/pMG/ASM/BoomerAMG | Native configuration from `[ELLIPTIC phasePressure]` |
| Pressure RHS/coefficient | native single-fluid force/stress construction, inverse density | Two-phase predictor weak divergence/boundary flux, coupled local mobility betaM/tau |
| Pressure gradient for velocity | `core-wGradientVolumeHex3D` | Identical registered kernel, no nodal averaging/mass inversion |
| Pressure force placement | Weak load added to momentum RHS before Helmholtz | Same placement, divided by each constant phase density since scalar transportCoeff=1 |
| Final velocity solve | component native elliptic Helmholtz (or vector stress option) | Six component native scalar elliptic Helmholtz solves with coupled drag |
| Boundary lifting | native velocity elliptic Dirichlet masks; natural traction | Native scalar elliptic masks; supplied p=0 outlet has zero pressure traction |

The previous `u_k=u_k*-beta_k*strongGrad(p)` correction did NOT use the same
pressure response as native `solveVelocity`. It has been removed from the
runtime path. Custom host GCR, stabilized composed pressure, and their knobs are
also removed from the runtime. `pressureSolveMode=native` is required;
`pressureSolveMode=composed` is rejected. Historical pointwise/cubature tests
and the standalone Krylov header remain for regression/history only; the UDF
neither includes nor executes that Krylov solver.

For each phase k, the final normalized equation is

```
(g0/dt + lambda_k - nu_k*Laplacian) u_k
 = BDF_EXT_base_k + lambda_k*u_partner + weakPressureLoad(p)/rho_k
```

Here weakPressureLoad denotes the element-local native integral `grad(test)*p`,
including its native sign convention. It already contains quadrature/geometric
weights: do NOT multiply it by mass or divide by mass again. The native
elliptic solve assembles it and applies boundary lifting. The pressure is the
latest solved absolute pressure, fixed during the final coupled drag iterations.
The pressure-free base is saved/restored; no pressure term enters EXT history
and no BDF/EXT history is advanced twice.

This matches the native PRESSURE-TO-VELOCITY mechanism for the supported
constant-density component-Laplacian formulation and supplied boundaries. It
is not the entire native single-fluid Navier-Stokes algorithm: the pressure
RHS/mobility and interphase drag are two-phase constructions. Phase-specific
stress-divergence/variable-viscosity terms, density splitting, nonzero outlet
pressure traction, slip and vector stress solves are not implemented here.
Native fluid's `grad(mu*divTarget)/3` is zero for its constant-viscosity,
zero-divergence target baseline; this example does not claim individual phase
velocities are divergence-free. It uses the stated scalar Laplacian phase model.

## Timestep

1. Evaluate conservative alpha transport and material phase convection with
   GL cubature; prepare six BDF/EXT base RHSs before one native history shift.
2. Advance/clip alpha, restore its inlet, freeze paired drag coefficients using
   current alpha and lagged slip.
3. Solve pressure-free gas/liquid Helmholtz predictors with fixed RHSs and
   coupled partner loads; reconstruct UM predictor.
4. Solve one native variable-coefficient pressure Poisson equation, normalized
   on BOTH sides by tau=dt/g0. betaG/L come from the coupled local inertia/drag
   inverse; betaM=alpha*betaG+(1-alpha)*betaL. betaM/tau is the pressure diffusion
   coefficient. No custom outer pressure iteration is called.
5. Evaluate the latest pressure with the native WEAK gradient kernel, add its
   load/rho_k to each phase base RHS, and solve both final phase Helmholtz
   equations with the SAME frozen coefficients and histories. Reconstruct UM.
6. Restore scalar boundary values and report diagnostics/output.

There are TWO sets of phase solves (predictors and pressure-loaded final solves),
with one pressure solve. This costs more than the earlier algebraic correction,
but includes the full phase Helmholtz response to pressure. Pressure construction
still uses a local mobility approximation, not the inverse spatial Helmholtz
operator. Small pressure residual does NOT establish zero final UM divergence.

## Supported boundaries and model

IDs: 1 inlet, 2 pressure outlet, 3 no-slip wall. Alpha has prescribed inlet and
zero normal gradient elsewhere. All six phase components have prescribed inlet,
zero normal gradient outlet and no-slip walls. Pressure has homogeneous outlet
Dirichlet and predictor normal-flux conditions at inlet/walls. Since outlet
pressure is zero, the native natural pressure traction `-p_out*n` vanishes,
matching the scalar zero-Neumann outlet. Other boundary maps are rejected.

Densities are constant, drag is Schiller-Naumann, and virtual mass is disabled.
The seven scalar transport coefficients must all be 1. Alpha diffusion/clipping
and phase diffusion/HPFRT are controlled by the supplied PAR file. Native UM
advection is disabled with equation=stokes; phase convection remains in the
user explicit sources. No mesh adaptation/moving-mesh equivalence is claimed.

## Run and controls

Use the existing column mesh; default path is
`../bubbleColumn2/bubbleColumn2.re2`. The mesh is not committed here.

```
nekrs --setup bubbleColumn3 --build-only
nekrs --setup bubbleColumn3
```

Use your MPI/GPU launcher. Method2 checkpoints lack the required seven-scalar
layout. The UDF writes separate gas/liquid vectors at checkpoint steps; default
output contains UM, pressure and seven scalar fields.

```
[CASEDATA]
pressureSolveMode = native
pressureWarmStart = 1
```

Pressure tolerance/PGMRES/multigrid settings come from `[ELLIPTIC phasePressure]`.
Native UM remains `solver=none`; the UDF never installs its own pressure instance
as `fluid->ellipticSolverP`. Old pressureProjection* and
pressurePreconditionerTolerance CASEDATA controls are obsolete and ignored.
The v26 parser rejects an elliptic `poisson` PAR key, so Setup0 sets the internal
Poisson/coefficient-field options before native kernel registration.

## Pressure increment comparison

Set `pressureIncremental=1` in CASEDATA to test a lagged-pressure predictor.
The supplied default is 0 for the absolute-pressure reference. The first step
of a run bootstraps with absolute pressure. Subsequent increment-mode steps
compute the two predictors with the previous final pressure, form the same
weak continuity RHS from that mixture, and solve one native Poisson equation
for deltaP with zero outlet deltaP. The final Helmholtz solves use
`pNew=pOld+deltaP`. Both phase solves use the same frozen base RHS and drag
coefficients. `pressureWarmStart=1` reuses the previous increment as the linear
solver guess after the first increment.

Let A denote the actual assembled weak continuity response of the coupled
phase Helmholtz solves to pressure, with the same tau normalization as K.
The absolute-pressure algorithm leaves the operator defect `(K-A)*p` in the
continuity residual. With the lagged-pressure predictor, it leaves
`(K-A)*deltaP`. This can reduce the defect from a large, slowly changing
hydrostatic pressure, while keeping one native pressure solve per step.
The operators still differ. This experiment does not establish exact
continuity, stable long-time behavior, or removal of the velocity bands.

Check `pressureForm=increment` after the first step. In this mode,
`nativeEquationResidual` and `correctionDefect` use the increment and the
actual change between the lagged-pressure predictor and final phase velocities.
Compare absolute/increment runs on the same mesh and physical times using
post-correction weak/strong divergence and the axial velocity profile.

`python3 tests/incremental_pressure.py` checks the residual identity with
assembled two-element SEM operators, two coupled phase Helmholtz inverses
and mixed inlet/outlet masks. Its manufactured pressure-lag case tests the
expected defect reduction. The target-column GPU comparison is pending.

## Logs and continuity

`method3Pressure mode=native` reports the actual native solve each step,
including initial/final residual, target, convergence and time.
`method3PressureCost velocityPressure=weakRhsHelmholtz nativeSolves=1 innerCalls=0`
identifies the new path. The cost includes final phase solves, not only Poisson.
`dragIterations`/`dragChange` on the final summary refer to pressure-loaded solves.

At monitorInterval, `weakL2(pre,corrected,BC)` uses one assembled weak operator
with fixed prescribed boundary flux and pressure-outlet rows excluded.
`nativeEquationResidual` measures the native pressure equation residual in
physical weak-RHS units. `correctionDefect` measures the difference between
native stiffness action and the ACTUAL velocity change from the final phase
Helmholtz solves. `residualBalanceError` verifies the vector identity

```
finalWeakResidual = nativePressureResidual + actualVelocityResponseDefect
```

Printed quantities are norms and are not generally additive. These algebraic
weak norms are not strong pointwise divergence or percentage mass imbalance.
`boundaryChange` isolates boundary restoration; old strong `divL2` remains a
separate diagnostic. The coefficient-explicit native `Ax()` API is used for
monitor actions, followed by one gather-add: `op()` is invalid after solve()
unbinds its internal coefficient pointers.

## Validation and limits

```
python3 tests/run.py
python3 tests/lifecycle.py
python3 tests/native_mode.py
python3 tests/native_action.py
```

Lifecycle tests compile production callbacks and pressure helpers with BDF2/
EXT2 and EXT3, checking pressure sign/density scaling, coupled phase response,
base restoration, no duplicate pressure load, frozen drag and one history shift.
They model uniform modes, not distributed SEM solves. The complete UDF passes
native-header C++17 syntax checking.

Real OCCA Serial weak-pressure-kernel test (adjust installed paths):

```
c++ -std=c++17 -I"$NEKRS_HOME/occa/include" tests/weak_pressure.cpp \
 -L"$NEKRS_HOME/occa/lib" -Wl,-rpath,"$NEKRS_HOME/occa/lib" -locca -o /tmp/weak-pressure
/tmp/weak-pressure /path/to/nekRS/src/core/kernels/wGradientVolumeHex3D.okl
```

This executes the actual native weak gradient, verifies all nodal loads and its
interior negative-gradient sign; maximum absolute error was 1.78e-15. OCCA CUDA
translation is also checked with `OCCA_BIN=... python3 tests/cuda_translation.py`.
Historical projection.py/projection3d.py test old projection algebra and do not
validate the new final-phase Helmholtz spatial response. The full target column
GPU/MPI run, long-time stability, velocity-band behavior and physical accuracy
remain unverified. No exact UM divergence-free claim is made.
