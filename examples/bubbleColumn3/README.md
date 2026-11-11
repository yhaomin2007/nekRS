# bubbleColumn3: bounded native pressure/phase corrections

This example advances ALPHA, UGX, UGY, UGZ, ULX, ULY, ULZ with native scalar
BDF/EXT/elliptic machinery. Native velocity storage contains only the reconstructed
volume velocity `UM=alpha*UG+(1-alpha)*UL`. Native UM momentum and its default
pressure solver stay disabled. No NekRS core files are changed.

## Native pressure/velocity audit

| Operation | Native fluid solver | Method3 |
|---|---|---|
| Pressure linear solver | `elliptic::solve` | Native `phasePressure` Poisson solve per outer correction; Schur comparison is selectable |
| Poisson preconditioner | native PGMRES/pMG/ASM/BoomerAMG | Same native machinery, configured by `[ELLIPTIC phasePressure]` |
| Pressure RHS/coefficient | native single-fluid force/stress construction, inverse density | Predictor weak divergence/boundary flux. Local inertia/viscosity/drag/VM mobility betaM/tau |
| Pressure gradient for velocity | `core-wGradientVolumeHex3D` | Identical registered kernel, no nodal averaging/mass inversion |
| Pressure force placement | Weak load added to momentum RHS before Helmholtz | Same placement, divided by each constant phase density since scalar transportCoeff=1 |
| Final velocity solve | component native elliptic Helmholtz (or vector stress option) | Six component native scalar elliptic Helmholtz solves with coupled drag and VM |
| Boundary lifting | native velocity elliptic Dirichlet masks; natural traction | Native scalar elliptic masks; supplied p=0 outlet has zero pressure traction |

`pressureSolveMode=outer` is the default experimental mode. It performs at
most three native Poisson corrections, each followed by full coupled phase
Helmholtz solves. No phase solve is called by the pressure Krylov operator.
The final post-boundary continuity residual determines whether the requested
target was achieved. A fixed correction budget can finish above that target.
The supplied PAR uses `pressureOuterRequireConvergence=0`, reports
`converged=no budgetExhausted=yes`, and continues for assessment. Set it to 1
to reject such steps. This mode needs GPU validation and does not establish
that three corrections eliminate velocity stripes.

`pressureSolveMode=consistent` selects the full phase-response Schur comparison.
`pressureSolveMode=native` selects the one-Poisson comparison.

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

VM uses `K_VM = C_VM*rhoLiquid*alpha` per mixture volume, with the configurable
default `C_VM=0.5`. Define `v_g=C_VM*rhoLiquid/rhoGas` and
`v_l=C_VM*alpha/(1-alpha)`. The combined partner coefficients are
`lambda_k=lambda_drag,k+v_k/tau`. VM adds
`v_g*(materialBase_g-materialBase_l)` to the gas base and the opposite
phase-normalized load to the liquid base. Material bases contain BDF history
and EXT material convection, including common gravity which cancels in their
difference. They have separate convection histories so that numerical
regularization forces are excluded from VM acceleration.

The local pressure block has diagonal entries
`a_g+lambda_g, a_l+lambda_l` and off-diagonal entries
`-lambda_g, -lambda_l`, where
`a_k=1/tau+average_c(diag(K_nu,k,c)/assembledMass)`.
The viscosity diagonals come from the native elliptic stiffness-diagonal
kernel, assembled at shared nodes. The three-component average supplies an
isotropic pressure approximation. Invert this block against
`[1/rhoGas,1/rhoLiquid]` and volume-weight its two responses to obtain betaM.
The full spatial viscosity remains in every phase Helmholtz solve.

The pressure constraint concerns the volume velocity. The default NekRS
velocity field holds that volume velocity. A density-weighted mixture velocity
is a different derived field and is not the constrained velocity here.

This matches the native PRESSURE-TO-VELOCITY mechanism for the supported
constant-density component-Laplacian formulation and supplied boundaries. It
is not the entire native single-fluid Navier-Stokes algorithm: the pressure
RHS/mobility and interphase drag are two-phase constructions. Phase-specific
stress-divergence/variable-viscosity terms, density splitting, nonzero outlet
pressure traction, slip and vector stress solves are not implemented here.
Native fluid's `grad(mu*divTarget)/3` is zero for its constant-viscosity,
zero-divergence target baseline; this example does not claim individual phase
velocities are divergence-free. It uses the stated scalar Laplacian phase model.

## Outer timestep

1. Assemble phase BDF/EXT bases and separate VM material histories once.
2. Advance alpha and freeze drag/VM coefficients at its updated value.
3. Solve the coupled phases using the previous pressure, with a pressure-free
   bootstrap on the first step.
4. Build local mobility and solve the native Poisson equation for deltaP.
5. Accumulate `p += pressureOuterRelaxation*deltaP`, solve both phase momenta
   with that pressure, restore boundaries, and measure the weak continuity.
6. Stop on the continuity target or the correction budget.

## Schur comparison timestep

1. Evaluate conservative alpha transport and material phase convection with
   GL cubature; prepare six BDF/EXT base RHSs before one native history shift.
2. Advance/clip alpha, restore its inlet, freeze paired drag coefficients using
   current alpha and lagged slip.
3. Solve gas/liquid Helmholtz predictors with fixed RHSs and coupled partner
   loads. The default incremental form includes the previous pressure after
   the first step. Reconstruct the UM predictor.
4. Form the assembled predictor weak continuity residual `r*`, including the
   inlet/wall flux. Pressure test rows at the fixed-pressure outlet are removed.
5. For each Schur direction, solve both coupled phase Helmholtz equations
   with only the native weak pressure load and homogeneous velocity boundary
   data. Normalize the pressure direction to unit weighted norm before the
   solves and rescale its weak continuity response afterwards. This computes
   `S = -D_m H^-1 G` directly, without subtracting two full-flow residuals.
6. Use flexible GCR with the native local-mobility Poisson preconditioner.
   Evaluate the updated total pressure with the physical phase RHSs and
   boundary conditions. Refine against that measured residual for up to three
   passes within the configured iteration budget. Commit the pressure and
   phase velocities together once the post-BC continuity target is reached.

The GCR residual is assembled in pressure DOF order with inverse-multiplicity
weights. For native preconditioning it is divided among shared element copies
because native elliptic assembles its input. Operator and RHS use the same
`tau=dt/g0` normalization. `betaM/tau` is the Poisson coefficient, where betaM
is the volume-weighted mobility from the local coupled inertia/drag inverse.

One Schur application costs a full pair of coupled phase solves. GCR vectors
reside on the device, with batched orthogonalization reductions. The
pressure equation uses the full numerical phase response up to the accuracy
of those solves. Equal-order pressure and fixed velocity boundaries can leave
null modes. If the target continuity residual is unattainable, the case reports
failure instead of accepting a pressure residual from the approximate Poisson
operator. This weak constraint does not enforce pointwise zero divergence.

## Supported boundaries and model

IDs: 1 inlet, 2 pressure outlet, 3 no-slip wall. Alpha has prescribed inlet and
zero normal gradient elsewhere. All six phase components have prescribed inlet,
zero normal gradient outlet and no-slip walls. Pressure has homogeneous outlet
Dirichlet and predictor normal-flux conditions at inlet/walls. Since outlet
pressure is zero, the native natural pressure traction `-p_out*n` vanishes,
matching the scalar zero-Neumann outlet. Other boundary maps are rejected.

Densities are constant and drag is Schiller-Naumann. The supplied test case
enables virtual mass with `virtualMassCoefficient=0.5`. Set
`virtualMassEnabled=0` for the drag-only comparison.
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
pressureSolveMode = outer
pressureOuterCorrections = 3
pressureOuterRelaxation = 1.0
pressureOuterRequireConvergence = 0
virtualMassEnabled = 1
virtualMassCoefficient = 0.5
pressureWarmStart = 1
pressureSchurMaxIterations = 100
pressureSchurRestart = 30
pressureSchurTolerance = 1e-6
pressureSchurAbsoluteTolerance = 1e-12
```

The outer iteration accepts `norm(r_final) <= max(absoluteTolerance,
relativeTolerance*norm(r_predictor))`, with both norms in physical weak-RHS
units and after pressure-row masking. The absolute floor allows small predictor
residuals to be accepted at the configured absolute accuracy. The absolute mode
uses the previous total pressure as its outer initial guess when available and
`pressureWarmStart=1`. The native preconditioner uses a loose relative tolerance
(`1e-2+relative` with native volume scaling) and the existing multigrid settings.
The native reference mode uses the original `[ELLIPTIC phasePressure]` tolerance.
Consistent mode uses `1e-11+relative` for the phase Helmholtz solves during
pressure coupling and restores their configured tolerances afterwards. The
physical drag-coupling target is capped at `1e-12`. The homogeneous response
uses relative drag convergence. The continuity acceptance tolerance is
independent of these inner tolerances.

Native UM uses `solver=none`. The case creates a separate native `phasePressure`
instance. Setup0 sets internal Poisson/coefficient-field options before kernel
registration, since the installed v26 parser rejects the elliptic `poisson` key.

## Pressure increment comparison

The default `pressureIncremental=1` uses the previous final pressure in the phase
predictors after the first solve. Both modes then compute a pressure increment
with zero outlet increment, and the final phases use `pNew=pOld+deltaP`.
Consistent mode initializes the increment to zero and reuses the predictor
for the first continuity evaluation. Native mode can reuse the
previous increment as its Poisson guess with `pressureWarmStart=1`.
Set `pressureIncremental=0` to compare with the absolute-pressure form.

In native reference mode the approximate pressure operator K differs from the
actual Helmholtz response S. The operator defect acts on the full pressure in
absolute mode and on deltaP in increment mode. Consistent mode uses S for both
forms and checks the residual against the final computed phase velocities.

## Logs and continuity

`method3OuterPressure` reports the native Poisson iteration count and residual.
`method3OuterContinuity` reports the actual post-BC weak residual on monitored
steps. `method3Outer` reports correction count, final continuity target status,
elapsed correction time, coupling iterations and force cancellation.
`couplingForceBalance` is the maximum relative sum of the two computed
mixture-volume drag/VM loads, including material history loads.
`method3Momentum` recomputes each phase residual with native Ax, assembled
RHS and final partner velocity, masking Dirichlet rows. It is printed on
monitored steps. The final summary identifies whether this residual was
measured. These diagnostics assess discrete equations and do not replace a
time-step/grid sensitivity study.


`method3Schur` reports outer iterations, phase evaluations, native Poisson
preconditioner calls, physical weak residual before/after the update, target,
and convergence. The final residual is measured after boundary restoration.
`passes` counts outer correction/refinement passes. `linearResidual` reports
the last homogeneous Schur residual in the same units as the measured weak
residual. `pressureSchurRestart` sets how many GCR directions are retained
before rebuilding the basis. Each direction uses two device pressure vectors.
The solver batches the inner products in each orthogonalization pass. The
`phaseTime` and `preconditionerTime` fields show elapsed host time inside the
phase evaluations and native Poisson preconditioner calls, respectively.
`method3SchurRestart` reports the initial and recomputed weak linear residuals
at each restart and at the final iteration of each pass. It is printed on
monitored steps and on failure. A small linear residual with a larger physical
residual indicates that the computed phase response and full-flow solve differ
at that scale.
A large linear residual can also reflect pressure-space compatibility or
exhausted iterations.
Use `mode=consistent` output to compare strong divergence and velocity fields.

`method3Pressure mode=native` reports the actual native solve each step,
including initial/final residual, target, convergence and time.
`method3PressureCost velocityPressure=weakRhsHelmholtz nativeSolves=1 innerCalls=0`
identifies the native reference path. The cost includes final phase solves, not only Poisson.
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
`tests/incremental_pressure.py` exercises production GCR with assembled two-element
coupled Helmholtz responses in absolute and incremental form. `tests/projection3d.py`
checks the full Helmholtz Schur operator on a compatible wall-masked space and
rejection of an incompatible RHS in the historical masked projection model. The full target column
GPU/MPI run, long-time stability, velocity-band behavior and physical accuracy
remain unverified. No exact UM divergence-free claim is made.
