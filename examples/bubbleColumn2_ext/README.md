# bubbleColumn2_ext: experimental within-timestep coupling

This is a standalone copy of bubbleColumn2 with native same-timestep Picard
iterations. Neither bubbleColumn2 nor NekRS core is changed. Physical equations,
QG variables, force switches, masks, filters, damping, boundaries, exports and
conservation diagnostics are inherited.

## Running

Use the same mesh as bubbleColumn2. Copy your mesh to
`bubbleColumn2_ext.re2` (and matching `bubbleColumn2_ext.ma2`, if used), then run
`nekrs bubbleColumn2_ext` with your usual launcher. The mesh binaries were not
available in this repository and are not included. Checkpoint/restart prefixes
must match the selected case; the restart line is commented out by default.

The new par file keeps tombo2 and dt=1e-5. It enables the original full drift
and mixture drag with stress correction and slip-limiter switches inherited
from the copied par file. Review these switches for the desired comparison.

```ini
[CASEDATA]
couplingMinIterations = 2
couplingMaxIterations = 20
couplingRelativeTolerance = 1e-5
couplingAbsoluteTolerance = 1e-8
couplingDivergenceTolerance = 1e-4
couplingRelaxation = 0.7
couplingRequireConvergence = 1.0
```

## Algorithm and temporal treatment

At timestep initialization, copy the previous physical-time solution history
before native lagSolution. Native runInnerStep shifts histories only at stage 1.
Every later stage repeats scalar, properties, mixture momentum and pressure
solves at the same time. The convergence callback rebuilds native RHS terms for
the next stage using the latest iterate, including conservative gas fluxes,
pressure sources, drag, stresses, native advection and regularization.

BDF coefficients and order are retained. Explicit-RHS weights are deliberately
set to [1,0,...]: these terms are evaluated at the current Picard iterate, rather
than EXT-extrapolated from older timesteps. On convergence this is a nonlinear
implicit BDF discretization, not the original BDF/EXT one-pass scheme. Forcing
assembly temporarily reads an immutable pre-step BDF history; the live iterate
and all native lagged solution slots are restored immediately afterwards. No
initStep, lagSolution or finishStep is called inside the callback. Private
advection history shifts are irrelevant because old RHS weights are zero.

Under-relax alpha, QG, mixture velocity and pressure after each trial solve.
Report componentwise maximum increments divided by atol+rtol*|current|, with
normalization by relaxation to prevent small relaxation hiding a raw increment.
All four groups must have scaled increments <=1 and assembled divergence RMS
must meet couplingDivergenceTolerance. This is an iterate-change criterion,
not a full nonlinear equation residual. Pressure's boundary reference fixes its
gauge in this case. A failure at the maximum iteration count aborts by default;
setting couplingRequireConvergence=0 explicitly permits an unconverged step
with a warning. Changing divergence tolerance can weaken continuity validation.

Trial clipping/masking accumulators are rolled back on rejected stages.
Virtual-mass previous velocities remain fixed during trial source evaluation.
Time-history diagnostics, phase-history updates and checkpoint output execute
only after the native timestep is accepted. This prevents repeated output and
repeated physical-time conservation accumulation.

## Scope and validation

Fixed mesh, no NekNek, and no advection subcycling are required. Native implicit
mixture drag remains the existing stabilized split, not a new coupled block
pressure operator. Picard convergence and larger-dt stability are not guaranteed.
The original gas masks and clipping can also affect convergence.

The serial mock test in tests/iteration_history.cpp compiles the actual iteration
header against constrained mock solver APIs. It checks immutable BDF history,
repeated RHS rebuilding, preservation of current and lagged iterate slots,
minimum iteration count and rejection on nonconvergence. Run:

```sh
c++ -std=c++17 -Wall -Wextra -Werror tests/iteration_history.cpp -o iteration_history
./iteration_history
```

The inherited OKL kernels were also serial-translated and syntax checked. No
full NekRS MPI/GPU build or simulation was available here. First compare a
short dt=1e-6 run with bubbleColumn2, then test dt=1e-5 while inspecting every
coupling record; a compilation pass does not establish physical correctness.

---

# bubbleColumn2_ext: uploaded QG working baseline

Primary fields are native volume velocity `uv`, pressure, `ALPHA`, and
`QGX/QGY/QGZ`, with `qg=alpha*ug` and `ul=(uv-qg)/(1-alpha)`.

The seven user-uploaded files are preserved verbatim in commit `4f79c46`.
The following repair retains the uploaded `.par`, `.usr`, and both boundary
files, all switches, HPFRT, outlet damping, gas masks, alpha/vector clipping,
phase-velocity exports, stability monitors and conservation CSV columns.
Gas pressure remains lagged; no new-pressure gas correction is added.

## Numerical repairs

- Cancel the exact native scalar-advection kernel for alpha and all QG
  components, including cubature and predicted contravariant velocity, rather
  than a pointwise approximation. Conservative flux divergences are unchanged.
  Nonzero scalar subcycling is rejected because its forcing ordering differs.
- Substitute `ul=(uv-qg)/beta` in gas drag: `lambda=Ki/(rhoGas*beta)`
  is implicit, while `alpha*lambda*uv` is explicit. The mixture reaction rate
  `alpha*(1-rhoGas/rhoLiquid)*lambda` is also put on its native Helmholtz
  diagonal. This remains segregated drag and retains the native `Ap` pressure
  mobility; it is not a block pressure/drag correction.
- Add the volume-mixture physical two-phase stress minus the native base stress
  as a lagged correction. Extra native viscosity from outlet damping remains.
  Existing explicit physical gas stress and scalar numerical diffusion remain.
- Initialize virtual-mass scratch before its first use, avoiding undefined
  data even when virtual mass is disabled.
- Use `alphaInitial` as the background plume fraction. With the supplied value
  zero, initialization is identical to the uploaded case, including initially
  stationary liquid and a localized, initially non-solenoidal volume velocity.

Clipping and masks retain their original nonconservative changes and accounting
in the CSV. No global redistribution limiter is introduced. Gas stress and
mixture stress corrections remain explicit and may restrict the timestep.

## Validation status

CPU-translated pointwise-kernel syntax/algebra and baseline-preservation checks
are available during development. Full NekRS MPI/GPU compilation and short-run
validation are required before judging stability or permissible timestep.

## Mixture viscous correction switch

`[CASEDATA] mixtureViscousCorrectionEnabled = 0.0` disables the explicit
physical two-phase stress minus native base stress correction. Set it to `1.0`
to recover that correction from commit `1179116`. The option defaults to zero
even when omitted. Native implicit viscosity, outlet damping, gas stress,
drag, filters, masks, diagnostics and the lagged gas pressure remain unchanged.

### Drag controls and local diagnostics

`mixtureImplicitDragEnabled = 1.0` retains the previous mixture drag split;
`0.0` uses the complete explicit mixture drag, removes its added cancellation
source and does not register the mixture drag implicit callback. QG drag remains
implicit in both modes. `dragEnabled` retains its existing global meaning.

`dragAlphaCutoff = 0.0` preserves the previous behavior. Set a positive value
(e.g. `1e-4`) to set drag to zero wherever bounded alpha is strictly below this
value, consistently in QG and mixture equations. It does not alter pressure,
virtual mass, gas reconstruction or other force switches.

At `stabilityMonitorInterval`, two `dragLocation` records identify the global
maximum completed-step QG drag diagonal and physical mixture drag magnitude
(excluding virtual mass), with rank, local node, coordinates, raw alpha,
lambdaD, lambdaD*dt, QG, gas/liquid/mixture velocities, slip magnitude, Reynolds number, raw Ki and
physical mixture drag rate/magnitude.
These use completed-step coefficients, not the frozen source-stage diagonal.
Host copies occur only at monitor intervals; set the interval to 1 when
investigating startup failures.

`mixtureDragEnabled = 0.0` disables only mixture drag, including its implicit
diagonal and cancellation source in either treatment mode. QG drag and virtual
mass retain their existing switches. The default `1.0` preserves prior behavior.

Alpha clipping reads the documented `alphaClipEnabled` key, retaining the old
misspelled key as an alias. Magnitude diagnostics use a case-local kernel to
avoid offset shadowing in the native entrywiseMag implementation.

`dragSlipLimitEnabled = 1.0` limits the magnitude of ug-um used only for
drag to `dragSlipMaximum` (positive, finite, m/s). The default is disabled.
With f=min(1,dragSlipMaximum/|ug-um|), Schiller-Naumann uses f*|ug-ul|
and the shared drag rate is f*Ki(f*|ug-ul|)/rhoG. QG and mixture explicit
sources, implicit diagonals and split compensation use this same rate.
No transported QG, alpha, convection, pressure, drift or stress is clipped.
Drag-location diagnostics also report |ug-um|, dragSlipFactor and ReUsed.
