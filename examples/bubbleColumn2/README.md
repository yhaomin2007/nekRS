# bubbleColumn2: explicit-drag one-through solver

Native velocity is the volume average `um=(1-alpha)*ul+alpha*ug`;
scalars are `ALPHA,QGX,QGY,QGZ`, with `qg=alpha*ug`.
The native pressure/velocity solver imposes zero mixture divergence.
Gas pressure is lagged in the original mode; the optional new-pressure path is described below. Conservative scalar fluxes, advection cancellation,
HPFRT, outlet damping, diffusion switches, boundary conditions, gas masks,
clipping, phase velocity exports and conservation diagnostics are retained.
The experimental `bubbleColumn2_ext` case is separate and unchanged.

## Uniform initialization

Fresh starts initialize `alpha=alphaInitial`, `qgx=qgy=0`,
`qgz=alphaInitial*gasInitialVelocity`, and `um=(0,0,qgz)` everywhere.
This corresponds to initially stationary liquid and gas velocity along z.
`gasInitialVelocity` defaults to zero; `ugInitial` is an alias, with the documented
key taking precedence. Plume initialization and its height/thickness controls
are removed. Restart fields are retained. Inlet values still use `alphaInlet`
and `gasInletVelocity`; wall/inlet conditions may create startup adjustments
when a uniform initial velocity differs from prescribed boundary values.
Gas reconstruction/masks still apply at low alpha.

## Drag treatment

Complete drag is always explicit in both equations, evaluated once from the
previous completed state. Gas receives `Sg=alpha*Ki/rhoGas*(ul-ug)` and
mixture receives `(1-rhoGas/rhoLiquid)*Sg` before the optional mixture ramp.
Only drag contributions in native EXT history slots are replaced by this
step's force; other sources retain native extrapolation and BDF is unchanged.
There are no drag implicit callbacks, diagonals or split compensation.
The obsolete `frozenDragEnabled` and `mixtureImplicitDragEnabled` options are
removed and no longer read. Explicit drag retains a timestep stability limit.

`dragEnabled` controls drag globally. `mixtureDragEnabled=0` disables mixture
drag only. `dragAlphaCutoff` suppresses drag below that alpha.
`dragSlipLimitEnabled` and `dragSlipMaximum` limit slip used in drag coefficients
only; transported fields and convection remain unchanged.

## Optional mixture-only drag ramp

In `[CASEDATA]`:

```ini
mixtureDragRampEnabled = 1.0
mixtureDragRampStartStep = 0
mixtureDragRampSteps = 10000
```

At native timestep `k`, the factor is
`clamp((k-startStep)/rampSteps,0,1)`. Drag is zero at/before start,
rises linearly to full strength at `startStep+rampSteps`, and stays full thereafter.
Default disabled. Start must be nonnegative and duration a positive integer.
The factor uses the native timestep counter, independent of dt or physical time;
a restarted run follows that run's native step numbering. The log reports the
applied factor. Only mixture drag is ramped; QG drag and other forces are unchanged.
Mixture drag under-relaxation and its buffers/history have been removed.
`mixtureDragRelaxation`, `mixtureDragRampStartTime`, and `mixtureDragRampDuration`
are obsolete and no longer read. Verify stability after reaching full drag.

## Other controls and diagnostics

`mixtureViscousCorrectionEnabled` toggles the explicit physical two-phase
stress minus native base stress. Native implicit viscosity and damping remain.
`driftStressEnabled` toggles the drift tensor divergence. Completed-step drag
location diagnostics recompute physical raw drag, not the filtered/ramped force.
Conservation CSV includes clipping and masking changes. Alpha clipping accepts
`alphaClipEnabled`, with the older spelling retained as an alias.

## Validation

Run `python tests/frozen_drag.py` and `python tests/drag_ramp.py` for serial
checks of actual kernel algebra, EXT history treatment, step-based ramp.
Full NekRS MPI/GPU compilation and stability testing remain required.

## QG solve after new mixture pressure (requires rebuilt core)

Set `[CASEDATA] qgNewPressureEnabled = 1.0` (enabled in the supplied par).
The timestep sequence is alpha solve/clipping, property update, mixture pressure,
QGX/QGY/QGZ solves, then mixture velocity. Set it to `0.0` to restore the old
all-scalars-first strong-pressure-source path. Core hooks default to inactive,
so other examples retain their original order. Rebuild and reinstall NekRS
before running this UDF; updating only case files cannot enable the new API.

The gas pressure source is absent from explicit/EXT history in the new mode.
Using updated bounded alpha, define `B=gasPressureEnabled*alpha/rhoGas`.
The already weighted element load is `D^T M(Bp) + M p grad(B)`, using native
weak-gradient and weighted strong-gradient kernels, plus the exterior load
`-integral(p B n_i v_i)` on all physical boundary faces. Interior contributions
cancel through normal elliptic assembly. Prescribed QG inlet/wall DOFs retain
native scalar Dirichlet constraints; scalar diffusion Neumann/Robin conditions
remain independent. Exterior pressure uses the newly solved pressure trace,
including prescribed outlet pressure. This is a coefficient-aware pressure-force
weak form, not a copy of the mixture viscous traction model.

No inverse mass or additional mass weighting is applied to this load. The scalar
RHS hook adds it just before each scalar elliptic solve. After QG advances,
existing gas masks/clipping run once, and ug is reconstructed. Mixture forcing,
pressure mobility and viscosity remain fixed after the pressure solve. Neither
BDF history nor explicit forcing is rebuilt/advanced between pressure and QG.
The new path requires four active native scalar fields in ALPHA,QGX,QGY,QGZ
order on the fluid mesh, an active pressure solver, no pressure rho splitting
and no scalar subcycling. It does not change density-averaged bubbleColumn.

This remains segregated: alpha and gas-dependent mixture sources are lagged
relative to the new QG solution, and drag remains explicit. Larger-dt stability
and gas conservation are not guaranteed. The new startup log identifies the
solve order, and monitor intervals log the after-pressure QG stage.

Serial tests `tests/deferred_order.py` compile actual scalar solve bodies and
fluid hook against stubs; `tests/weak_pressure.py` translates the actual native
volume kernels and case pressure kernels, checking constant/linear pressure,
variable alpha, all components and a shared two-element interface. These do
not replace a complete MPI/GPU build and run.
