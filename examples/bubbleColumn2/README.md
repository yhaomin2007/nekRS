# bubbleColumn2: explicit-drag one-through solver

Native velocity is the volume average `um=(1-alpha)*ul+alpha*ug`;
scalars are `ALPHA,QGX,QGY,QGZ`, with `qg=alpha*ug`.
The native pressure/velocity solver imposes zero mixture divergence.
Gas pressure remains lagged. Conservative scalar fluxes, advection cancellation,
HPFRT, outlet damping, diffusion switches, boundary conditions, gas masks,
clipping, phase velocity exports and conservation diagnostics are retained.
The experimental `bubbleColumn2_ext` case is separate and unchanged.

## Initialization

With `uniformInitialCondition=1`, fresh starts initialize `alpha=alphaInitial`, `qgx=qgy=0`,
`qgz=alphaInitial*gasInitialVelocity`, and `um=(0,0,qgz)` everywhere.
This corresponds to initially stationary liquid and gas velocity along z.
`gasInitialVelocity` defaults to zero; `ugInitial` is an alias, with the documented
key taking precedence. `uniformInitialCondition=0` selects the original smooth
plume, described below. Restart fields are retained. Inlet values still use `alphaInlet`
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

The branch is restored to the pre-new-pressure baseline `18b99f3`, with only
these case changes. Native core source and scalar-first solve order are restored;
`qgNewPressureEnabled` is removed. Rebuild/reinstall NekRS after pulling if the
previous core modifications were installed.

QG uses the original lagged strong-pressure source. Its three pressure-gradient
components are zeroed on inlet/outlet surface nodes (boundary IDs 1 and 2),
including all shared copies of those nodes. Interior nodes and wall-only nodes
retain pressure forcing. Mixture pressure/velocity and all boundary conditions
are unchanged. Pressure diagnostics use the unmasked gradient.

Initialization (ignored on restart):

```ini
[CASEDATA]
uniformInitialCondition = 1
alphaInitial = 0.016667
gasInitialVelocity = 0.3
initialPlumeHeight = 0.05
initialPlumeThickness = 0.01
```

`1` initializes uniform alpha and vertical gas velocity from `alphaInitial` and
`gasInitialVelocity`. `0` selects the original smooth bottom plume:
`profile=0.5*(1-tanh((z-initialPlumeHeight)/initialPlumeThickness))`,
`alpha=alphaInlet*profile`, `ug_z=gasInletVelocity*profile`.
Height/thickness are in meters. Both modes set `qgz=alpha*ug_z`, `um_z=qgz`,
horizontal components zero, and initially stationary liquid. Inlet/wall boundary
conditions are subsequently applied normally. Default is uniform.

Run `python tests/frozen_drag.py` and `python tests/drag_ramp.py` for serial
checks of actual kernel algebra, EXT history treatment, step-based ramp.
Full NekRS MPI/GPU compilation and stability testing remain required.
