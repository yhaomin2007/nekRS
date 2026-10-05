# bubbleColumn2: explicit-drag one-through solver

Native velocity is the volume average `um=(1-alpha)*ul+alpha*ug`;
scalars are `ALPHA,QGX,QGY,QGZ`, with `qg=alpha*ug`.
The native pressure/velocity solver imposes zero mixture divergence.
Gas pressure remains lagged. Conservative scalar fluxes, advection cancellation,
HPFRT, outlet damping, diffusion switches, boundary conditions, gas masks,
clipping, phase velocity exports and conservation diagnostics are retained.
The experimental `bubbleColumn2_ext` case is separate and unchanged.

## Drag treatment

Complete drag is always explicit in both equations, evaluated once from the
previous completed state. Gas receives `Sg=alpha*Ki/rhoGas*(ul-ug)` and
mixture receives `(1-rhoGas/rhoLiquid)*Sg` before optional mixture controls.
Only drag contributions in native EXT history slots are replaced by this
step's force; other sources retain native extrapolation and BDF is unchanged.
There are no drag implicit callbacks, diagonals or split compensation.
The obsolete `frozenDragEnabled` and `mixtureImplicitDragEnabled` options are
removed and no longer read. Explicit drag retains a timestep stability limit.

`dragEnabled` controls drag globally. `mixtureDragEnabled=0` disables mixture
drag only. `dragAlphaCutoff` suppresses drag below that alpha.
`dragSlipLimitEnabled` and `dragSlipMaximum` limit slip used in drag coefficients
only; transported fields and convection remain unchanged.

## Optional mixture drag relaxation

`mixtureDragRelaxation=1.0` disables smoothing (default). For `0<omega<1`,
`D_filtered[n]=omega*D_raw[n]+(1-omega)*D_filtered[n-1]`.
The filter advances once per timestep and initializes from raw force at startup
or restart. It affects only mixture drag, not QG drag or other forces.

## Optional mixture-only drag ramp

In `[CASEDATA]`:

```ini
mixtureDragRampEnabled = 1.0
mixtureDragRampStartTime = 0.0
mixtureDragRampDuration = 0.01
```

Time and duration are in seconds. The multiplier is zero at/before start,
rises linearly to one over duration, and stays one afterward. The default is
disabled. It uses the native source callback's absolute physical simulation time,
so restart continues the ramp rather than restarting it. The ramp applies after
relaxation, without scaling filter history. The log reports its applied factor.
QG drag is never ramped. Ramp and relaxation temporarily alter gas/mixture
exchange consistency; verify stability at full drag and timestep sensitivity.

## Other controls and diagnostics

`mixtureViscousCorrectionEnabled` toggles the explicit physical two-phase
stress minus native base stress. Native implicit viscosity and damping remain.
`driftStressEnabled` toggles the drift tensor divergence. Completed-step drag
location diagnostics recompute physical raw drag, not the filtered/ramped force.
Conservation CSV includes clipping and masking changes. Alpha clipping accepts
`alphaClipEnabled`, with the older spelling retained as an alias.

## Validation

Run `python tests/frozen_drag.py` and `python tests/relaxed_drag.py` for serial
checks of actual kernel algebra, EXT history treatment, relaxation and ramp.
Full NekRS MPI/GPU compilation and stability testing remain required.
