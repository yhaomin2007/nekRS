# bubbleColumn2: volume-averaged Eulerian--Eulerian formulation

This is a separate development example derived from `bubbleColumn`; the
original case is unchanged. It retains the conservative gas variables

```text
alpha = alpha_g
qg    = alpha_g*ug
```

but interprets the native nekRS fluid velocity as the phase-volume average

```text
uv = (1-alpha_g)*ul + alpha_g*ug = (1-alpha_g)*ul + qg.
```

For incompressible phases without phase change, the two phase-continuity
equations give the exact pressure constraint `div(uv)=0`. The
`userDivergence` callback therefore leaves the zero-filled nekRS divergence
target unchanged. No reconstructed mixture-divergence source, filter,
extrapolation, or ramp is used.

## Reconstructed phase velocities

```text
ug = qg/max(alpha_g, alphaFloor),
ul = (uv-qg)/max(1-alpha_g, alphaFloor).
```

The existing low-alpha mask and optional gas-velocity limiter remain active.
The emergency native-velocity limiter is exposed as
`volumeVelocityClipEnabled` and `volumeVelocityMaximum`.

## Alpha and gas momentum

The solved gas equations remain

```text
d(alpha_g)/dt + div(qg) = diffusion/regularization,
d(qg_i)/dt + div(qg_i*ug) = gas momentum sources.
```

nekRS supplies native scalar advection with `uv`. The user source cancels that
native term and replaces it with the assembled conservative gas flux, using
the same approach as the current `bubbleColumn` case.

## Volume momentum and pressure projection

Dividing each phase momentum equation by its constant phase density and adding
the two equations gives

```text
d(uv)/dt + div(uv*uv + Tdrift)
  = -Ap*grad(p) + g + viscous terms + interphase-volume term,

Ap     = (1-alpha_g)/rho_l + alpha_g/rho_g,
Tdrift = alpha_g*(1-alpha_g)*(ug-ul)*(ug-ul).
```

nekRS uses `1/rho` in its pressure operator. The property callback therefore
stores `rhoPressure=1/Ap` as the native fluid density, so the native
zero-divergence projection applies the required variable pressure mobility.

The effective implicit kinematic viscosity is approximated by

```text
nuEffective = (1-alpha_g)*mu_l/rho_l + alpha_g*mu_g/rho_g,
muEffective = rhoPressure*nuEffective.
```

Because interphase forces cancel only in mass-weighted momentum, the volume
equation retains

```text
alpha_g*(1-rho_g/rho_l)*agInterphase,
```

where `agInterphase` contains the drag and optional virtual-mass acceleration
used by the gas equation.

## Boundary conditions

Boundary IDs are `1=inlet`, `2=outlet`, and `3=wall`. At the inlet the liquid
velocity is zero, so

```text
uv_z = alphaInlet*gasInletVelocity,
qg_z = alphaInlet*gasInletVelocity.
```

The outlet uses zero-normal-gradient velocity/scalar conditions and the same
turbulent pressure outlet as `bubbleColumn`. The wall uses no-slip `uv`, zero
normal gradient for alpha, and zero QG.

## Output and current limitation

The primary checkpoint velocity is `uv`. Reconstructed `ug` and `ul` are
written separately. Conservation diagnostics are written to
`bubbleColumn2_conservation.csv`; total mass flux is evaluated from

```text
rho_l*uv + (rho_g-rho_l)*qg.
```

This remains a one-pass user-side coupling. The QG equation uses the lagged
pressure gradient because the scalar solve precedes the current-step pressure
projection. A fully OpenFOAM-like phase-momentum pressure corrector would
require deeper solver coupling or outer iterations.
