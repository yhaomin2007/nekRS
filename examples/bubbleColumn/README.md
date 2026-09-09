# bubbleColumn

Initial one-pass Eulerian mixture/gas-velocity scaffold for nekRS. No mesh is
included yet. A future mesh must expose only these boundary IDs:

The column axis and upward inlet-flow direction are (+z); gravity acts in
(-z).

For a fresh run, a smooth gas plume is initialized above the inlet at `z=0`:

`profile=0.5*(1-tanh((z-initialPlumeHeight)/initialPlumeThickness))`.

Both `alpha` and `ugz` use this profile. The mixture z-velocity is initialized
consistently as `alpha*rhoGas*ugz/rhoM`, corresponding to stationary liquid;
all x- and y-velocity components remain zero. The plume height and transition
thickness are adjustable in `[CASEDATA]`.

| ID | Patch | Mixture velocity | `alpha` | Gas velocity |
|---:|---|---|---|---|
| 1 | inlet | density-averaged inlet value | fixed inlet value | fixed vertical value |
| 2 | outlet | zero normal gradient | zero normal gradient | zero normal gradient |
| 3 | wall | no slip | zero normal gradient | no slip (`u_g=0`) |

The standard entry files stay small. `bubbleColumnTerms.hpp` owns device fields
and SEM operators; `bubbleColumnEquations.okl` contains equation kernels; and
`bubbleColumnBoundary.oudf` contains boundary data.
The equation kernels are registered as a separate OCCA request, so they are not
injected into every native nekRS boundary-kernel translation unit.

## Equation mapping

- Eq. (15): `divSource`, installed through `nrs->userDivergence`.
- Eq. (20): `driftStress`.
- Eq. (21): native pressure/viscosity plus acceleration source
  `-div(driftStress)/rhoM+g` (nekRS multiplies `o_EXT` by `rhoM`).
- Eq. (27): `alphaSource` for passive scalar `ALPHA`.
- Eq. (29): `ugSource` for passive scalars `UGX`, `UGY`, and `UGZ`.

The prescribed mixture divergence uses `nrs->userDivergence` directly; the
thermodynamic `LOWMACH` option remains disabled because it would require a
thermodynamic pressure `p0th`. The variable mixture density is retained in the
pressure operator through `FLUID PRESSURE ELLIPTIC COEFF FIELD`.

Scalar `diffusionCoeff` and `transportCoeff` values are read directly from the
four `.par` sections and are no longer overwritten by `userProperties()`. The
initial alpha diffusivity is `1e-5`; it is an adjustable numerical
regularization rather than a physical phase-diffusion model.

The four scalar sections also expose nekRS's native HPFRT regularization:

`regularization = hpfrt + nModes=1 + scalingCoeff=1.0`.

The initial setting applies a mild relaxation to only the highest polynomial
mode of `ALPHA`, `UGX`, `UGY`, and `UGZ`. Set `regularization = none` in an
individual scalar section to disable HPFRT for that field. This scalar HPFRT is
independent of the optional direct divergence filter in `[CASEDATA]`.

## One-pass ordering

There are no corrector iterations inside a time step. nekRS first constructs all
explicit sources, then solves all four scalars, refreshes mixture properties and
the prescribed divergence, and finally solves mixture velocity/pressure. Thus
Eq. (29) uses the pressure gradient available at source assembly (the previous
or extrapolated pressure), not the pressure produced later in the same step.
Using same-step pressure would require a second scalar pass or core orchestration
changes, both intentionally excluded here.

## Interphase momentum transfer

`dragEnabled` and `virtualMassEnabled` in `[CASEDATA]` are numeric switches:
use `1.0` to enable a term and `0.0` to disable it independently. Drag uses the
OpenFOAM dispersed-gas Schiller--Naumann model with constant `bubbleDiameter`.
The physical slip is reconstructed from the density-averaged mixture velocity,

`u_l=(rho_m*u_m-alpha*rho_g*u_g)/((1-alpha)*rho_l)`.

Virtual mass uses `virtualMassCoefficient` and the lagged material-acceleration
difference `D_l(u_l)/Dt-D_g(u_g)/Dt`. The history is refreshed after each time
step. This explicit, one-pass treatment is intentionally not algebraically
identical to OpenFOAM's implicit virtual-mass coupling and may require a smaller
time step, especially because `rho_l/rho_g` is large.
Drag defaults to `1.0` for the stabilized test; virtual mass remains `0.0` so
the two closures can be introduced separately.

The Schiller--Naumann drag is treated semi-implicitly as
`lambdaD*(um-ug)`: `lambdaD*um` is explicit and `lambdaD*ug` is added to the
gas-scalar Helmholtz diagonal. The prescribed mixture divergence is constructed
after the alpha solve from the alpha-equation RHS,

`q=-(rhoGas-rhoLiquid)/rhoM`
`*(Salpha+div(diffusionCoeff*grad(alpha))+Sregularization)`.

It therefore does not use a separate finite-difference approximation to
`d(alpha)/dt`, and the alpha numerical-diffusion contribution is retained in
mixture-density continuity. `Sregularization` is the HPFRT/GJP contribution
that nekRS adds to the alpha explicit terms. This expression assumes the
configured alpha `transportCoeff` is one.

The completed divergence source is optionally filtered directly with nekRS's
native HPFRT modal matrix before it is copied to `fluid->o_div`:

`qFiltered=qRaw-strength*(qRaw-F(qRaw))`.

`divergenceFilterModes` selects the highest polynomial modes included in the
filter, and `divergenceFilterStrength` must lie between zero and one. The
available settings use one mode and strength `0.25`, which damps only the
highest mode by 25 percent when enabled. Filtering defaults off because a
filtered divergence is no longer locally identical to the alpha-equation RHS.
Because HPFRT retains the constant modal component, enabling it does not
deliberately remove the mean divergence required by mixture mass continuity.

The mixture velocity also exposes nekRS's native HPFRT regularization in the
`[FLUID VELOCITY]` section. The supplied setting removes one highest mode with
unit relaxation strength; set `regularization = none` to disable it.

`divergenceExtrapolationEnabled` selects how the divergence supplied to the
pressure projection is evaluated. With the default value `0.0`, the pressure
solve uses the divergence reconstructed from the newly solved alpha field.
With value `1.0`, completed-step divergence histories are used instead: the
first prediction is EXT1, `q^(n+1)=q^n`, and subsequent predictions are EXT2,
`q^(n+1)=2*q^n-q^(n-1)`. After a fresh start or restart, the direct current
divergence is used until history is available. Filtering, when enabled, is
applied before a divergence field is entered into this history.

Lift, turbulent dispersion, and wall lubrication remain zero, matching the
official OpenFOAM Foundation `multiphaseEuler/bubbleColumn` tutorial. The
alpha-weighted gas viscous-stress contribution in Eq. (29) is still absent; the
tiny scalar diffusivity is numerical, not a physical model.
