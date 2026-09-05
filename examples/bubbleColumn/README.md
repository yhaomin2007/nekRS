# bubbleColumn

Initial one-pass Eulerian mixture/gas-velocity scaffold for nekRS. No mesh is
included yet. A future mesh must expose only these boundary IDs:

The column axis and upward inlet-flow direction are (+z); gravity acts in
(-z).

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

The scalar diffusivities are set to `1e-12` only to retain well-posed scalar
Helmholtz solves. They approximate the desired nondiffusive transport.

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
Both switches default to `0.0`, preserving the force-free baseline and allowing
drag and virtual mass to be activated and tested one at a time.

Lift, turbulent dispersion, and wall lubrication remain zero, matching the
official OpenFOAM Foundation `multiphaseEuler/bubbleColumn` tutorial. The
alpha-weighted gas viscous-stress contribution in Eq. (29) is still absent; the
tiny scalar diffusivity is numerical, not a physical model.
