# bubbleColumn

Initial one-pass Eulerian mixture/gas-velocity scaffold for nekRS. No mesh is
included yet. A future mesh must expose only these boundary IDs:

| ID | Patch | Mixture velocity | `alpha` | Gas velocity |
|---:|---|---|---|---|
| 1 | inlet | density-averaged inlet value | fixed inlet value | fixed vertical value |
| 2 | outlet | zero normal gradient | zero normal gradient | zero normal gradient |
| 3 | wall | no slip | zero normal gradient | no slip (`u_g=0`) |

The standard entry files stay small. `bubbleColumnTerms.hpp` owns device fields
and SEM operators; `bubbleColumnEquations.oudf` contains equation kernels; and
`bubbleColumnBoundary.oudf` contains boundary data.

## Equation mapping

- Eq. (15): `divSource`, installed through `nrs->userDivergence`.
- Eq. (20): `driftStress`.
- Eq. (21): native pressure/viscosity plus `-div(driftStress)+rhoM*g`.
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

The current drag term is a clearly isolated placeholder controlled by
`dragCoefficient`; its default is zero. Replace it with the selected interphase
momentum-transfer closure before treating the case as a physical model.
The alpha-weighted gas viscous-stress contribution in Eq. (29) is likewise left
as an explicit next closure; the tiny scalar diffusivity is numerical, not a
claim that this physical term has been modeled.
