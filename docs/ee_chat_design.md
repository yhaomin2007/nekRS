# `ee_chat`: fixed-alpha Euler-Euler pressure/velocity development

## Scope of the first stage

This branch intentionally excludes alpha transport and all interphase forces.

- `alpha_g = 0.1`
- `alpha_l = 0.9`
- no alpha equation
- no drag, lift, virtual mass, turbulent dispersion, or wall lubrication
- both phases use no-slip pipe-wall boundary conditions
- the initial verification uses identical phase density, viscosity, initial velocity, and velocity boundary data

The unknowns for the eventual pressure/velocity implementation are

- liquid velocity `u_l`
- gas velocity `u_g`
- one common pressure `p`

with mixture velocity

`u_m = alpha_l*u_l + alpha_g*u_g`.

Mixture velocity is a derived field. It must not replace the two phase-specific velocity boundary-condition spaces.

## Boundary masks

NekRS constructs velocity elliptic masks from the velocity boundary map and derives the pressure map from the velocity map. The E-E implementation must preserve this principle for both phase velocities.

For the first supported boundary set, both phase velocity masks have identical topology:

- pipe wall: zero Dirichlet/no-slip for `u_l` and `u_g`
- recycled/prescribed velocity boundary: phase velocity values prescribed identically in the reduction test
- pressure/outflow boundary: common pressure treatment

Even though the masks are identical in this first test, they should be represented conceptually as separate phase constraints. Masking only `u_m` is not sufficient because `alpha_l*u_l + alpha_g*u_g` can satisfy a boundary condition while the individual phase velocities do not.

## Reduction requirement

For identical phase properties and no interphase forces,

`u_l = u_g = u_single`

must be an invariant solution. Therefore

`u_m = 0.9*u_l + 0.1*u_g = u_single`.

The eventual two-fluid pressure operator must also reduce to the standard single-phase pressure operator in this limit. This is the primary go/no-go test before any additional physics is added.

## Development gates

1. Preserve phase-specific velocity boundary masks.
2. Build `u_l`, `u_g`, and `u_m` state/diagnostic handling with fixed alpha.
3. Verify the two phase states remain identical under identical properties and BCs.
4. Introduce the dedicated E-E pressure correction only after its discrete single-fluid limit has been derived from the current NekRS operators.
5. Run the turbPipe-derived comparison and require velocity and pressure agreement with the original single-phase turbPipe case to numerical tolerance.
6. Only after this passes consider unequal properties, drag, or alpha transport.
