# Experimental Eulerian--Eulerian solver: minimal branch

## Implemented model

The branch advances two velocity fields on the same SEM mesh and one shared
mechanical pressure. Gas volume fraction is advanced in conservative form,
`d(alpha_g)/dt + div(alpha_g u_g - D_alpha grad(alpha_g)) = 0`, and
`alpha_l = 1 - alpha_g` is enforced pointwise. `D_alpha` is an optional
constant numerical diffusivity and defaults to zero. Advection retains the
custom conservative gas flux, while nonzero diffusion is advanced implicitly
with NekRS's native CG scalar Helmholtz/elliptic machinery. Its equal-and-opposite
liquid flux makes it cancel exactly from the mixture flux. Phase properties are
constant. The pressure equation uses the
mixture-volume constraint

`div(alpha_l u_l + alpha_g u_g) = 0`.

Interphase drag is Schiller--Naumann. Its same-phase contribution is placed on
each velocity Helmholtz diagonal, and the opposite-phase velocity is refreshed
inside configurable coupling iterations. The shared pressure coefficient is
formed from the inverse local two-phase momentum diagonal, including implicit
drag. Each pressure correction updates both velocities together.

The mixture pressure RHS uses the same curl/curl, grad/div, gather-scatter,
inverse-mass, volume-divergence, prescribed-divergence, and boundary-flux
operators as the standard NekRS pressure path. Two regression cases isolate the
new algorithm:

- `twoFluidSingleFluid`: `K = 0`, equal phase properties and equal initial
  velocities; verifies phase equality and mixture continuity.
- `twoFluidMixture`: projection-only mode with prescribed provisional
  velocities; verifies the shared mixture projection without momentum solves.

## Parameter file

Supply the native `TWO FLUID` section:

```ini
[TWO FLUID]
gasVolumeFraction = 0.1
bubbleDiameter = 1e-3
gasDensity = 1.0
gasViscosity = 1.8e-5
gravityX = 0
gravityY = 0
gravityZ = -9.81
alphaFloor = 1e-8
alphaDiffusivity = 0
dragMultiplier = 1
couplingIterations = 2
pressureCorrectors = 2
projectionOnly = false
```

Liquid density and viscosity remain in `[FLUID VELOCITY]`. The gas momentum
solver inherits the liquid velocity solver, preconditioner, tolerance, and
boundary map settings. `alphaDiffusivity` has units of length squared per time;
set it to zero for pure advection.

Each alpha--momentum--pressure coupling iteration reports
`max|deltaAlphaG|` and `relL2`, where `deltaAlphaG` is the change from the
previous gas-volume-fraction fixed-point iterate. These iteration diagnostics
are distinct from the final `Rg`, `Rl`, and `Rm` phase-continuity residuals.

## UDF access

The liquid and gas velocities are available as:

```cpp
nrs->fluid->o_U
nrs->gas->o_U
```

The shared pressure is `nrs->fluid->o_P`. The prescribed volume-fraction fields
are `nrs->twoFluid->o_alphaG` and `o_alphaL`.

## Deliberate restrictions

The volume-fraction transport presently supports `tombo1`/BDF1. Its bound
correction redistributes any clipping defect over the remaining admissible
capacity, preserving the integral of the raw conservative update for periodic
or impermeable boundaries.

Set `dragMultiplier = 0` to disable interphase drag. `projectionOnly = true`
skips both momentum solves and applies only the shared pressure correction; it
is a debugging option rather than a physical time-advancement mode.

For the inlet--outlet transport verification, `nativeAlphaScalar = true`
requires exactly one standard NekRS scalar named `alpha`. Its scalar solver is
configured with the gas velocity (`o_U`, `o_Ue`, and `o_relUrst`) instead of the
carrier-liquid velocity. The two-fluid coordinator aliases this native scalar
as `alpha_g`, applies the existing conservative bound correction, and enforces
`alpha_l = 1-alpha_g`. This is a narrow verification mode; general simultaneous
alpha and species transport will require per-scalar transport velocities.

The two-fluid weak-divergence operator includes both its volume contribution
and `-n.F` on every physical boundary. Periodic/interior faces receive no
surface term. The same volume-plus-surface operator is used for phase and
mixture diagnostics and for the exact pressure-correction Schur operator. This
prevents prescribed inlet/outlet through-flow from being counted as a local
continuity residual.

This is a verification branch, not yet the full production solver. It currently
requires a fixed mesh and supports inlet/outlet alpha transport only through
the narrow native-alpha-scalar mode described above.
It does not yet include independent gas boundary callbacks, dynamic volume
fraction, lift, wall lubrication, turbulent dispersion, virtual mass, phase
change, or turbulence. Pressure rho-splitting, low-Mach mode, NekNek,
constant-flow-rate control, and subcycling are outside the supported v1
combination.

General inlet/outlet volume-fraction fluxes for the custom conservative alpha
update must still be completed before that path is used for open-boundary
physical validation.
