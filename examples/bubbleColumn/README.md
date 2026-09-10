# bubbleColumn

Initial one-pass Eulerian mixture/gas-flux scaffold for nekRS. No mesh is
included yet. A future mesh must expose only these boundary IDs:

The column axis and upward inlet-flow direction are (+z); gravity acts in
(-z).

For a fresh run, a smooth gas plume is initialized above the inlet at `z=0`:

`profile=0.5*(1-tanh((z-initialPlumeHeight)/initialPlumeThickness))`.

Both `alpha` and the reconstructed `ugz` use this profile, while the transported
`qgz=alpha*ugz`. The mixture z-velocity is initialized consistently as
`rhoGas*qgz/rhoM`, corresponding to stationary liquid;
all x- and y-velocity components remain zero. The plume height and transition
thickness are adjustable in `[CASEDATA]`.

| ID | Patch | Mixture velocity | `alpha` | Gas flux `q_g` |
|---:|---|---|---|---|
| 1 | inlet | density-averaged inlet value | fixed inlet value | fixed vertical value |
| 2 | outlet | zero normal gradient | zero normal gradient | zero normal gradient |
| 3 | wall | no slip | zero normal gradient | zero (`q_g=0`) |

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

Because the drift stress is the only mixture-momentum source that depends on
the reconstructed gas velocity, it is suppressed wherever
`alpha < alphaFloor`. Both the drift tensor and its final divergence source are
masked, preventing the SEM derivative from leaking a neighboring gas-dependent
source into nodes where the gas phase is numerically absent. Gravity and the
native mixture pressure/viscosity terms remain active.

Gas velocity is reconstructed as `u_g=q_g/max(alpha,alphaFloor)`. The smooth
activation is used in postprocessing to suppress reconstructed velocity near
the absent-phase limit; the conservative pressure, gravity, drag, and virtual
mass sources retain their physical alpha factors.

With `smoothGasVelocityMaskEnabled = 1.0`, the completed gas-flux solution is
converted to velocity, multiplied by the cubic alpha smoothstep, and converted
back using `q_g=alpha*u_g`. The mask is zero at and below
`gasMomentumCutoff` and one at `gasMomentumFullyActive`.

`gasVelocityClipEnabled = 1.0` additionally caps the post-solve vector
magnitude at `gasVelocityMaximum` while preserving its direction. The supplied
limit is `1.0 m/s`. This is a numerical safeguard rather than part of the
Eulerian--Eulerian model; clipping should be reported and sensitivity-tested
in validation runs. Set the switch to `0.0` to disable the cap.

The optional stability monitor prints one global-max line at the configured
step interval. `max|divTarget|` is the divergence actually supplied to the
pressure solve after optional filtering/extrapolation. `min(alpha)`,
`max(alpha)`, and `mean(alpha)` track void-fraction boundedness and total gas
content. `max|qg|` tracks the transported gas volumetric flux, while
`max|qg-alpha*ug|` detects inconsistency introduced by low-alpha reconstruction
or postprocessing. The raw gas pressure
acceleration is reported separately over alpha values above and below
`gasMomentumCutoff`; `max|ug|` is gas-speed magnitude. `CFL(ug)` is the global
maximum gas-phase CFL evaluated from the reconstructed gas velocity with
NekRS's native CFL operator and the current timestep; it is diagnostic only
and does not control timestep selection.
`max|tauDrift|` is the Frobenius norm of the mixture drift-stress tensor; and
`max(lambdaD*dt)` measures the drag relaxation over one timestep.
`inletMean(alpha)` and `inletIntegral(qg.n)` directly check the alpha
Dirichlet value and conservative gas-volume flux on boundary ID 1; the latter
uses the outward normal and is normally negative at the z=0 inlet.
`prescribed|inletFlux|=alphaInlet*abs(gasInletVelocity)*inletArea` supplies the
reference magnitude. Configure the monitor
with `stabilityMonitorEnabled` and `stabilityMonitorInterval` in `[CASEDATA]`.
- Eq. (27): `alphaSource` for passive scalar `ALPHA`.
- Conservative gas momentum: `qgSource` for passive scalars `QGX`, `QGY`, and
  `QGZ`, where `q_g=alpha*u_g`.

The implemented gas equation is

`d(q_g)/dt + div(q_g*u_g) = -alpha*grad(p)/rho_g`
`+ div(alpha*tau_g)/rho_g + alpha*g + M_g/rho_g`.

NekRS natively advances each `QG*` scalar with `u_m.grad(q_g)`. The explicit
correction `-(u_g-u_m).grad(q_g)-q_g*div(u_g)` converts that operator to the
conservative `div(q_g*u_g)`. Both this correction and the corresponding
`u_m.grad(alpha)-div(q_g)` alpha source use element-local SEM gradients so
their reconstructed native-advection term matches NekRS's element-local
`strongAdvectionVolumeScalarHex3D` operator. Gather-scatter-averaged gradients
remain in use for alpha diffusion, gas stress, and other constitutive terms.
A Newtonian Stokes stress is used for `tau_g`.

The prescribed mixture divergence uses `nrs->userDivergence` directly; the
thermodynamic `LOWMACH` option remains disabled because it would require a
thermodynamic pressure `p0th`. The variable mixture density is retained in the
pressure operator through `FLUID PRESSURE ELLIPTIC COEFF FIELD`.

Scalar `diffusionCoeff` and `transportCoeff` values are read directly from the
four `.par` sections. The supplied case uses `diffusionCoeff=1e-5` for `ALPHA`
and all three `QG*` components. The `QG*` scalar diffusion is numerical; the
physical `div(alpha*tau_g)/rho_g` term is assembled explicitly from the
reconstructed gas-velocity gradient.

The `[CASEDATA]` switches `subtractAlphaDiffusion`,
`subtractQgxDiffusion`, `subtractQgyDiffusion`, and
`subtractQgzDiffusion` optionally apply an IMEX deferred correction. A value
of zero retains the corresponding native implicit numerical diffusion. A value
of one adds the lagged source `-div(diffusionCoeff*grad(s))`, reconstructed
with the same gather-scatter-averaged SEM gradient, pointwise scalar diffusion
coefficient, and strong-divergence sequence for all four fields. This does not
algebraically cancel the new-time Helmholtz or boundary diffusion, and enabling
it reduces the smoothing supplied by `diffusionCoeff`.

The four scalar sections also expose nekRS's native HPFRT regularization:

`regularization = hpfrt + nModes=1 + scalingCoeff=100.0`.

The initial setting applies `scalingCoeff=100.0` to the highest polynomial
mode of the mixture velocity, `ALPHA`, `QGX`, `QGY`, and `QGZ`. Set
`regularization = none` in an
individual scalar section to disable HPFRT for that field. This scalar HPFRT is
independent of the optional direct divergence filter in `[CASEDATA]`.

After all four scalar solves, `alphaClipEnabled = 1.0` clips the completed
void-fraction field to the configurable interval `[alphaMinimum, alphaMaximum]`;
the supplied physical bounds are `[0,1]`. This happens through `nrs->postScalar`
before mixture properties, the alpha-based divergence source, and gas-flux
postprocessing are evaluated, so those operations all see the bounded field.
Clipping is a non-conservative numerical safeguard and its effect on total gas
content should be sensitivity-tested. Set `alphaClipEnabled = 0.0` to disable it.

At each checkpoint the normal case file retains the conservative transported
fields `ALPHA`, `QGX`, `QGY`, and `QGZ` for restart. A second field-file series,
`ug0.f*****`, stores the postprocessed reconstructed gas velocity as its
`velocity` vector, so `u_g` can be visualized directly without replacing QG.

## One-pass ordering

There are no corrector iterations inside a time step. nekRS first constructs all
explicit sources, then solves all four scalars, refreshes mixture properties and
the prescribed divergence, and finally solves mixture velocity/pressure. Thus
The gas-flux equation uses the pressure gradient available at source assembly (the previous
or extrapolated pressure), not the pressure produced later in the same step.
Using same-step pressure would require a second scalar pass or core orchestration
changes, both intentionally excluded here.

`gasPressureEnabled` controls only the lagged `-grad(p)/rho_g` contribution in
the three gas-flux equations. Use `1.0` for the physical equation or `0.0`
for a diagnostic run without gas-pressure forcing. This switch does not alter
the native mixture pressure projection or its variable-density coefficient.

## Interphase momentum transfer

`dragEnabled` and `virtualMassEnabled` in `[CASEDATA]` are numeric switches:
use `1.0` to enable a term and `0.0` to disable it independently. Drag uses the
OpenFOAM dispersed-gas Schiller--Naumann model with constant `bubbleDiameter`.
The physical slip is reconstructed from the density-averaged mixture velocity,

`u_l=(rho_m*u_m-rho_g*q_g)/((1-alpha)*rho_l)`.

Virtual mass uses `virtualMassCoefficient` and the lagged material-acceleration
difference `D_l(u_l)/Dt-D_g(u_g)/Dt`. The history is refreshed after each time
step. This explicit, one-pass treatment is intentionally not algebraically
identical to OpenFOAM's implicit virtual-mass coupling and may require a smaller
time step, especially because `rho_l/rho_g` is large.
Drag defaults to `1.0` for the stabilized test; virtual mass remains `0.0` so
the two closures can be introduced separately.

For the conservative gas-flux equation, Schiller--Naumann drag is
`alpha*Ki*(u_l-u_g)/rho_g`. The `alpha*Ki*u_l/rho_g` part is explicit and
`Ki*q_g/rho_g` is added to the gas-scalar Helmholtz diagonal. The prescribed mixture divergence is constructed
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

`divergenceRampEnabled` optionally introduces the final divergence target
gradually during startup. For `0 < tstep < divergenceRampSteps`, the field
sent to the pressure solver is multiplied by the half-cosine ramp
`0.5*(1-cos(pi*tstep/divergenceRampSteps))`; it is unmodified after the
configured number of steps. The ramp is applied after filtering and optional
extrapolation. Set the switch to `0.0` to disable it.

Lift, turbulent dispersion, and wall lubrication remain zero, matching the
official OpenFOAM Foundation `multiphaseEuler/bubbleColumn` tutorial. The
alpha-weighted Newtonian gas stress is included explicitly; the tiny `QG*`
scalar diffusivity is an additional numerical regularization, not a physical
model.
