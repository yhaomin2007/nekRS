`gasPressureInletMaskEnabled=1` suppresses only the UG pressure-gradient force at inlet face nodes (boundary ID 1), including every shared CG/MPI copy of inlet/wall corner nodes. It applies after gradient selection and time filtering, to all three components. Interior nodes, raw gradient diagnostics, mixture pressure projection, and other UG forces are unchanged. Default is 0. This is an exact node mask, not an inlet damping layer.

# Direct native gas advection

Alpha and all three UG scalars now use separate scalar-owned velocity/contravariant buffers populated from the latest available UG scalars. Mixture fluid velocity and its Urst buffers are never overwritten. Native BDF/EXT scalar transport supplies gas material advection directly; there is no mixture-advection cancellation or replacement gas self-advection source.

The alpha RHS is `-alpha*div(UG_advector)` plus implicit numerical diffusion and HPF. With `gasTransportMethod=1` the compression product is formed at Gauss cubature points before projection; with 0 it uses GLL strong gradients. Gas pressure, drag, gravity, stress and lagged liquid VM sources retain their prior effective-inertia scaling. The gas advector uses latest-state alpha for the phase-activity mask and latest-state UG, so the same advector is used for native alpha advection and its compression source. Scalar subcycling and moving meshes are not supported in this case.

The current equations (before filtering/clipping) are

```text
dt(alpha) + Ag(alpha) = -alpha*div(UG_advector) + div(Dalpha*grad(alpha))
dt(ug_i) + Ag(ug_i) + (K/rhoEff)*ug_i
  = -grad_i(p)/rhoEff + rhoGas*g_i/rhoEff + K*ul_i/rhoEff
    + c*Dl(ul_i)/rhoEff + div(alpha*tau_g)_i/(alpha*rhoEff)
    + configured net scalar numerical diffusion
rhoEff = rhoGas + c,  c = VMEnabled*VMcoeff*rhoLiquid
```

`Ag` is NekRS's native scalar operator with latest-state UG as the advector. The alpha compression source uses that same advector. `scalarExtrapolationEnabled=0` (default) gives the scalar solver independent EXT coefficients `[1,0,...]`, so convection and every explicit scalar RHS, including compression and HPF, use the latest available state without history extrapolation. BDF and fluid EXT are unchanged. This is first-order explicit treatment, not implicit convection. With 1, the configured EXT combines nonlinear-term histories; the gas advector itself is still not predicted. The two existing mixture-velocity reconstruction modes are unchanged.

Mixture divergence method 0 reconstructs `(rhoL-rhoG)/rhoM * [Salpha + Am(alpha)-Ag(alpha)+diffusion]`; method 1 uses `(rhoL-rhoG)/rhoM * [BDF(alpha)+Am(alpha)]`. `Am` explicitly uses fluid-owned mixture Urst, whereas `Ag` uses scalar-owned gas Urst. Both divergence choices, filtering, ramp and extrapolation remain. Method 0 still omits HPF/clipping effects and reconstructs diffusion with strong operators.

The historical cancellation diagnostic columns in the CSV now report NaN because cancellation is no longer performed. The obsolete `alphaConvectionMethod` and `gasConvectionMethod` selectors no longer alter transport. The notes below describe historical implementations, not the current direct-advection algorithm.

---

## Cubature gas transport

`gasTransportMethod=1` (default) interpolates alpha and gas velocity to native Gauss cubature points, computes `ug.grad(ug_i)` and `ug.grad(alpha)+alpha*div(ug)` there, multiplies by cubature Jacobian/quadrature weights and projects the complete products to GLL with mass-weighted CG assembly. The alpha `alpha*div(ug)` product is formed before projection; no GLL flux differentiation is used in this mode. Native mixture cancellation remains controlled by the independent convection selectors; set both to zero for the intended cubature equations. `gasTransportMethod=0` restores the prior GLL transport for comparison.

This is a continuous Galerkin quadrature discretization of conservative alpha transport, not a positivity-preserving or DG flux scheme. On curved meshes conservation depends on geometry/quadrature consistency; clipping, masking and regularization still affect inventories. VM, drag, stress, pressure treatment and mixture equations are unchanged. Cubature transport requires native cubature enabled with cubNq >= Nq and scalar subcycling disabled. It uses two four-field cubature workspaces and two four-field GLL workspaces; MPI/GPU runtime validation is still needed.

# Direct gas-velocity transport (UG)

This case now transports `ALPHA, UGX, UGY, UGZ`. The three gas scalars are velocities in m/s. `qg = alpha*ug` is reconstructed only for conservative alpha transport, liquid reconstruction and gas-volume diagnostics. No QG momentum equation is solved.

For each gas component, the implemented equation is

```
dt(ug_i) + ug.grad(ug_i) + (K/rhoEff)*ug_i
 = -grad_i(p)/rhoEff + (rhoGas/rhoEff)*g_i
   + (K/rhoEff)*ul_i + (c/rhoEff)*al_i
   + div(alpha*tau_g)_i/(alpha*rhoEff) + numerical diffusion,
c = virtualMassEnabled*virtualMassCoefficient*rhoLiquid,
rhoEff = rhoGas + c.
```

The native BDF time derivative and linear drag are implicit. The effective inertia is incorporated by dividing all physical force terms and drag by `rhoEff`; `transportCoeff` stays one. Liquid acceleration is lagged: at each completed step it is computed as `(ul^n-ul^(n-1))/dt_n + ul^n.grad(ul^n)`, then used by the next source assembly (with NekRS explicit extrapolation). Its initial value is zero. No explicit negative gas acceleration remains. VM creates no additional mixture force.

Native mixture scalar advection is canceled using the same native scalar operator, including cubature, before adding gas material advection. Scalar subcycling must be disabled. Pressure, gravity, drag, VM and stress sources are masked below `alphaFloor`; division by alpha is protected. Optional smooth masking and velocity clipping apply to UG; their gas-flux changes are reported after multiplication by alpha.

`alphaConvectionMethod=0` (default) uses native scalar advection in the alpha source. Set it to `1` to use `um dot opSEM::strongGrad(alpha)` instead. This affects only the alpha explicit source; the native advection in mixture divergence method 1 remains unchanged. UG uses its independent `gasConvectionMethod` selector. With method 1, the net alpha RHS includes the discretization difference `A_m,strong(alpha)-A_m,native(alpha)`.

`gasConvectionMethod=0` (default) cancels native mixture advection in all three UG sources. Set it to `1` to retain `(um-ug) dot opSEM::strongGrad(ug_i)` as the convection correction instead. Gas self-advection always uses the strong gradient; only the mixture-advection cancellation changes. This can leave the difference `A_m,strong(ug_i)-A_m,native(ug_i)` in the net equation. The alpha and UG selectors are independent.

Both original mixture divergence choices remain available: `mixtureDivergenceMethod=0` reconstructs the alpha RHS, and `1` uses the BDF alpha derivative plus native mixture advection. Density averaging, variable mixture density, drift stress, filters and outlet damping remain configurable.

**Restart:** old `ALPHA,QGX,QGY,QGZ` checkpoints are incompatible with UG scalar semantics. Start a new run (the supplied restart line is disabled), or explicitly convert each old gas momentum scalar to velocity before loading. New checkpoints store UG. `subtractUgxDiffusion`, `subtractUgyDiffusion`, and `subtractUgzDiffusion` replace the former QG diffusion-subtraction parameter names. VM is enabled with coefficient 0.5 in the supplied parameters.

The following historical notes describe the earlier QG implementation and are retained for reference only.

---

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

At outlet ID 2, pressure uses the same smooth turbulent/open-outlet condition
as the NekRS `turbPipe` example. It approaches zero for outward flow and adds
the kinetic-pressure correction `-0.5*|u_m|^2` during local backflow, with the
transition controlled by `0.5*(1-tanh(20*u_m.n))`.

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

NekRS's native dealiased scalar-advection operator continues to use the mixture
velocity and supplies `A_m(s)=u_m.grad(s)`. The explicit alpha source is
`A_m,user(alpha)-D(q_g)`, and the explicit source for each gas-flux component is
`A_m,user(q_i)-D(q_i*u_g)`. The user-side mixture-advection terms cancel the
native contribution, leaving the conservative gas continuity and gas-flux
equations. All user-side SEM gradients and divergences use NekRS's normalized
gather-scatter assembly (`avg=true`), including `D(q_g)`, `D(q_i*u_g)`, and the
constitutive gradients. The native scalar advection remains cubature-dealiased;
the reconstructed user terms are GLL-grid operators, so their cancellation is
mathematically exact in the continuum but not necessarily identical at the
discrete level. A Newtonian Stokes stress is used for `tau_g`.

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
of one adds the lagged weak-form source `+M^{-1} K(diffusionCoeff) s`, using
the same SEM stiffness discretization and pointwise scalar diffusion
coefficient as the native Helmholtz solve. `sumMakef` multiplies this nodal
source by the mass matrix, so it opposes the implicit `+K s` term at
unconstrained degrees of freedom, up to the expected explicit temporal
lag/extrapolation. Dirichlet values remain imposed by the native scalar
boundary treatment.

Near the top outlet, `zRampBottom`, `zRampTop`, and
`outletDampingFactor` define a smooth damping layer. The multiplier is one
below `zRampBottom`, follows the cubic smoothstep
`1+(factor-1)*s^2*(3-2*s)` between the two heights, and equals the requested
factor above `zRampTop`. It multiplies the implicit mixture viscosity and the
native implicit diffusion coefficients of ALPHA, QGX, QGY, and QGZ. The
explicit gas viscous-stress source is deliberately not scaled because doing so
would tighten its explicit timestep restriction. The scalar coefficients are
restored from their original `.par` values before each update, so the
multiplier never compounds over successive time steps.

`validationOutputInterval` controls how often integral conservation checks are
written to `bubbleColumn_conservation.csv`. The diagnostics track gas-volume
inventory and full advective/diffusive boundary fluxes, total mass and mass
boundary fluxes, separate inlet/outlet contributions, alpha-volume changes
caused by clipping, QG changes caused by the low-alpha mask, and raw and
clip-corrected cumulative conservation errors. They also report two targeted
discretization checks: the difference between the volume integral of the
assembled `D(q_g)` and the boundary integral of `q_g.n`, and the difference
between the native dealiased and reconstructed user-side volume integrals of
`u_m.grad(alpha)`. Integral histories are updated every completed step so
changing the CSV output interval does not change the cumulative balances. Flux
columns use the outward-normal sign convention.

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
A third field-file series, `ul0.f*****`, stores the liquid velocity reconstructed
from the current alpha, QG, and density-averaged mixture velocity. Both phase
velocities are refreshed from the current solved fields immediately before
checkpoint output.

## One-pass ordering

There are no corrector iterations inside a time step. nekRS first reconstructs
`u_g` and constructs all explicit sources, including the conservative
corrections to its native mixture-velocity scalar advection. It then solves all
four scalars, refreshes mixture properties and the prescribed divergence, and
finally solves mixture velocity/pressure. The gas-flux equation therefore uses
the pressure gradient available at source assembly, not the pressure produced
later in the same step.

- `gasPressureGradientMethod = 0|1` selects the pressure-gradient discretization used by the QG pressure source: 0 uses the assembled strong gradient and 1 uses a mass-normalized weak-gradient field derived from `core-wGradientVolumeHex3D`. Both raw fields are retained for diagnostics; checkpoint output writes `gradpStrong` and `gradpWeak`, and the stability-monitor cadence prints their L2 difference, relative L2 difference, and RMS-magnitude difference.
When `gasPressureGradientFilterEnabled = 1.0`, that lagged gradient is
exponentially relaxed once per physical timestep:

`gradP_used <- (1 - weight) gradP_used + weight gradP_lagged`.

The stored value consequently contains a decaying history of previous completed
steps. Setting `gasPressureGradientFilterWeight = 1.0`, or disabling the
filter, recovers the unfiltered one-pass pressure forcing. This option does not
change the NekRS BDF/EXT orders, time stepper, or pressure projection.

`gasPressureEnabled` controls only the resulting
`-grad(p)/rho_g` contribution in the three gas-flux equations. Use `1.0` for
the physical equation or `0.0` for a diagnostic run without gas-pressure
forcing.

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

## Uniform initialization for the density-averaged formulation

Fresh starts use `alpha=alphaInitial`, `qgx=qgy=0`, and
`qgz=alphaInitial*gasInitialVelocity` uniformly. Initial mixture velocity is
`um_z=rhoGas*qgz/((1-alphaInitial)*rhoLiquid+alphaInitial*rhoGas)`,
with zero transverse components. This is the density-averaged velocity for
initially stationary liquid; it differs from the volume-averaged initialization
in bubbleColumn2. `gasInitialVelocity` defaults to zero; `ugInitial` is an alias.
Plume height/thickness controls are removed. Restart fields are preserved.
Inlet controls remain independent, and prescribed boundaries may adjust the
uniform initial field. Gas reconstruction and low-alpha masking remain active.
Existing divergence method, extrapolation, ramp, force terms and time integration
are unchanged; this change does not establish stability with nonzero divergence.
Run `python tests/uniform_initialization.py` for a serial kernel check.

Divergence consistency correction: method 0 now follows `alphaConvectionMethod` when reconstructing its mixture-advection term. Native advection used by method 1 is assembled as `M^-1 Q^T M_e A_native`, because the native unweighted kernel already returns normalized nodal values. HPF, clipping, and differences between strong and implicit diffusion still prevent method 0 from reproducing the complete discrete alpha update.
