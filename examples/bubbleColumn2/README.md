# bubbleColumn2: method2 with method1 scalar transport

The native velocity is the volume average
`UM=(1-alpha)*UL+alpha*UG`, with `div(UM)=0`.
`UL=(UM-alpha*UG)/(1-alpha)` uses denominator protection near alpha=1.
The checkpoint scalars are `ALPHA,UGX,UGY,UGZ`; `qg=alpha*UG` is derived.
NekRS core and scalar / pressure / velocity solve ordering are unchanged.
Method1 and other cases are unchanged by this update.

## Alpha and gas velocity

### Current test: solved UG, user mixture forces off

The supplied `.par` selects `frozenGasVelocityEnabled=0`,
`smoothGasVelocityProfileEnabled=1`, `mixtureForcingEnabled=0`, and
`virtualMassEnabled=0`. Start a fresh run. All three UG scalars are solved
with their existing pressure, gravity, implicit drag, convection and viscosity
terms. Alpha diffusion, filters, clipping and conservative transport are unchanged.

The initial UG and inlet use the same uniform core / smooth 5 mm wall taper
as the successful frozen test, using the profile geometry controls below.
The smooth profile initializes UG throughout the column only on a fresh start;
it is not restored during timestepping unless freezing is explicitly enabled.
Fresh UM is initialized to `alpha*UG`, retaining initially stationary liquid.
Restarts retain the loaded UG/UM/alpha; the inlet still uses the smooth profile.

`mixtureForcingEnabled=0` independently disables all user UM acceleration
sources (drag, drift stress, gravity and optional viscous correction), including
mixture drag history. It does not disable gas momentum forces or native UM
convection, viscosity, pressure projection, boundary conditions, or alpha-dependent
properties. The startup line must report `UG solved=1, smooth initial/inlet
profile=1, user mixture forcing=0`. This is a diagnostic, not a fully coupled
physical solution. Set the master switch to 1 to restore user mixture forces;
the existing individual switches then apply. Its default when absent is 1.

### Frozen UG isolation test

To reproduce the successful frozen test, set `frozenGasVelocityEnabled=1`
and keep `mixtureForcingEnabled=0`, `virtualMassEnabled=0`. Start a fresh run
for a clean comparison. Set `frozenGasVelocityEnabled=0` to solve UG again.

The prescribed field is `UG=(0,0,gasInletVelocity*w(r))` at every height,
including above the initial alpha plume and in zero-alpha regions. For a
circular column along z, `r=hypot(x-centerX,y-centerY)` and
`w=s*s*(3-2*s)`, `s=clamp((R-r)/width,0,1)`. Thus the core is uniform,
the last `frozenGasWallRampWidth=0.005` m tapers smoothly, and the wall is zero.
`frozenGasRadius=0` infers R from boundary ID 3 across all ranks; a positive
value supplies R explicitly. The center defaults to `(0,0)` and is controlled
by `frozenGasCenterX/Y`. These distances use mesh units. Resolve the wall ramp
with sufficient nodes; a smooth analytic profile can still be underresolved.

UG fields retain their scalar/checkpoint layout, but their native compute
flags are disabled on both host and device, so they are not advanced or
subjected to native scalar Dirichlet updates. Reconstruction restores the
stored prescribed field; gas clipping and alpha masks are bypassed in this
mode. The inlet uses exactly the same radial weight, including the native UM
inlet `alphaInlet*gasInletVelocity*w`. On a fresh start, UM is initialized to
`alpha*UG` (initially stationary liquid); on restart, UM/alpha are retained
while UG is replaced. A startup `FROZEN UG` message reports the actual radius,
speed and width.

Alpha still uses the existing conservative cubature source and native UM
advection cancellation. Its diffusion, HPFRT/AVM, clipping, explicit filter,
initial plume and boundary conditions are retained for a controlled comparison.
AVM uses the prescribed UG for alpha; UG AVM and implicit drag callbacks are
skipped. UM continues to evolve with its native convection, pressure projection,
viscosity and boundary conditions, and its properties still depend on alpha.
By default `frozenGasMixtureForcingEnabled=0` zeros all user mixture forces
(drag, drift stress, gravity and the optional viscous correction), including
the drag history contribution. This avoids applying a sum of phase momentum
equations while one phase velocity is externally fixed. In particular, the
mixture drag is proportional to `alpha*Ki/(rhoGas*(1-alpha))*(UM-UG)` and has
positive feedback on UM when UG cannot respond. The original frozen-UG test
retained this source and could therefore develop a fluid CFL blowup even if
alpha transport was stable. Set `frozenGasMixtureForcingEnabled=1` only to
reproduce that coupled-source diagnostic (also set `mixtureForcingEnabled=1`).
This frozen-only switch has no effect when
`frozenGasVelocityEnabled=0`. This is a prescribed-gas-velocity transport test,
not a coupled momentum solution. VM must be off (validated at startup).

The scalar transport follows method1 at the synchronization commit:

- Alpha uses direct cubature differentiation of the product `alpha*UG`,
  projected to GLL and assembled with lumped-mass weights. This represents
  `d(alpha)/dt + div(alpha*UG) = numerical diffusion/regularization`.
- UG uses cubature `UG.grad(UG)`, not `div(UG tensor UG)`.
- The exact native mixture-advection contribution is canceled in each scalar
  RHS with matching mass assembly; native scalar velocity pointers still refer
  to UM. There is no temporary replacement of the fluid velocity.
- All scalar nonlinear sources/advection use EXT1, using separate scalar EXT
  coefficient storage. Native BDF history/order and fluid EXT are retained.
- Gas drag uses the native scalar implicit reaction: `lambda=Ki/rhoEff`,
  with lagged `Ki` and `UL`; the explicit RHS contains `lambda*UL` and the
  left side contains `lambda*UG`. Mixture drag remains a full explicit slip
  force, with its existing independent switch/ramp/cutoff/limiter.
- UG is retained at zero/dilute alpha by default, as in method1.
  `gasVelocityLowAlphaCutoffEnabled=1` enables the optional hard/smooth mask;
  magnitude clipping remains independent. Physical stress uses
  `div(alpha*tau_g)/(max(alpha,alphaFloor)*rhoEff)`.

Conservative cubature transport is the fixed case implementation. Set
`cubaturePolynomialOrder >= 2*polynomialOrder` (example: 14 for degree 7).
The copied user interpolation-header helper supplies missing native `c_I`
combinations without modifying installed NekRS files. Fixed mesh, four native
scalars with `transportCoeff=1`, and no scalar subcycling are required.

`alphaExplicitFilterModes` / `alphaExplicitFilterStrength` configure the same
standalone bubble-basis filter as method1. After each scalar advancement the
sequence is filter alpha, mass assembly, reapply scalar Dirichlet values, then
optional clipping. Strength 0 disables this filter; HPFRT remains independent.
The example uses one mode at strength 0.05, diffusion 1e-5, and HPFRT one mode
at scaling 10 for alpha/UG, matching method1. Filtering is not a positivity
limiter and need not preserve gas volume on curved elements. Conservation CSV
accounts for filter and clipping volume changes separately.

## Initial and boundary conditions

Fresh starts use the same plume alpha/UG fields and parameters as method1:

```
f = 0.5*(1-tanh((z-initialPlumeHeight)/initialPlumeThickness))
alpha = alphaInlet*f
UG = (0,0,gasInitialVelocity*f)
UL = 0
UM = alpha*UG
```

The last line is deliberately the volume average, rather than method1's density
average. The example uses alphaInlet=0.016667, gasInitialVelocity=0.3,
height=0.05 and thickness=0.01. The obsolete uniformInitialCondition/alphaInitial
path is removed. Restart files take precedence over fresh initialization.
Only UG checkpoints are compatible; old QG fields cannot be read as velocity.

At the inlet alpha=alphaInlet, UGz=gasInletVelocity and
UMz=alphaInlet*gasInletVelocity. Both velocities are zero on walls. Shared
inlet/wall rim nodes have wall precedence using the same nodal boundary mask as
method1. The outlet keeps its existing pressure and zero-Neumann scalar types.

UG pressure remains lagged. `gasPressureGradientMethod=0/1` selects assembled
strong / mass-normalized weak **volume** gradient; it is not a new-pressure or
full native phase-pressure boundary correction. Optional exponential filtering
uses `gasPressureGradientFilterEnabled/Weight`. `gasPressureInletMaskEnabled=1`
suppresses UG pressure forcing only on inlet nodes (including shared copies);
the outlet remains active. Defaults match method1: strong, unfiltered, unmasked.

## Virtual mass consistency

Define one physical coefficient for both equations:

`c = virtualMassEnabled * ramp(step) * virtualMassCoefficient * rhoLiquid`.

UG solves the normalized semiimplicit equation

```
(rhoGas+c) * Dg(UG)/Dt = -gradP + rhoGas*g + Ki*(UL-UG)
                         + div(alpha*tau_g)/alpha + c*aL_lagged
```

The gas BDF time derivative uses native UG histories; its own acceleration is
not added again as an explicit VM source. All gas physical forces use the same
`rhoEff=rhoGas+c`. Numerical diffusion/HPFRT remain in their configured UG units.

UM retains the explicit physical volume-mixture source

`alpha*(1-rhoGas/rhoLiquid)*(c/rhoGas)*(aL_lagged-aG_lagged)`.

Do not divide this UM source by rhoEff or omit it as if UM were a density
average. Its liquid acceleration comes from the same completed-step history
field used in UG. The gas history and liquid history use the same selected
first/second-order variable-dt derivative and current material convection.
`virtualMassTimeDerivativeOrder=1` matches method1's liquid-history derivative;
2 remains available, with first-order startup until enough samples exist.
The implicit gas derivative order is controlled by native `timeStepper`.

`virtualMassStartStep=S` leaves VM off through S;
`virtualMassRampStep=R>0` gives `min(1,(step-S)/R)` afterward, and R=0 activates
immediately. History is collected during the delay when VM is enabled.
The default is VM on, coefficient 0.5, no delay/ramp, matching method1's
activation. UM remains explicit with native fluid EXT, while scalar explicit
parts use EXT1. Physical coefficients are paired; the coupled equations are
not solved implicitly together. Pressure mobility remains
`(1-alpha)/rhoLiquid + alpha/rhoGas`. Stability at larger dt is not established.

## Verification

Tests under `tests/` exercise actual kernels/native BDF forcing, cubature
transport and high-degree quadrature, shared-node cancellation, initialization,
inlet rim agreement, dilute UG, implicit gas drag, explicit mixture drag/ramp,
VM histories/coefficient pairing, alpha postprocessing and interpolation fallback.
Full NekRS MPI/GPU build and flow stability require a run on the target system.

## AVM with the method2 property callback

`updateProperties` calls `applyGasVelocityAVM()` after restoring scalar
baseline diffusion and applying the spatial outlet ramp. This replaces the
automatic AVM invocation skipped by NekRS when `userProperties` is registered.
Scalars using HPFRT or `regularization = none` receive no AVM contribution.
To test alpha AVM, replace only its regularization line, retaining other entries:

```ini
[SCALAR ALPHA]
regularization = avm + noiseThreshold=1e-6
```

This threshold is an uncalibrated starting value. AVM requires polynomialOrder
at least 5. The native sensor and viscosity algorithm use UG for their velocity scale.
Each enabled scalar retains its own modal sensor and native AVM parameters.
Keep `subtractAlphaDiffusion`, `subtractUgxDiffusion`, `subtractUgyDiffusion`
and `subtractUgzDiffusion` at zero when testing AVM. The user alpha explicit
filter remains independent and retains its configured strength.

The case calls native `avm::viscosity()` directly with the packed UG field,
then adds the result to the current baseline. It does not call `applyAVM()` or
restore a cached initial diffusivity. Baseline restoration before every property
update prevents artificial diffusion from accumulating. Verbose output reports
per-scalar min/max AVM viscosity with `velocity=UG`. This change leaves the restored
semi-implicit UG drag, explicit UM drag and `freezeDragHistory` unchanged.
Use `virtualMassEnabled = 0` for the current no-VM stability baseline.

## Near inlet gas pressure source ramp

`gasPressureInletRegionEnabled = 1` attenuates all components of the lagged
pressure-gradient force in UG throughout a coordinate-defined inlet layer.
It is independent of `gasPressureInletMaskEnabled`, which masks inlet surface
nodes only. The new region option defaults to zero to preserve the baseline.
For the current inlet at z=0, one example is:

```ini
gasPressureInletMaskEnabled = 0
gasPressureInletRegionEnabled = 1
gasPressureInletZeroEndZ = 0.01
gasPressureInletRampEndZ = 0.02
```

The coordinates are absolute mesh z coordinates. The UG pressure term is
`-w(z)*gradP_used/rhoEff`. The weight is zero at/below zeroEndZ, one at/above
rampEndZ, and `t*t*(3-2*t)` in between, where
`t=(z-zeroEndZ)/(rampEndZ-zeroEndZ)`. The ramp has zero endpoint slopes.
Enabled regions require finite rampEndZ greater than zeroEndZ.

This is a diagnostic modification of the gas momentum equation. It does not
change the native UM pressure solve, density/mobility, drag, gas advection,
alpha transport, AVM, or boundary conditions. Masks multiply when both options
are enabled. Raw strong/weak pressure gradients and the temporal filter history
remain unattenuated; repeated callbacks cannot compound the spatial weight.
The pressure acceleration monitor uses the resulting attenuated UG gradient.
No claim of improved long-time stability follows from this option alone.
