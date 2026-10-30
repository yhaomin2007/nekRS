# bubbleColumn2: method2 with method1 scalar transport

The native velocity is the volume average
`UM=(1-alpha)*UL+alpha*UG`, with `div(UM)=0`.
`UL=(UM-alpha*UG)/(1-alpha)` uses denominator protection near alpha=1.
The checkpoint scalars are `ALPHA,UGX,UGY,UGZ`; `qg=alpha*UG` is derived.
NekRS core and scalar / pressure / velocity solve ordering are unchanged.
Method1 and other cases are unchanged by this update.

## Alpha and gas velocity

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
- Gas drag uses the native scalar implicit reaction with lagged coefficient
  and alpha: `lambdaG=Ki/(beta*rhoEff)`. The RHS contains `lambdaG*UM_old`;
  the LHS contains `lambdaG*UG_new`. This also retains the reconstructed
  liquid-velocity dependence on UG implicitly.
- UM drag uses the fully explicit `D*(UM_old-UG_old)`, where
  `D=rampDrag*alpha*(1/rhoGas-1/rhoLiquid)*Ki/beta`. The coefficient includes
  the cutoff/limiter/ramp. The complete drag acceleration is added once to the
  mixture force during source assembly, before scalar solves and fluid EXT.
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

## Explicit mixture drag

UG retains the native scalar implicit reaction and lagged UM load. UM drag
is restored to a complete explicit source, `D*(UM_old-UG_old)`, alongside VM,
drift and other mixture forces. Only drag is multiplied by the optional
step ramp; the VM load is unchanged. Native fluid EXT/BDF assembly supplies
this same known force to pressure and velocity. There is no UM implicit
reaction callback or post-scalar `preFluid` drag adjustment, and density,
pressure mobility and velocity Helmholtz coefficients are unchanged.

The former `D*dt/g0 < 1` implicit-diagonal abort is removed. Explicit drag
still imposes timestep/coupling constraints; this restoration does not establish
long-term stability or resolve the previously observed alpha oscillations.
The source-stage ramped rate is retained for the `max(Dm*dt/g0)` diagnostic.
Existing mixture drag enable/ramp and cutoff/limiter controls remain available.

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
activation. UM VM remains explicit with native fluid EXT, while scalar explicit
parts use EXT1. Physical coefficients are paired; the coupled equations are
not solved implicitly together. The pressure mobility remains
`(1-alpha)/rhoLiquid + alpha/rhoGas`, without drag rescaling. Stability at larger dt is not established.

## Verification

Tests under `tests/` exercise actual kernels/native BDF forcing, cubature
transport and high-degree quadrature, shared-node cancellation, initialization,
inlet rim agreement, dilute UG, implicit gas drag, explicit mixture drag/ramp and native force assembly,
VM histories/coefficient pairing, alpha postprocessing and interpolation fallback.
Full NekRS MPI/GPU build and flow stability require a run on the target system.

### Prescribed uniform gas velocity diagnostic

Set `prescribedGasVelocityEnabled = 1` in `[CASEDATA]` to hold
`UG = (0, 0, gasInitialVelocity)` everywhere, including the boundaries and
zero-alpha cells. The three UG scalar solvers are disabled through native
`SOLVER = NONE`; UG boundary updates, regularization, low-alpha masking and
clipping cannot change the prescribed field. Alpha retains its existing
conservative cubature transport, diffusion and filters. Pressure and UM still
advance, including their configured drag and explicit virtual mass terms.
This is a prescribed transport test, with gas velocity imposed through walls
rather than the usual gas wall condition.

On fresh starts alpha keeps its plume profile and UM is initialized as
`alpha * UG` (initial UL zero). On restart only UG is replaced; alpha and UM
are retained. Use a fresh start to test whether oscillations develop from
smooth alpha. For the initial diagnostic use `virtualMassEnabled = 0` and
match `gasInletVelocity` to `gasInitialVelocity`. The log prints the constant
UG and confirms that its solves are disabled; no UG residuals are expected.
Set `prescribedGasVelocityEnabled = 0` (default) to restore normal UG solves
and plume UG initialization.
