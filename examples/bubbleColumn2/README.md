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
- UM drag uses `D*(UM_new-UG_new)`, where
  `D=rampDrag*alpha*(1/rhoGas-1/rhoLiquid)*Ki/beta`. D is frozen at source
  assembly, including its cutoff/limiter/ramp, and UG is taken after its scalar
  solve and postprocessing. Its UM reaction is negative on the LHS. The
  explicit mixture source contains VM, drift and other forces, with no drag.
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

## Direct native implicit mixture drag

Let `aBdf=g0/dt` and `D` be the frozen UM drag rate above. The physical source
`D*(UM_new-UG_new)` is split as follows:

```
velocity LHS addition = -rhoPressure*D
known acceleration RHS = -D*UG_new
rhoSolve = rhoPressure
```

The case registers `fluid->userImplicitLinearTerm` and returns a field holding
`-rhoPressure*D`. Native NekRS adds it to `aBdf*rhoPressure`, giving the desired
velocity Helmholtz diagonal `rhoPressure*(aBdf-D)`. The sign is negative for
our positive D; adding positive rhoPressure*D would change the physical force.

The existing `preFluid` callback forms the known mass-weighted RHS term once,
after UG solves, from an unmodified native forcing snapshot. It does not divide
or rescale the forcing, BDF history, explicit VM, viscosity or density. The
pressure property and pressure operator remain `rhoPressure` and
`Bp=1/rhoPressure=beta/rhoLiquid+alpha/rhoGas`. The native implicit callback
is consumed by the velocity solver only; its reaction does not automatically
enter the pressure Poisson operator. This intentionally retains native
pressure/velocity splitting, not a fully coupled drag-pressure projection.

The source-stage D includes the drag ramp/cutoff/limiter and is independent of
property/monitor scratch fields. Repeated fluid stages rebuild the known UG
term from the saved forcing rather than accumulate it. UM drag is excluded
from explicit EXT histories; all other explicit loads are preserved.

The case stops if `max(D*dt/g0)>=1`, because the velocity mass/reaction
coefficient would be nonpositive. This is a velocity-diagonal condition, not
a change to density or a guarantee of flow stability. There are no
explicit/implicit drag switches. Existing mixture drag enable/ramp and drag
cutoff/limiter controls remain available. Core files are unchanged.

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
inlet rim agreement, dilute UG, implicit gas drag, semi-implicit mixture drag/ramp and direct native velocity reaction,
VM histories/coefficient pairing, alpha postprocessing and interpolation fallback.
Full NekRS MPI/GPU build and flow stability require a run on the target system.
