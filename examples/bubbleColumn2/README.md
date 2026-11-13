# bubbleColumn2: method2 with native SEM gas transport

The native velocity is the volume average
`UM=(1-alpha)*UL+alpha*UG`, with `div(UM)=0`.
`UL=(UM-alpha*UG)/(1-alpha)` uses denominator protection near alpha=1.
The checkpoint scalars are `ALPHA,UGX,UGY,UGZ`; `qg=alpha*UG` is derived.
NekRS core and scalar / pressure / velocity solve ordering are unchanged.
Method1 and other cases are unchanged by this update.

## Alpha and gas velocity

The restored baseline now uses native SEM operators for alpha/UG convection:

- Alpha uses `opSEM::strongDivergence(alpha*UG)`, with native shared-node
  assembly and normalization. This represents
  `d(alpha)/dt + div(alpha*UG) = numerical diffusion/regularization`.
- UG uses `opSEM::strongGradVec(UG)` (assembled component gradients),
  followed by pointwise contraction `UG.grad(UG)`. It is material convection,
  not `div(UG tensor UG)`. No custom cubature transport kernels are registered.
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

The native strong operators return pointwise derivatives after shared-node
assembly. The bare `opSEM::divergence` API returns a weak volume load instead;
it cannot directly replace the pointwise alpha source without boundary and
mass treatment. No extra mass assembly is applied after the strong operators.

The user gas operators are evaluated at GLL nodes without cubature projection.
Native UM/scalar advection cancellation still uses native cubature, with the
restored P7 / cubature14 settings and interpolation compatibility helper.
There is no longer a `cubaturePolynomialOrder >= 2*polynomialOrder` requirement
for the user gas operators. Fixed mesh, four native scalars with
`transportCoeff=1`, and no scalar subcycling are required.

This changes spatial discretization only. Scalar EXT1, native BDF, implicit UG
drag with lagged UL, UM drag excluded from EXT, pressure forcing, inlet profile,
outlet enhancement retain the restored baseline treatment. The additional
user alpha filter has been removed; native HPFRT and alpha clipping remain.
The supplied input has `virtualMassEnabled = 0`. This change does not establish
long-time stability; compare alpha volume balance and stripe onset in runs.

There is no additional user alpha filter or filter-kernel registration.
The supplied input retains diffusion 1e-5 and native HPFRT (one mode, scaling
10) for alpha/UG. `postProcessAlpha` only applies the configured clipping and
records its gas-volume change. Historical conservation CSV filter columns
are retained as zeros for compatibility. The old standalone filter source
files are unused by the active case.

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
The supplied input disables VM; coefficient 0.5 and zero delay/ramp
remain configured for optional VM-on tests. UM remains explicit with native fluid EXT, while scalar explicit
parts use EXT1. Physical coefficients are paired; the coupled equations are
not solved implicitly together. Pressure mobility remains
`(1-alpha)/rhoLiquid + alpha/rhoGas`. Stability at larger dt is not established.

## Verification

Tests under `tests/` exercise actual kernels/native BDF forcing, cubature
transport and high-degree quadrature, shared-node cancellation, initialization,
inlet rim agreement, dilute UG, implicit gas drag, explicit mixture drag/ramp,
VM histories/coefficient pairing, alpha postprocessing and interpolation fallback.
Full NekRS MPI/GPU build and flow stability require a run on the target system.

### SEM transport check

`OCCA_DIR=/path/to/occa python3 tests/sem_transport.py` executes the actual
native GLL gradient/divergence kernels and the actual pointwise UG contraction
with OCCA Serial. It checks polynomial derivatives, nonzero compression,
material versus conservative UG convection, mass normalization, padding and
source wiring. `conservative_alpha.py` checks source signs and exact native
advection cancellation. The retained custom-cubature source file and its old
kernel tests are historical comparisons; that source is not registered or
called by this case. Full target MPI/GPU time integration remains untested.
