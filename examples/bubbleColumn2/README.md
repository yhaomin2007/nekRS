# bubbleColumn2: method 2 with gas velocity scalars

Native velocity is the volume average `um=(1-alpha)*ul+alpha*ug`, with native
zero-divergence projection. Scalars/checkpoint fields are now `ALPHA,UGX,UGY,UGZ`.
Gas flux `qg=alpha*ug` is derived for conservative alpha transport, liquid
reconstruction, force evaluation and conservation diagnostics; it is not solved.
Density-averaged `bubbleColumn` and `bubbleColumn2_ext` are unchanged.
NekRS core and scalars-first / pressure / mixture-velocity ordering are unchanged.

## Gas velocity equation and virtual mass

The physical gas acceleration equation is

`rho_g D_g(ug)/Dt = -grad(p) + rho_g g + Ki(ul-ug)
                    + div(alpha*tau_g)/alpha
                    + c (D_l(ul)/Dt - D_g(ug)/Dt)`

where `c=virtualMassEnabled*ramp*C_VM*rho_l`. Move the gas material acceleration
in VM to the left:

`(rho_g+c) D_g(ug)/Dt = -grad(p) + rho_g g + Ki(ul-ug)
                       + div(alpha*tau_g)/alpha + c*a_l_lagged`.

The implemented normalized equation divides by `rhoEff=rho_g+c`. Native scalar
BDF then advances UG and its native UG history, which is algebraically the same
as multiplying the BDF current/history terms by rhoEff. No explicit negative
gas acceleration is included in the UG source, and no artificial damping
(diagonal-only) approximation is used. All gas physical forces, including
pressure, gravity, drag and stress, use the same effective inertia. The
explicit convection correction cancels native mixture convection and inserts
`ug.grad(ug)` rather than the old conservative `div(qg*ug)` momentum flux.
Alpha still uses conservative `div(alpha*ug)` with exact native-advection
cancellation. `transportCoeff=1` is required for all four normalized scalars.
Numerical scalar diffusion and HPFRT retain their configured UG units.

This is a semiimplicit, segregated formulation: liquid material acceleration
is lagged; the VM contribution to volume-mixture momentum remains the existing
explicit completed-step acceleration-difference source. Mixture pressure
mobility remains `beta/rho_l+alpha/rho_g`. Thus this is not a jointly implicit
VM treatment of both phases/mixture and does not guarantee a larger stable dt.
Do not count the old explicit gas-VM relative-acceleration source again in UG.

## Virtual mass controls

```ini
[CASEDATA]
virtualMassEnabled = 1.0
virtualMassCoefficient = 0.5
virtualMassTimeDerivativeOrder = 2
virtualMassStartStep = 100
virtualMassRampStep = 1000
```

VM is off for steps <= start. Positive ramp duration R applies
`min(1,(step-start)/R)` afterward, reaching full strength at start+R; R=0
activates immediately. Both gas added inertia and the paired mixture VM source
use that factor. History is collected while VM is enabled, even during delay.

`virtualMassTimeDerivativeOrder=1/2` controls the completed-step liquid
acceleration and explicit mixture VM diagnostic/history derivative. Order 2
uses three-level variable-step backward differences, with coefficients
`((2h+k)/(h(h+k)), -(h+k)/(hk), h/(k(h+k)))`. Startup/restart uses order 1 until
sufficient history exists. The implicit gas derivative uses the native BDF
selected by `timeStepper`, not this liquid-history option. Default VM remains off.
Native EXT extrapolation remains for explicit liquid/mixture source parts.

## Pressure, drag and boundaries

Gas pressure remains lagged, with a strong gradient source divided by rhoEff;
it is suppressed only on inlet surface nodes (boundary ID 1), including shared
copies. Outlet (ID 2), wall-only and interior nodes retain pressure forcing.
Existing inlet/outlet/wall boundary types remain. The UG inlet is
`(0,0,gasInletVelocity)`; mixture inlet is `(0,0,alphaInlet*gasInletVelocity)`.
Wall UG is zero and outlet is scalar zeroNeumann.

Drag remains fully explicit in gas and mixture. Old-state gas drag acceleration
is divided by rhoEff, while physical volume-mixture drag retains its original
weight. Drag alone is held fixed in EXT history; other source terms are not.
`mixtureDragEnabled`, `dragEnabled`, `dragAlphaCutoff`, `dragSlipLimitEnabled`,
`dragSlipMaximum` and `mixtureDragRampEnabled/StartStep/Steps` remain available.
There is no drag under-relaxation or implicit drag. Mixture drag ramp is separate
from VM ramp. Inertia does not remove every explicit-drag or coupling constraint.

## Initialization and checkpoint compatibility

```ini
uniformInitialCondition = 1
alphaInitial = 0.016667
gasInitialVelocity = 0.3
initialPlumeHeight = 0.05
initialPlumeThickness = 0.01
```

Uniform mode initializes alpha and UG directly, with `um_z=alpha*ug_z`.
`uniformInitialCondition=0` uses the original smooth bottom profile
`f=0.5*(1-tanh((z-height)/thickness))`, `alpha=alphaInlet*f`,
`ug_z=gasInletVelocity*f`, and `um_z=alpha*ug_z`. Horizontal velocities are zero;
both modes initially have stationary liquid. Restart skips initialization.
The low-alpha mask and velocity clipping now act directly on UG; qg masking
conservation diagnostics still measure changes in the derived alpha*ug flux.

**Old ALPHA,QGX,QGY,QGZ checkpoints cannot be read as UG checkpoints.**
Start fresh for this formulation; only use checkpoints produced by this UG case.
Fields 2/3/4 in new files contain gas velocity, not gas flux. Exports of ug/ul
vectors and the derived-flux conservation diagnostics remain available.
Diffusion subtraction keys are now `subtractUgxDiffusion`,
`subtractUgyDiffusion`, `subtractUgzDiffusion` (default 0).

## Validation

Serial tests under `tests/` compile the actual case kernels. `ug_equation.py`
also invokes the actual native scalar BDF forcing kernel to check effective
inertia and matching time histories. Tests cover direct UG pressure, convection,
flux/liquid reconstruction, physical stress, initialization, masks, explicit
drag histories, step ramps and variable-dt VM histories. They do not replace
full NekRS MPI/GPU compilation and coupled-flow stability testing.
