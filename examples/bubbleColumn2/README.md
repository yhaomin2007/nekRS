# bubbleColumn2: method2 with method1 scalar transport

The native velocity is the volume average
`UM=(1-alpha)*UL+alpha*UG`, with `div(UM)=0`.
`UL=(UM-alpha*UG)/(1-alpha)` uses denominator protection near alpha=1.
The checkpoint scalars are `ALPHA,UGX,UGY,UGZ`; `qg=alpha*UG` is derived.
NekRS core is unchanged; the supplied case defers UG until after native P/UM.
Method1 and other cases are unchanged by this update.

## Current solve order: alpha -> P -> UM -> UG

The supplied `.par` selects `gasSolveAfterFluid=1`. NekRS core is unchanged:
the first native scalar solve advances only alpha; native fluid pressure and UM
then advance; the case calls the native scalar solver once for UG at the end
of the accepted step, using the newly solved pressure. `div(UM)=0` is retained.
There is no Picard iteration and no second UG pressure-corrector solve.

At source preparation, the case builds alpha and UG convection with their
existing live fluid EXT coefficients and prepares UG's native BDF/EXT1/HPF
right-hand sides **before** native solution/history lagging. It then sets the
host solve mask to `(1,0,0,0)`. Native makeForcing shifts source histories once,
and native lagSolution shifts solution histories once. After P/UM, the case
changes the host mask to `(0,1,1,1)`, applies UG boundary conditions and solves
UG once. It restores `(1,1,1,1)` before diagnostics and output.

UG's private native UM-advection arrays are not exposed by v26. The case keeps
the device gas-advection flags zero from initialization and omits their matching
source-side cancellation; manual UG BDF forcing uses a separate zero-advection
buffer. The actual UG cubature convection and its independent fluid-EXT history
are retained. Alpha's native-advection cancellation is unchanged. Host and device
masks intentionally serve different purposes: device flags disable only unwanted
native gas-scalar advection, while host flags schedule actual solves.

For `rhoEff=rhoGas+ramp*Cvm*rhoLiquid` in the selected semi-implicit VM mode,
only the saved direct pressure load is replaced in the already prepared UG RHS:

```
RHS_UG += M * (-gasPressureEnabled/rhoEff)
              * (G_selected(p_new) - G_selected(p_source))
```

The pressure difference is mass-weighted once. It is followed by one full UG
solve, not by a predictor solve plus a corrector. Strong/weak gradient selection
and inlet spatial masks remain available. Temporal pressure relaxation and the
hydrostatic-only gradient option are incompatible with this latest-pressure mode.

All other explicit forces/convection stay at their source-time levels; UG drag
still uses lagged UL. This change does **not** recalculate drag from newly solved
UM, nor does it make the phase coupling fully implicit. The source-time implicit
drag coefficient is snapshotted to keep it paired with its explicit UL load.
UG diffusion/AVM coefficients are captured after alpha properties update and
restored for the deferred solve. Final phase fields/properties are refreshed
after UG; completed-step VM acceleration histories update only afterward.
P5/cubature10, dt, VM ramp, VM-source EXT and all force switches are retained.

`gasSolveAfterFluid` defaults to 0 for other cases that reuse these headers;
those retain the legacy order. Set it to 0 and explicitly set
`gasPressureCorrectorEnabled=1` to restore the preceding pressure-corrector test.
Setting it to 1 automatically turns that corrector off. The mode requires four
active elliptic scalars, fixed mesh, one native stage, no scalar/fluid advection
subcycling or NekNek, and no coupled-VM inertia-elimination mode.

`gasSolveOrderLogInterval=10` prints the first step and every tenth step on rank
zero. Set 1 for every step or 0 to disable. The stages are `ALPHA_PENDING`,
`ALPHA_DONE`, `P_UM_BEGIN`, `P_UM_DONE`, `UG_BEGIN_CURRENT_PRESSURE`, `UG_DONE`,
and `STEP_FIELDS_READY`, with the actual host solve mask on each line. Native solver residual summaries
retain their usual end-of-step reporting.

For a fresh run, rebuild the UDF cache after pulling this change. The new host
orchestration is checked in C++ against native scalar BDF forcing and direct
new-pressure solves, including variable dt, VM ramp normalization, operator
snapshots, boundary values, masks and history preservation. Full target MPI/GPU
build and long-time stability still require a run on the target installation.

## Pressure-gradient output

At every native checkpoint/output step, two additional vector field series are
written using the same `writeToFile` path as the existing `ug` and `ul` files:

| Prefix | Vector contents | Time represented |
| --- | --- | --- |
| `gps` | Raw assembled `opSEM::strongGrad` of the newly solved pressure | Current output time |
| `gpp` | Selected gradient actually used by UG, including inlet attenuation | Latest pressure in after-fluid mode; lagged predictor pressure in legacy mode |

In VisIt, open each corresponding `.nek5000` database and plot the `velocity`
vector components (especially the z component); these slots contain **positive
pressure gradients**, not velocities or pressure accelerations. Units are Pa/m
for this dimensional case. The UG pressure source has the opposite sign and
includes its density normalization. `gasPressureEnabled` is not included in
these exported gradients.

With the supplied strong-gradient settings, filtering and inlet attenuation
off, `gps` and `gpp` both show the latest gradient used by the deferred UG solve.
In legacy order, `gpp` instead records the lagged predictor gradient. Both files
carry the output time for alignment, although in legacy order `gpp` represents
the lagged predictor pressure. If gradient selection/masking/filtering is changed, `gps`
continues to show the raw current strong gradient. It does not claim to show
the configured corrector gradient in those other modes.

The UG gradient is snapshotted at source assembly in legacy order and refreshed
immediately before the deferred solve in after-fluid mode, because later
property/monitor callbacks overwrite the shared gradient buffers. Current
strong-gradient output uses a separate temporary buffer. These diagnostics do
not change forcing, solution histories, pressure or restart scalar layout.

## Alpha and gas velocity

### Convection time extrapolation

Alpha and UG now use the same live native fluid EXT coefficients as UM for
**complete assembled nonlinear convection operators**. For constant dt after
startup, the current EXT3 setting applies `3*N[n]-3*N[n-1]+N[n-2]`, where
`Nalpha=div(alpha*UG)` and `Ng=UG.grad(UG)` use the existing conservative and
material cubature kernels. Startup and variable-dt coefficients come directly
from `nrs->fluid->o_coeffEXT`; no fixed EXT3 constants are imposed in the case.
The gas advector is not predicted before forming the operator.

A separate case-local history stores only these four convection fields. It
advances once per physical source step; repeated source calls refresh the same
slot, and initial CFL probes do not advance history. First use fills missing
prehistory with the initial operator. Native scalar coefficients remain EXT1,
so viscous sources and diffusion counterterms keep their existing time treatment.
Direct pressure is refreshed separately in after-fluid mode. Native UM scalar
advection is canceled for alpha; its UG pair is disabled in after-fluid mode. The explicit
liquid VM source can also use
its separate native-fluid EXT history in the selected VM mode. UG implicit
drag, UM sources/convection, BDF histories, filters and the VM delay/ramp are
unchanged by the convection update. Alpha's instantaneous flux diagnostics retain their existing
meaning; the actual timestep convection load is the EXT combination of flux
operators rather than the instantaneous diagnostic alone.


### Current test: semi-implicit UG VM, extrapolated explicit VM loads

The supplied `.par` selects `virtualMassExtrapolationEnabled=1`,
`explicitVirtualMassEnabled=0`, `coupledVirtualMassEnabled=0`, and paired
`virtualMassEnabled=1`, `mixtureVirtualMassEnabled=1`, `Cvm=0.5`.
The existing P5/cubature10, dt=1e-5, physical phase-stress
correction, filters and VM delay/ramp (100 / 10000 steps) remain selected.
The supplied case now uses the after-fluid order above with its separate pressure
corrector disabled. Start from a fresh state and confirm startup prints both
`VM EXTRAPOLATION` and `Method2 solve order`.

Let `c=virtualMassEnabled*ramp*Cvm*rhoLiquid`. UG keeps its own VM
material-inertia contribution on the left:

```
(rhoGas+c)*Dg(UG)/Dt = -gradP + rhoGas*g + Ki*(UL-UG)
                       + div(alpha*tauG)/alpha + c*aL
```

Native BDF handles the gas time derivative and native UG histories; gas
convection retains the separate fluid-EXT treatment described above. This is
semi-implicit gas inertia, not a fully implicit convection/phase solve.
Only the explicit liquid-acceleration VM load in the normalized UG RHS changes
from its current sample `qG=c/(rhoGas+c)*aL` to `EXT_fluid[qG]`. The history
stores the full normalized VM source, including the sampled ramp coefficient,
so no pressure, gravity, drag or viscous source is newly extrapolated with it.
At constant dt after startup this is `3*qG[n]-3*qG[n-1]+qG[n-2]`.

UM retains its full physical VM acceleration source

```
qM=alpha*(1-rhoGas/rhoLiquid)*(c/rhoGas)*(aL-aG)
```

and native fluid EXT applies to this complete source. It is not divided by
`rhoGas+c`, and its EXT history is not flattened by the fully explicit
last-step experiment. The UG explicit part uses `aL`; the UM VM uses the
complete `aL-aG` difference. Both acceleration histories still come from the
last completed step, with `virtualMassTimeDerivativeOrder=1` as before.

The separate UG VM history uses live native fluid EXT coefficients for startup
and variable dt. Initial CFL probes do not advance it; repeated same-step
source calls refresh rather than shift it. The first sample fills prehistory
with a constant value. The shared ramp multiplies each source sample before
extrapolation, so the extrapolated load need not equal the instantaneous ramp
factor times one acceleration sample. Histories continue collecting during
the VM delay.

`virtualMassExtrapolationEnabled` defaults to 0 and is mutually exclusive with
both coupled inertia elimination and fully explicit gas inertia. With VM on,
it requires paired mixture VM and user mixture forces. The phase-stress
correction remains active automatically in this experiment, preserving the
physical viscosity treatment from the preceding case. Neither this temporal
extrapolation nor the ramp establishes full spatial stability.

### Previous test: fully explicit UG/UM virtual mass, P5 / cubature 10

The preceding fully explicit test selects `polynomialOrder=5`,
`cubaturePolynomialOrder=10`, `explicitVirtualMassEnabled=1`,
`coupledVirtualMassEnabled=0`, `virtualMassEnabled=1`,
`mixtureVirtualMassEnabled=1`, and `virtualMassCoefficient=0.5`.
`dt=1e-5`, the pressure corrector, solved UG, smooth initial/inlet profile,
mixture forces, drag model, filters, scalar diffusion and outlet enhancement
retain their existing settings. Use a fresh start with the existing mesh.
Check that startup prints `FULLY EXPLICIT VM` and `UG pressure corrector ON`.

VM is delayed and then introduced gradually in both equations using the same
factor. The supplied settings are `virtualMassStartStep=100` and
`virtualMassRampStep=10000`:

- Steps 1 through 100: VM force is zero in both UG and UM.
- Steps 101 through 10100: `r=min(1,(step-100)/10000)`.
- Step 10100 onward: full `Cvm=0.5`.

At the retained `dt=1e-5`, the delay ends at t=0.001 s and full VM is
reached at t=0.101 s. Completed-step acceleration histories continue updating
during the delay, so the ramp uses recent accelerations rather than an
accumulated difference from initialization. All other force settings and the
fully explicit formulation are unchanged. The ramp tests startup sensitivity;
it does not establish stability of the fully explicit VM coupling at full strength.

For the step from n to n+1, both equations use the same saved material
acceleration difference from the last completed step:

```
aG[n] = (UG[n]-UG[n-1])/dt[n] + UG[n].grad(UG[n])
aL[n] = (UL[n]-UL[n-1])/dt[n] + UL[n].grad(UL[n])
c = virtualMassEnabled * ramp(step) * Cvm * rhoLiquid
UG_VM[n] = (c/rhoGas) * (aL[n]-aG[n])
UM_VM[n] = alpha[n]*(1-rhoGas/rhoLiquid) * UG_VM[n]
```

The supplied derivative order is 1; the existing variable-step second-order
history option remains available. History updates only after the completed
UM solve and UG pressure correction. The first step uses zero VM acceleration
because no completed-step time derivative is available. Histories are not
reconstructed from old algorithm-mode checkpoints.

There is **no added gas inertia** in this mode: gas pressure, drag and stress
use `rhoGas`, not `rhoGas+c`. The entire VM difference, including the gas
acceleration, is an explicit RHS load. UG VM and other non-convective scalar sources retain EXT1.
`freezeExplicitVirtualMassHistory()` separately holds only the UM VM part
constant across native fluid EXT slots, so it also applies exactly the last
completed-step VM load. Other forces retain their existing time treatment:
UG drag is an implicit reaction with lagged UL, UM drag stays explicit,
and native fluid convection/other source extrapolation is retained.

The local coupled-inertia inversion is bypassed. Its transformed coefficients
are replaced by the uneliminated phase-equation coefficients: gas pressure
response `1/rhoGas` and mixture mobility
`alpha/rhoGas+(1-alpha)/rhoLiquid`. The gas pressure corrector and pressure
acceleration monitors use the same ordinary gas density. These coefficient
changes are required by moving all VM inertia back to the RHS; phase force
models and material properties are unchanged.

The full physical phase stresses and native-baseline stress subtraction that
were automatically active in coupled mode remain active in explicit mode,
even with `mixtureViscousCorrectionEnabled=0`. The existing outlet enhancement
is retained. This prevents the VM switch from silently removing the mixture
viscous correction. No VM acceleration is recomputed with a partially solved
current-step velocity.

`explicitVirtualMassEnabled` defaults to 0 for compatibility and cannot be
combined with `coupledVirtualMassEnabled=1`. It requires both phase/mixture VM
and user mixture forcing to remain paired. With `virtualMassExtrapolationEnabled=0`, restore the older gas
semi-implicit / UM explicit path by setting both other mode switches to 0;
restore coupled inertia by setting explicit=0 and coupled=1.

Host verification covers actual explicit VM kernels, paired coefficients,
ordinary gas BDF inertia/drag, non-VM coefficient invariance, UM EXT1/2/3
load handling, padding, pressure corrector equivalence with both elliptic APIs,
legacy modes, and the actual P5/cubature10 conservative transport kernels.
Native MPI/GPU compilation and long-time spatial stability require a target run.

### Previous test: analytically coupled VM inertia, gas pressure corrector retained

The previous coupled test enables `gasPressureCorrectorEnabled=1`, uses a
computed strong pressure gradient in both the predictor and corrector
(`gasPressureGradientMethod=0`, `gasPressureCorrectorGradientMethod=0`), and
retains solved UG, smooth initial/inlet profile and mixture forces. It selects
`coupledVirtualMassEnabled=1`, `virtualMassEnabled=1`,
`mixtureVirtualMassEnabled=1`, and `virtualMassCoefficient=0.5`.
The startup message must contain `COUPLED VM`. No delay/ramp is added.
Use a fresh start: checkpoints do not store an algorithm-mode identity.

At each point let `a=alpha`, `b=1-a`, `k=Cvm*rhoLiquid*ramp`. The material
accelerations satisfy the local phase-volume inertia block

```
[ a*(rhoGas+k)    -a*k             ] [ ag ] = [ Fg ]
[ -a*k           b*rhoLiquid+a*k   ] [ al ]   [ Fl ]
```

Here Fg/Fl are the respective non-VM phase forces per mixture volume, including
pressure, gravity, drag and physical stresses. Invert this block before native
time discretization, rather than evaluating either VM acceleration from old
velocity differences. Define

```
den = b*rhoGas*rhoLiquid + k*(b*rhoLiquid+a*rhoGas)
Ggg = (b*rhoLiquid+a*k)/den
Ggl = b*k/den
Bg  = (b*rhoLiquid+k)/den
Bl  = (b*rhoGas+k)/den
chi = a*Bg+b*Bl
```

For unweighted phase forces fg=Fg/a and fl=Fl/b, `ag=Ggg*fg+Ggl*fl`.
The gas pressure acceleration is `-Bg*gradP`; volume-mixture pressure
acceleration is `-chi*gradP`. The property callback therefore supplies
`rhoPressure=1/chi` to the native UM pressure/velocity solve. Gas implicit drag
is `lambda=Ki*b*rhoLiquid/den` and the mixture drag acceleration is
`a*(1-rhoGas/rhoLiquid)*lambda*(UL-UG)`. Both are derived from the same block.
Gravity remains g in both equations. The original physical alpha is used in
phase-volume weights; the pre-existing `alphaFloor` protection is applied to
vanishing phase denominators (including b in the compact inverse).

UG keeps native BDF histories and explicit material convection. UM keeps its
native BDF, pressure solve, convection and kinematic drift divergence. The
complete VM material-inertia block is eliminated, so there is no separate
lagged time-acceleration or VM convective-acceleration source to add again.
VM history collection is skipped in this mode. Native advection/time splitting
and alpha transport/regularization are retained; this is not a monolithic
implicit solve of convection, viscosity and interphase drag.

Physical stress forces are transformed consistently: UG receives
`Ggg*div(alpha*tauG)/alpha + Ggl*div(beta*tauL)/beta`; UM receives
`Bg*div(alpha*tauG)+Bl*div(beta*tauL)`. Coefficients multiply the stress
divergences after differentiation. This automatically replaces the optional
legacy mixture viscous-correction path. The native baseline implicit viscosity
is retained as a stabilization operator and its matching lagged baseline
stress is subtracted explicitly. The outlet viscosity enhancement and scalar
numerical diffusion/filters remain. The stress treatment is therefore a
deferred explicit correction, not a fully implicit two-phase stress solve.

The pressure corrector snapshots the predictor's pointwise Bg, in addition to
its diffusion and reaction coefficients. It uses that same snapshot for both
old and new pressure acceleration; property refreshes cannot change its operator.

This mode requires paired mixture forces/VM, computed shared pressure without
gas-only masks or temporal filters, solved unclipped velocities, drift stress,
and no independent mixture-drag ramp. Startup rejects incompatible settings.
It defaults to 0 when absent. To recover the previous UG-only VM diagnostic,
set both VM mode switches to 0 and `mixtureVirtualMassEnabled=0`; to recover
the old paired explicit-UM VM treatment, set coupled=0 and mixture VM=1.

Validation compiles the actual kernels and compares their gas and mixture
accelerations with an independent phase-block solve over multiple alpha,
density and VM values. It checks phase stress transfer, drag, pressure
mobility, positive inertia, BDF1/BDF2 consistency, zero-VM/pure-phase limits,
padding and a 1000-step force-free impulse. The uniform VM impulse no longer
self-amplifies, and a 5000-step homogeneous slip test relaxes with the retained
gas/mixture drag treatment at dt=1e-5. Corrector tests exercise spatially varying Bg snapshots and
both elliptic APIs. Full spatial MPI/GPU stability is not yet verified.

### Optional gas pressure corrector

Set `gasPressureCorrectorEnabled=0` to recover the original one-through
algorithm with the selected predictor pressure mode. The code default is 0
when the new key is absent. For the previous hydrostatic-only test, also set
`gasPressureGradientMethod=2` and leave the corrector off.

The source callback saves the actual predictor pressure acceleration, including
any hydrostatic replacement, temporal filtering, inlet masks and VM inertia.
The implicit gas-drag callback snapshots the predictor's three diffusion
fields (including AVM/outlet ramp) and `g0/dt*transportCoeff+lambdaDrag` at the
actual scalar solve, before post-scalar properties can overwrite live buffers.
After the native pressure/UM solve, `UDF_ExecuteStep` solves, componentwise,

`A_g deltaUG = M_local * (aPressure_new - aPressure_used)`

with the saved coefficients, then adds deltaUG to the predictor. New pressure
uses the current native `o_P` with a strong (0) or weak (1) gradient selected
independently of the predictor. Spatial inlet weights apply to the new load;
temporal pressure filtering does not. The scalar elliptic solver performs
gather-scatter internally, so the element-local RHS mass is applied only once.
The same Dirichlet mask and zero correction initial guess preserve specified
gas boundary values; Neumann correction loads are zero. Robin boundaries are
not supported by this corrector and are rejected when it is enabled.

This adds three scalar Helmholtz solves per physical step. Native predictor
solver statistics are restored after reuse; separate `UG pressure corrector`
log lines report correction iterations, residuals and maxDelta at the monitor
interval (or every step in verbose mode). A startup `UG pressure corrector ON`
line confirms activation. Snapshot step checks prevent stale loads, and repeat
execute callbacks cannot apply the correction twice. Corrected UG/alpha*UG/UL
are refreshed before VM history, diagnostics and checkpoint output. Alpha,
UM, pressure, BDF history, scalar nonlinear source history and fluid properties
are not advanced or re-solved by this correction. Thus same-step alpha flux
and mixture-force consistency still require a future coupled iteration.

The enabled mode requires solved UG and gas clipping/low-alpha masks off, so
postprocessing cannot invalidate the predictor equation before correction.
With a hydrostatic predictor, enabling this corrector restores the computed
pressure contribution; the final UG is no longer hydrostatic-only. Validation
compiles the actual capture/correction helpers against a small diffusion/drag
operator and compares with a direct new-pressure solve, including boundary
values, nonuniform mass, post-predictor coefficient mutations and disabled mode.
This is host algebra validation, not a full native MPI/GPU stability test.

### Previous test: full mixture forces, prescribed hydrostatic UG pressure

The hydrostatic-only test selects both VM mode switches 0, `frozenGasVelocityEnabled=0`,
`smoothGasVelocityProfileEnabled=1`, `mixtureForcingEnabled=1`, and
`virtualMassEnabled=0`. Start a fresh run. All three UG scalars are solved
with gravity, implicit drag, convection and viscosity active. User mixture
forces (gravity, drag and drift stress) are restored. The optional mixture
viscous correction retains its existing switch; VM remains off by prior choice.
Alpha diffusion, filters, clipping and conservative transport are unchanged.

`gasPressureGradientMethod=2` replaces only the gas pressure source by the
constant physical gradient `gradP=rhoRef*gravity`. `gasHydrostaticDensity=0`
uses `rhoLiquid`; a positive value supplies a fixed reference density.
For gravity `(0,0,-9.81)` and rhoLiquid=998.2, `gradPz=-9792.342 Pa/m`.
The gas acceleration contribution is `-gradP/rhoGas+gravity`, hence upward
buoyancy. Local pressure and alpha fluctuations do not change this prescribed
gradient. Gas inlet masks/region and temporal gradient filters are bypassed
in this mode. `gasPressureEnabled` still controls application of the source.
The native UM pressure solve, coefficients and forcing are unchanged by the
gradient selector. This is a pressure-coupling diagnostic, not a replacement
for the shared pressure in the final coupled model. Methods 0/1 restore the
computed strong/weak gradient (default 0 when the key is absent).

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
profile=1, user mixture forcing=1` for this test. The hydrostatic startup line
also prints rhoRef and all three imposed gradient components. This is a diagnostic, not a fully coupled
physical solution. Set the master switch to 1 to restore user mixture forces;
the existing individual switches then apply. Its default when absent is 1.

### Frozen UG isolation test

To reproduce the successful frozen test, set both VM mode switches to 0,
`frozenGasVelocityEnabled=1`
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
- Alpha/UG convection uses separate operator histories with native fluid EXT.
  Other scalar sources (except the selected explicit liquid VM) and native-advection cancellation retain EXT1 through
  separate scalar coefficient storage. Native BDF history/order is unchanged.
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

With both `coupledVirtualMassEnabled=0` and `explicitVirtualMassEnabled=0`,
the legacy formulation below applies.
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

With `mixtureVirtualMassEnabled=1`, UM retains the explicit physical volume-mixture source

`alpha*(1-rhoGas/rhoLiquid)*(c/rhoGas)*(aL_lagged-aG_lagged)`.

Do not divide this UM source by rhoEff. The diagnostic switch
`mixtureVirtualMassEnabled=0` omits it deliberately while retaining UG VM;
this is not the fully paired physical model. Its liquid acceleration comes from the same completed-step history
field used in UG. The gas history and liquid history use the same selected
first/second-order variable-dt derivative and current material convection.
`virtualMassTimeDerivativeOrder=1` matches method1's liquid-history derivative;
2 remains available, with first-order startup until enough samples exist.
The implicit gas derivative order is controlled by native `timeStepper`.

`virtualMassStartStep=S` leaves VM off through S;
`virtualMassRampStep=R>0` gives `min(1,(step-S)/R)` afterward, and R=0 activates
immediately. History is collected during the delay when VM is enabled.
The default is VM on, coefficient 0.5, no delay/ramp, matching method1's
activation for the legacy mode. When enabled in that mode, UM VM
remains explicit with native fluid EXT, while non-convective scalar explicit
parts use EXT1. Scalar convection follows the fluid EXT coefficients as above. Physical coefficients are paired; the coupled equations are
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
