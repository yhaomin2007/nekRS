# Method 1: UG, alpha, then native pressure/UM

The case transports four scalars, stored as `ALPHA, UGX, UGY, UGZ`.
The solve order is **UG -> ALPHA(new UG) -> native pressure -> native UM**.
There is no independent liquid momentum solve or end-step phase correction.
No NekRS core files are changed.

## Initial condition and inlet ramp

The supplied par defaults to uniform gas velocity with zero initial alpha:

```ini
[CASEDATA]
startupMode = 2
gasInitialVelocity = 0.3
inletRampStartTime = 0.0
alphaInletRampDuration = 0.1
gasInletRampDuration = 0.1
alphaInlet = 0.016667
gasInletVelocity = 0.3
```

`startupMode=2` initializes alpha and UM to zero and UG to
(0,0,gasInitialVelocity). The inlet UGz is constant at gasInletVelocity,
while inlet alpha alone ramps from zero to alphaInlet. Reconstructed UL is
initially zero. The existing zero-velocity wall BCs remain in effect, so the
uniform gas field applies to the unconstrained domain interior. UG then
evolves under the momentum equation; this is not a frozen-velocity test.
Set gasInitialVelocity equal to gasInletVelocity to match their initial values.
If omitted, gasInitialVelocity defaults to gasInletVelocity.

`startupMode=1` initializes alpha, all UG components and UM to exactly zero;
reconstructed UL is zero as well. At the inlet, alpha and vertical gas
velocity use independent quintic smoothstep ramps to the specified final
values. All times are physical seconds. For either duration T,

\[
s=\operatorname{clamp}((t-t_{start})/T,0,1),\qquad
r(s)=6s^5-15s^4+10s^3.
\]

Thus alpha_in(t)=alphaInlet*r_alpha(t) and UGz_in(t)=gasInletVelocity*r_g(t),
with the existing wall-precedence mask applied to gas velocity. Each ramp
has zero first and second derivatives at its endpoints. The initial 0.1 s
values are diagnostic settings, not a calibrated physical injection model.

The inlet liquid remains stationary. Native UM uses the same instantaneous
alpha and UG through UMz_in=alpha_in*rhoG*UGz_in/rhoM_in, including the same
velocity mask. In mode 2 the gas flux scales as r_alpha(t); gas ramp duration
is ignored. In mode 1 with equal durations it scales as r(t)^2, because both
alpha and UG are ramped.

Set `startupMode=0` to restore the original tanh plume and constant inlet.
`initialPlumeHeight` and `initialPlumeThickness` retain their original meanings;
the three ramp-time parameters are ignored in this mode. Omitting startupMode
also retains the original mode for older par files.

Restart preserves the checkpoint fields and uses absolute simulation time
for the inlet ramp; it does not zero the solution or restart the ramp clock.
Use a fresh run to test either new startup mode. UG still evolves everywhere
under the existing gas momentum equation even in regions with alpha=0.

## Liquid velocity and gas momentum

Liquid velocity is reconstructed using the density-averaged mixture identity:

\[
\mathbf u_l=\frac{\rho_m\mathbf u_m-\alpha\rho_g\mathbf u_g}
{(1-\alpha)\rho_l},\qquad
\rho_m=(1-\alpha)\rho_l+\alpha\rho_g.
\]

The existing denominator floor protects the vanishing-liquid limit. No UL
scalar fields, UL boundary sections, liquid Helmholtz solves or symmetric
phase-inertia inversion remain. In plume mode the fresh-start helper initializes UM
consistently with the gas field and a stationary liquid.

The gas-only equation keeps explicit Schiller–Naumann drag and physical gas
stress, and semi-implicit gas virtual-mass acceleration. Its liquid material
acceleration is evaluated from successive completed reconstructed UL fields
and UL.grad(UL), then extrapolated using the native fluid EXT weights. This
history is updated once after the completed native pressure/UM solve.
At startup/restart the reconstructed liquid velocity is the history baseline
and the unavailable liquid acceleration is initialized to zero.
The gas pressure gradient remains the latest assembled strong gradient,
with `gasPressureGradientExtrapolation=1` selecting its time extrapolation.

## Alpha scheduling and pressure

The first native scalar sweep solves only UG. In `postScalar`, the UDF builds
alpha transport from the just-solved UG and completed alpha, then solves only
alpha using the native scalar Helmholtz solver. Alpha is advanced exactly
once. Its BDF history load and native HPFRT contribution are saved before
native history lagging; no second lag or forcing-history update occurs.
This split does not establish second-order coupled accuracy despite BDF2.

The pressure divergence target is computed explicitly after alpha:

\[
(\nabla\cdot\mathbf u_m)_{\rm target}=
\frac{\rho_l-\rho_g}{\rho_m^{n+1}}
\left(D_t^{\rm BDF}\alpha^{n+1}
+\mathbf u_m^{\rm EXT}\cdot\nabla\alpha^{n+1}\right).
\]

UM_EXT is copied from native `fluid->o_Ue` at source time, after native velocity
extrapolation and before boundary updates/pressure. It uses native startup,
order and variable-step coefficients. The material alpha advection is evaluated
directly at cubature points and projected/assembled. No extrapolation of the
resulting divergence is applied. The weighted alpha history is captured
before native lagging, so BDF2 never reads a discarded oldest history slot.

This target uses the actual postprocessed alpha change, including its HPFRT,
clipping and boundary effects. Alpha diffusion is already present in that
change and is not added again. It replaces the previous relative-flux target.
Even when alphaCompressionEnabled=0, this still enforces the mixture mass
identity for the modified alpha equation; phase gas-volume conservation is
not thereby restored. No unknown current UM appears on the RHS.

Native UM retains its momentum equation and properties and is never reset.
Alpha retains native diffusion, HPFRT and clipping. Custom stabilization
remains absent. The explicit UM extrapolation is a temporal split; this does
not guarantee removal of stripes or exact nonlinear final-UM continuity.

## Parameters and output

The par file contains physical/input parameters and native solver controls.
UL sections have been removed. Prefer a fresh run for comparison with the
previous seven-scalar implementation.

At the configured diagnostic interval, logs show:
`UG_BEGIN -> UG_DONE -> ALPHA_BEGIN_NEW_UG -> ALPHA_DONE ->
DIV_UM_EXPLICIT_UM_EXT -> NATIVE_PRESSURE_UM_BEGIN -> NATIVE_PRESSURE_UM_DONE`.

The single auxiliary checkpoint `diag` is unchanged:

| Field | Meaning |
| --- | --- |
| s1 | Target div(UM), copied immediately before pressure/UM |
| s2 | Actual assembled strong div(UM) after the solve |
| velocity vector | Latest strong pressure gradient |

The strong-divergence comparison differs from the native weak continuity
residual and pressure linear-solver residual.

## Verification

Run tests `inlet_startup.py`, `plume_initialization.py`, `phase_momentum.py`, `phase_alpha_order.py`, `ug_virtual_mass.py`,
`ug_liquid_acceleration_ext.py`, `explicit_divergence.py`,
`cubature_transport.py` and `scalar_time_treatment.py` under `tests/`.
These check reconstruction kernels, deferred-alpha host scheduling, gas VM
and drag, liquid-acceleration extrapolation, and native forcing algebra.
A full GPU/MPI run with the original mesh is still needed to assess stripes.
METHOD1_AUDIT.md describes an older implementation.
