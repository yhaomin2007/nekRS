# Method 1: UG, alpha, then native pressure/UM

The case transports four scalars, stored as `ALPHA, UGX, UGY, UGZ`.
The solve order is **UG -> ALPHA(new UG) -> native pressure -> native UM**.
There is no independent liquid momentum solve or end-step phase correction.
No NekRS core files are changed.

## Liquid velocity and gas momentum

Liquid velocity is reconstructed using the density-averaged mixture identity:

\[
\mathbf u_l=\frac{\rho_m\mathbf u_m-\alpha\rho_g\mathbf u_g}
{(1-\alpha)\rho_l},\qquad
\rho_m=(1-\alpha)\rho_l+\alpha\rho_g.
\]

The existing denominator floor protects the vanishing-liquid limit. No UL
scalar fields, UL boundary sections, liquid Helmholtz solves or symmetric
phase-inertia inversion remain. The fresh-start plume helper initializes UM
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

The relative-flux pressure divergence target uses the same new-UG transport
as alpha. Native UM retains its own momentum equation, density, mixture
viscosity, gravity and drift-stress forcing. UM is never reset during stepping.
Alpha retains native numerical diffusion, HPFRT and post-solve clipping.
`alphaCompressionEnabled=0` removes alpha*div(UG), changing the conservation
law. The reduced pressure target still omits HPFRT/clipping rates.
Custom alpha stabilization remains absent.

## Parameters and output

The par file contains physical/input parameters and native solver controls.
UL sections have been removed. Prefer a fresh run for comparison with the
previous seven-scalar implementation.

At the configured diagnostic interval, logs show:
`UG_BEGIN -> UG_DONE -> ALPHA_BEGIN_NEW_UG -> ALPHA_DONE ->
NATIVE_PRESSURE_UM_BEGIN -> NATIVE_PRESSURE_UM_DONE`.

The single auxiliary checkpoint `diag` is unchanged:

| Field | Meaning |
| --- | --- |
| s1 | Target div(UM), copied immediately before pressure/UM |
| s2 | Actual assembled strong div(UM) after the solve |
| velocity vector | Latest strong pressure gradient |

The strong-divergence comparison differs from the native weak continuity
residual and pressure linear-solver residual.

## Verification

Run tests `phase_momentum.py`, `phase_alpha_order.py`, `ug_virtual_mass.py`,
`ug_liquid_acceleration_ext.py` and `scalar_time_treatment.py` under `tests/`.
These check reconstruction kernels, deferred-alpha host scheduling, gas VM
and drag, liquid-acceleration extrapolation, and native forcing algebra.
A full GPU/MPI run with the original mesh is still needed to assess stripes.
METHOD1_AUDIT.md describes an older implementation.
