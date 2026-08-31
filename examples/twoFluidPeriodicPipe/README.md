# Minimal Eulerian--Eulerian flow in a periodic pipe

This case exercises the experimental two-fluid solver in a round pipe that is
periodic in the axial (`z`) direction and no-slip at the radial wall. It reuses
the mesh from `examples/turbPipePeriodic` through the `[MESH]` entry in the
parameter file; no mesh copy or generation step is required.

The model is intentionally limited to the first implementation increment:

- conservative gas-volume-fraction transport using the gas phase flux;
- optional conservative numerical diffusion controlled by `alphaDiffusivity`;
- pointwise `alpha_l = 1 - alpha_g`;
- conservative bound correction for `0 <= alpha_g <= 1`;
- constant liquid and gas properties;
- separate liquid and gas velocities;
- one shared mixture-continuity pressure;
- Schiller--Naumann drag with diagonal-implicit coupling;
- a common axial body acceleration.

Turbulence, dynamic void fraction, lift, wall lubrication, virtual mass, phase
change, constant-flow-rate control, and advection subcycling are not enabled.

Run from this directory:

```bash
nekrs --setup twoFluidPipe.par
```

The liquid starts with a parabolic axial profile with centerline velocity
`0.20`, while the gas starts at `0.30`. Every ten steps the UDF prints the
phase bulk velocities and bulk slip. The continuity diagnostics report the gas,
liquid, and mixture mass residuals separately. Short fixed-point iterations
update `alpha_g`, momentum, and shared pressure without advancing physical time
more than once per step.

Every coupling iteration prints `max|deltaAlphaG|` and the relative `L2` norm
of the gas-volume-fraction fixed-point update. The separately reported `Rg`,
`Rl`, and `Rm` values remain the final phase-continuity equation residuals.

This example sets `alphaDiffusivity = 1.0e-6`. Set it to zero to recover pure
advection. The gas diffusive flux is `-D_alpha grad(alpha_g)` and the liquid
flux receives its exact opposite, so numerical diffusion does not enter the
mixture-volume flux or change total mixture volume.

Checkpoint files contain the standard liquid velocity and pressure fields. For
this case, which has no transported scalars, the legacy Nek scalar slots are:

- `scalar00`--`scalar02`: gas velocity;
- `scalar03`: gas volume fraction;
- `scalar04`--`scalar06`: slip velocity `u_g - u_l`;
- `scalar07`--`scalar09`: mixture velocity `alpha_l u_l + alpha_g u_g`;
- `scalar10`: pointwise mixture mass-continuity residual;
- `scalar11`: interphase drag coefficient `K`;
- `scalar12`--`scalar14`: liquid-directed interphase force `K(u_g-u_l)`.

When transported scalars are enabled, these diagnostics follow their existing
Nek scalar slots. ADIOS output retains the diagnostic field names directly.

The mesh is a radius-`0.5`, length-`6` pipe. The radial wall uses
`zeroDirichlet` for both phase velocities; the two axial faces are periodic.
