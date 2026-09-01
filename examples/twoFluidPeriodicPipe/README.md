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
- a common axial body acceleration;
- native NekRS constant-flow control applied to the carrier liquid, followed
  by another shared-pressure projection to preserve mixture continuity.

Turbulence, dynamic void fraction, lift, wall lubrication, virtual mass, phase
change, gas-flow-rate control, and advection subcycling are not enabled.

Run from this directory:

```bash
nekrs --setup twoFluidPipe.par
```

The liquid starts with a parabolic axial profile with centerline velocity
`0.20`, corresponding to bulk velocity `0.10`, while the gas starts at `0.30`.
The `[GENERAL] constFlowRate` setting maintains the liquid bulk velocity at
`0.10` in the axial (`Z`) direction. The gas bulk velocity is not prescribed;
it evolves from its momentum equation. Every step the UDF prints the phase bulk
velocities, bulk slip, the drag-free analytical gas bulk velocity
`0.15 + gravityZ*time`, and its error. The final assertion verifies that three
coupling iterations produce only one physical gas momentum advance. The
continuity diagnostics report the gas, liquid, and mixture mass residuals
separately. Short fixed-point iterations update `alpha_g`, momentum, and shared
pressure without advancing physical time more than once per step.
The BDF liquid and gas velocity histories are frozen at the beginning of each
physical step and remain read-only throughout those coupling iterations.

Every coupling iteration prints `max|deltaAlphaG|` and the relative `L2` norm
of the gas-volume-fraction fixed-point update. The separately reported `Rg`,
`Rl`, and `Rm` values remain the final phase-continuity equation residuals.
The mixture-pressure corrector uses the exact matrix-free Schur operator formed
by the same weak gradient, phase response, homogeneous velocity masks, mixture
flux, and weak divergence used in the actual velocity update. The native NekRS
pressure operator is retained as a preconditioner. `operatorConsistency` checks
the matrix-free operator against the applied correction and should be near
roundoff; `kspIters` and `kspResidual` report convergence of the exact Schur
solve.

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

The mesh is a radius-`0.5`, length-`6` pipe. The radial wall uses no-slip
`zeroDirichlet` for the liquid and slip/no-penetration
`zeroDirichletN/zeroNeumann` for the gas; the two axial faces are periodic.
