# Minimal Eulerian--Eulerian flow in a periodic pipe

This case exercises the experimental two-fluid solver in a round pipe that is
periodic in the axial (`z`) direction and no-slip at the radial wall. It reuses
the mesh from `examples/turbPipePeriodic` through the `[MESH]` entry in the
parameter file; no mesh copy or generation step is required.

The model is intentionally limited to the first implementation increment:

- prescribed uniform gas volume fraction;
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
phase bulk velocities and bulk slip, followed by the mixture-divergence norm.
The drag term should reduce the phase slip while the shared pressure projection
controls `div(alpha_l u_l + alpha_g u_g)`.

The mesh is a radius-`0.5`, length-`6` pipe. The radial wall uses
`zeroDirichlet` for both phase velocities; the two axial faces are periodic.
