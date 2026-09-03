# OpenFOAM bubble-column cross-code benchmark

Reference case: OpenFOAM Foundation 5.x
`tutorials/multiphase/twoPhaseEulerFoam/LES/bubbleColumn`.

The OpenFOAM block mesh is a rectangular 0.15 m x 1.0 m x 0.10 m domain with 25 x 75 x 1 finite-volume cells. The bottom is `inlet`, the top is `outlet`, and the two side boundaries are no-slip `walls`; the front/back are the unpatched/default faces of the one-cell-thick tutorial mesh.

OpenFOAM phase setup:
- phases: air and water;
- air bubble reference diameter d0 = 3e-3 m;
- surface tension = 0.07 N/m;
- drag: Schiller-Naumann for dispersed configurations plus the tutorial's segregated drag closure;
- virtual-mass coefficient Cvm = 0.5;
- lift, wall lubrication and turbulent dispersion are empty in this tutorial;
- air viscosity = 1.84e-5 Pa s and perfect-gas EOS;
- water reference density rho0 = 1027 kg/m^3 and viscosity = 3.645e-4 Pa s.

OpenFOAM initial/boundary values relevant to the hydraulic comparison:
- U_air initial/inlet = (0, 0.1, 0) m/s in OpenFOAM coordinates;
- U_water initial/inlet = (0, 0, 0) m/s;
- alpha_air initial = 0 and inlet = 0.5;
- outlet uses pressure-inlet-outlet velocity;
- walls are no-slip for both velocities.

## nekRS benchmark stages

### Stage A0: pressure-operator verification

Use the same physical box and inlet/outlet topology, but freeze alpha_g to a constant and set K=0, gravity=0. Use identical properties for both phases. This is not intended to reproduce the physical bubble column; it is the clean discrete pressure-coupling test on the OpenFOAM geometry.

Acceptance:
- I7 pressure operator/correction relative L2 error near roundoff/discrete tolerance;
- alpha_g U_g + alpha_l U_l satisfies the common-pressure continuity constraint;
- U_g and U_l stay equal when initialized/equipped identically;
- the solution collapses to the native single-fluid nekRS result.

Run `bubbleColumnA0.par` for the equal-phase two-fluid result and
`bubbleColumnA0SingleFluid.par` for the native NekRS reference. Both cases use
the same mesh, time step, initial condition, velocity boundary conditions,
density and viscosity. Compare the reported `UzL` and `p` ranges and means,
then compare the checkpoint velocity and pressure fields directly. Equality
of the two phase velocities alone is not an acceptance criterion.

Generate the shared mesh once and run both cases from this directory:

```bash
genbox <<<'input.box'
nekrs --setup bubbleColumnA0SingleFluid.par
nekrs --setup bubbleColumnA0.par
```

The first comparison is deliberately limited to Stage A0. Do not enable drag,
gravity, unequal properties or alpha transport until the two runs produce the
same liquid velocity and pressure fields to solver tolerance.

### Stage A1: inlet/outlet kinematic test

Keep K=0 and frozen alpha, but use the OpenFOAM inlet topology. This tests the common-pressure boundary handling without adding interfacial physics.

### Stage B: OpenFOAM drag benchmark

After Stage A passes, turn on air/water properties, d_b = 3 mm and Schiller-Naumann drag using the Option-A lagged-drag pressure predictor plus exact local partial elimination. Initially keep alpha frozen to isolate momentum/pressure/drag coupling.

### Stage C: full tutorial comparison

Enable conservative alpha transport, gravity and then virtual mass. Match the OpenFOAM inlet alpha_air=0.5 and air inlet speed 0.1 m/s. Compare time histories and profiles of alpha_air, U_air, U_water, pressure and mixture continuity.

## Coordinate mapping

The OpenFOAM tutorial uses its second coordinate as the 1.0-m vertical direction. The nekRS case uses z as vertical, so map OpenFOAM (x,y,z) -> nekRS (x,z,y) for easier use of standard nekRS inlet/outlet conventions. Physical dimensions remain 0.15 x 0.10 x 1.0 m.

## Comparison quantities

Always report:
- ||D(alpha_g U_g + alpha_l U_l)||_2 and max norm;
- ||alpha_g phi_g + alpha_l phi_l - phi_mix||;
- phase mean vertical velocities;
- pressure min/max and centerline profile;
- alpha min/max/mean when alpha transport is enabled;
- relative L2 differences between the nekRS and OpenFOAM centerline/profile exports after mapping coordinates.
