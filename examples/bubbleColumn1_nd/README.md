# Method 1 nondimensional bubble column

This case uses the same physical inputs and one-pass solve sequence as
[`bubbleColumn`](../bubbleColumn), with a separate case name and nondimensional
fields. The reference scales are fixed at the values below.

| Scale | Value |
| --- | --- |
| Length, L0 | 0.1 m, reference column width |
| Velocity, U0 | 0.3 m/s, inlet gas velocity |
| Density, rho0 | 998.2 kg/m³, liquid water |
| Water viscosity, mu_water | 0.001 Pa·s, giving Re = rho0 U0 L0 / mu_water = 29,946 |
| Time, T0 = L0/U0 | 1/3 s |
| Pressure and stress, P0 = rho0 U0² | 89.838 Pa |
| Acceleration, A0 = U0²/L0 | 0.9 m/s² |

| Quantity | Conversion to nondimensional | Default in this case |
| --- | --- | --- |
| Coordinates and lengths | Divide by L0 | Mesh coordinates ×10, bubble diameter 0.03 |
| Gas and mixture velocity | Divide by U0 | UG inlet 1, density-weighted UM inlet ≈ 2.04e-5 away from the rim |
| Phase density | Divide by rho0 | Liquid 1, gas 0.001202163895 |
| Dynamic viscosity | Divide by rho0 U0 L0 | Liquid 1/Re, gas 6.010819475e-7 |
| Scalar diffusion | Divide by U0 L0 | 3.333333333e-4 for alpha and each UG component |
| Gravity | Divide by A0 | z component −10.9 |
| Time step | Divide by T0 | 3e-5, equivalent to 1e-5 s |
| HPFRT rate | Multiply by T0 | 3.333333333, equivalent to 10 s⁻¹ |

Alpha, filter attenuation, VM coefficient, extrapolation weights and all step
counts are dimensionless. The drag law evaluates the same bubble Reynolds
number. The pressure outlet law uses the converted velocity switch and
pressure scale. `residualTol` values match the baseline's absolute stopping
thresholds after rescaling the assembled weak residual and mesh volume.
Relative to the dimensional `residualTol`, the factors are `sqrt(L0)/U0²`
for pressure, `1/(sqrt(L0)*rho0*U0²)` for UM, `1/(sqrt(L0)*U0)` for alpha,
and `1/(sqrt(L0)*U0²)` for UG. This follows the NekRS threshold
`residualTol * sqrt(mesh volume)` for the absolute residual of the weak system.

Run from `examples/bubbleColumn1_nd` with the **same metre-based mesh** used
for the dimensional comparison case. For the default layout:

```sh
ln -s ../bubbleColumn/bubbleColumn.re2 bubbleColumn.re2
mpiexec -n <ranks> nekrs --setup bubbleColumn1_nd
```

Use your actual mesh path if it differs from the example. The `usrdat2` hook
scales the full curved GLL coordinates before metric construction. Connectivity,
boundary IDs and aspect ratio come from the input mesh. A physical width of
0.1 m becomes 1. Use fresh starts for the dimensional versus nondimensional
comparison, or a checkpoint already produced in this case's units.

The startup log lists the reference scales. Checkpoint coordinates, time,
velocity, pressure and UG diagnostic terms are nondimensional. To compare
with the baseline, multiply them respectively by L0, T0, U0, P0 and A0.
Multiply pressure-gradient exports by P0/L0, and divergence and `gcf` scalar01
(drag rate) by U0/L0. `gcf` scalar00 (alpha) and the free-row mask are unchanged
by scaling. Multiply `gcf` scalar02 (effective density) by rho0. CFL and
`maxDragStep` can be compared directly. Compare fields at the same physical
time to measure changes from numerical scaling.
