# singleFluidPeriodicNative

This is a control case for the two-fluid pressure-coupling work.  It uses the
**unmodified native NekRS single-fluid pressure and velocity solvers** on the
same fully periodic 0.1 m x 0.1 m x 0.1 m box.

The initial field is a longitudinal/irrotational mode,

\[
u_x=A\sin\left(2\pi(x-x_0)/L_x\right),\qquad
u_y=u_z=0,
\]

with

\[
A=0.20\;\mathrm{m/s},\quad x_0=-0.05\;\mathrm{m},\quad L_x=0.10\;\mathrm{m}.
\]

The default fluid properties match the gas phase used in
`twoFluidPeriodic`:

\[
\rho=1.20\;\mathrm{kg/m^3},\qquad
\mu=1.8\times10^{-5}\;\mathrm{Pa\,s}.
\]

The purpose is to measure the divergence left by the **native NekRS**
pressure-before-velocity-Helmholtz splitting, without any custom phase
pressure algorithm.

The UDF only:

1. initializes the velocity and pressure;
2. computes a strong-divergence diagnostic after each native NekRS timestep.

It does **not** override `preFluid`, pressure solve, velocity solve, forcing,
or momentum equations.

The log prints

```
singleFluidPeriodicNative initial ...
singleFluidPeriodicNative step=... divRMS=... uRMS=... pRMS=...
```

Compare `divRMS` directly with the approximately 2e-4 plateau observed in
the two-fluid native-style prototype.

## Mesh

Use the exact same periodic mesh as `twoFluidPeriodic`.  In the run
directory, copy or link

```
twoFluidPeriodic.re2
```

as

```
singleFluidPeriodicNative.re2
```

before running.

A second useful run is obtained by changing the native fluid properties to the
liquid values:

```ini
rho = 998.20
viscosity = 1.00e-3
```

No other code changes are required.
