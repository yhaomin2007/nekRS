# Experimental Eulerian--Eulerian solver: minimal branch

## Implemented model

The branch advances two velocity fields on the same SEM mesh and one shared
mechanical pressure.  The prescribed uniform volume fractions satisfy
`alpha_l + alpha_g = 1`.  Phase properties are constant.  The pressure equation
uses the mixture-volume constraint

`div(alpha_l u_l + alpha_g u_g) = 0`.

Interphase drag is Schiller--Naumann.  Its same-phase contribution is placed on
each velocity Helmholtz diagonal; the opposite-phase velocity is lagged across
the normal NekRS corrector iteration.  Momentum transfer is equal and opposite
before division by phase volume and density.

## Parameter file

Supply the native `TWO FLUID` section:

```ini
[TWO FLUID]
gasVolumeFraction = 0.1
bubbleDiameter = 1e-3
gasDensity = 1.0
gasViscosity = 1.8e-5
gravityX = 0
gravityY = 0
gravityZ = -9.81
alphaFloor = 1e-8
```

Liquid density and viscosity remain in `[FLUID VELOCITY]`. The gas momentum
solver inherits the liquid velocity solver, preconditioner, tolerance, and
boundary map settings.

## UDF access

The liquid and gas velocities are available as:

```cpp
nrs->fluid->o_U
nrs->gas->o_U
```

The shared pressure is `nrs->fluid->o_P`. The prescribed volume-fraction fields
are `nrs->twoFluid->o_alphaG` and `o_alphaL`.

## Deliberate restrictions

This is a verification branch, not yet the full production solver. It currently
requires a fixed mesh and is intended for periodic cases. It does not yet
include surface terms in the mixture pressure RHS, independent gas boundary
callbacks, dynamic volume fraction, lift, wall lubrication, turbulent
dispersion, virtual mass, phase change, or turbulence. Pressure rho-splitting,
low-Mach mode, NekNek, constant-flow-rate control, and subcycling are outside
the supported v1 combination.

The next required numerical addition is the gas-volume-fraction transport
equation with consistent momentum transport. Boundary pressure fluxes should
be completed before wall-bounded physical validation.
