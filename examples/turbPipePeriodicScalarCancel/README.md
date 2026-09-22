# Periodic pipe scalar-advection cancellation test

This case reuses the mesh from `../turbPipePeriodic` and adds one passive
scalar. It is a minimal test of the scalar-advection cancellation currently
used by the `bubbleColumn` case.

NekRS normally advances the passive scalar as

```text
ds/dt + A_native(s) = S_user.
```

The user source reconstructs

```text
A_user(s) = u dot strongGrad(s, avg=true)
```

and supplies `S_user=A_user(s)`. The resulting discrete equation is therefore

```text
ds/dt = A_user(s) - A_native(s).
```

The initial scalar is periodic in the axial direction and has zero radial
derivative at the wall. Scalar diffusion and regularization are disabled, so
the field should remain stationary if the two spatial operators cancel
exactly. Scalar advection subcycling is also disabled so both terms enter the
same EXT treatment.

Run from this directory with

```bash
nekrs turbPipeCancel
```

`scalar_advection_cancellation.csv` reports the scalar change from its initial
condition and the L2 defect between the exact native advection kernel and the
assembled user reconstruction. A nonzero defect is expected when native
cubature advection is compared with a GLL-grid `strongGrad`, even though the
two expressions are mathematically identical.
