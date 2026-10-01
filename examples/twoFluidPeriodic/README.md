# Two-fluid periodic-box pressure-projection prototype

This case is a case-layer prototype for testing two independent phase
velocities coupled to one common pressure field without modifying NekRS core
code.

It intentionally contains no drag, viscosity, convection, lift, virtual mass,
wall lubrication, phase change, or alpha transport. The first goal is only to
prove that a common pressure can project two phase velocities so that the
volumetric mixture velocity satisfies continuity.

## Mesh required

Create a mesh named:

`twoFluidPeriodic.re2`

The first test should be a rectangular box that is fully periodic in x, y, and
z. The default parameters assume x and y both span [0,1]. Update `xOrigin`,
`yOrigin`, `xLength`, and `yLength` if your mesh is different.

There should be no physical boundary faces in this first prototype.
Periodicity must be encoded in the Nek mesh connectivity. Therefore the par
file uses:

```ini
[FLUID VELOCITY]
boundaryTypeMap = none
```

## Algorithm

The phase predictors are initialized as

[
u_{g,H}=(A_gsin(2pi(x-x_0)/L_x),0,0),
]

[
u_{l,H}=(0,A_lsin(2pi(y-y_0)/L_y),0).
]

For constant gas volume fraction alpha,

[
U_H=alpha u_{g,H}+(1-alpha)u_{l,H}.
]

Define

[
D_g=rac{Delta t}{gamma_0ho_g},
qquad
D_l=rac{Delta t}{gamma_0ho_l},
]

and

[
lambda_p=alpha D_g+(1-alpha)D_l.
]

The prototype reuses the already-created native NekRS pressure elliptic solver,
but supplies its own two-fluid pressure coefficient and RHS:

[

ablacdot(lambda_p
abla p)=
ablacdot U_H.
]

Then both phases are corrected with the same pressure:

[
u_g=u_{g,H}-D_g
abla p,
]

[
u_l=u_{l,H}-D_l
abla p.
]

The acceptance criterion is

[

ablacdot(alpha u_g+(1-alpha)u_l)ightarrow 0.
]

## Printed diagnostics

Each step prints a line beginning with `twoFluidPeriodic` containing:

- `divPreRMS`: RMS mixture divergence before projection,
- `divPostRMS`: RMS mixture divergence after projection,
- `divRatio`: post/pre divergence ratio,
- `pExactRMSE`: error relative to the analytic zero-mean pressure solution,
- pressure iteration count and final residual.

## Checkpoint files

At checkpoint steps the case writes:

- `ug`: corrected gas velocity,
- `ul`: corrected liquid velocity,
- `upred`: mixture predictor,
- `ucorr`: corrected mixture velocity,
- `gradp`: pressure gradient,
- `tfdiag`:
  - scalar00 = pre-projection mixture divergence,
  - scalar01 = post-projection mixture divergence,
  - scalar02 = analytic pressure,
  - scalar03 = numerical pressure minus analytic pressure.

The native NekRS U field is overwritten with the corrected volumetric mixture
velocity only for visualization/checkpoint convenience.

## Scope

This first version proves only the pressure projection. The phase predictors are
not yet full phase momentum/Helmholtz solves. Once this works, the next step is
to replace the analytic predictors with two actual phase momentum solves while
keeping the same common-pressure projection.
