# Two-fluid periodic-box pressure-coupling prototype

This case is a case-layer prototype for testing two independently solved phase
velocities coupled to one common pressure field without modifying NekRS core
code.

The current version now contains two actual phase Helmholtz momentum predictor
solves plus one common-pressure projection. It still intentionally excludes
convection, drag, gravity, lift, virtual mass, wall lubrication, phase change,
and alpha transport.

## Mesh required

Create a mesh named:

`twoFluidPeriodic.re2`

The current box is

- x in [-0.05, 0.05] m,
- y in [-0.05, 0.05] m,
- z in [0, 0.10] m,

and should be fully periodic in x, y, and z. There should be no physical
boundary faces. `usrdat2` resets all `boundaryID` values to zero.

## Phase momentum predictor

The stored phase velocities are initialized as

[
u_g^0=(A_gsin(2pi(x-x_0)/L_x),0,0),
]

[
u_l^0=(0,A_lsin(2pi(y-y_0)/L_y),0).
]

At every timestep, gas and liquid predictors are now solved independently from

[
left[
rac{ho_ggamma_0}{Delta t}
-
ablacdot(mu_g
abla)
ight]u_{g,H}
=
rac{ho_g}{Delta t}u_g^n,
]

[
left[
rac{ho_lgamma_0}{Delta t}
-
ablacdot(mu_l
abla)
ight]u_{l,H}
=
rac{ho_l}{Delta t}u_l^n.
]

The present case uses `tombo1`, so (gamma_0=1) and this is a
backward-Euler transient-plus-viscous-diffusion predictor.

Two case-layer NekRS `elliptic` solvers named `ug` and `ul` are created
without any core modification. Each scalar Helmholtz solver is reused for the
x, y, and z components of its phase.

## Common pressure projection

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

The prototype reuses the native NekRS pressure elliptic solver but supplies its
own two-fluid coefficient and RHS:

[

ablacdot(lambda_p
abla p)=
ablacdot U_H.
]

The same pressure then corrects both phase velocities:

[
u_g^{n+1}=u_{g,H}-D_g
abla p,
]

[
u_l^{n+1}=u_{l,H}-D_l
abla p.
]

The primary continuity diagnostic is

[

ablacdotleft[alpha u_g^{n+1}+(1-alpha)u_l^{n+1}ight].
]

## What is deliberately not included yet

The momentum predictor currently contains only:

- transient term,
- implicit viscous diffusion.

It does **not** yet contain:

- convection,
- drag,
- gravity,
- lift,
- virtual mass,
- turbulent dispersion,
- wall lubrication.

Those terms should be added only after the two-velocity/one-pressure skeleton
is verified.

## Printed diagnostics

Each step prints a line beginning with `twoFluidPeriodic` containing:

- `divPreRMS`: mixture-divergence RMS after the two phase predictor solves,
- `divPostRMS`: mixture-divergence RMS after common-pressure correction,
- `divRatio`: post/pre divergence ratio,
- `pExactRMSE`: first-step analytic pressure check,
- `ugIters`, `ugResidual`: final gas-component Helmholtz solve diagnostics,
- `ulIters`, `ulResidual`: final liquid-component Helmholtz solve diagnostics,
- pressure iterations and residual.

## Checkpoint files

At checkpoint steps:

- `ugH`: gas momentum predictor,
- `ulH`: liquid momentum predictor,
- `ug`: pressure-corrected gas velocity,
- `ul`: pressure-corrected liquid velocity,
- `upred`: volumetric mixture predictor,
- `ucorr`: corrected volumetric mixture velocity,
- `gradp`: common pressure gradient,
- `tfdiag`:
  - scalar00 = pre-projection mixture divergence,
  - scalar01 = post-projection mixture divergence,
  - scalar02 = first-step analytic pressure,
  - scalar03 = numerical minus analytic pressure.

The native NekRS fluid velocity is used only as an auxiliary/output field. The
actual phase velocities are the case-layer `ug` and `ul` fields.


## Helmholtz-diagonal pressure mobility

The pressure response has now been upgraded from the transient-only
approximation

[
D_k = rac{Delta t}{gamma_0ho_k}
]

to an OpenFOAM-style diagonal momentum response based on the actual assembled
SEM Helmholtz diagonal.

For each phase,

[
A_k = rac{ho_kgamma_0}{Delta t}M + mu_k K,
]

and the nodal pressure mobility is approximated as

[
rAU_k = rac{M_{ii}}{operatorname{diag}(A_k)_{ii}}.
]

The mass factor is required because the Helmholtz matrix is a weak-form
operator whereas the pressure correction is applied to the physical nodal
gradient. In the transient-only limit,

[
rAU_k ightarrow rac{Delta t}{gamma_0ho_k}.
]

The common pressure coefficient is now

[
lambda_p(mathbf{x})
=
alpha_g rAU_g(mathbf{x})
+
(1-alpha_g)rAU_l(mathbf{x}),
]

and the phase correction is

[
u_g^{n+1}=u_{g,H}-rAU_g
abla p,
]

[
u_l^{n+1}=u_{l,H}-rAU_l
abla p.
]

This is still a diagonal approximation to the full Schur complement
(A_k^{-1}G), but it is consistent with the actual phase Helmholtz operators
rather than using only their transient terms.

Because the resulting pressure coefficient varies over SEM nodes, the previous
closed-form sinusoidal pressure check is no longer exact and `pExactRMSE` is
therefore reported as `nan`.
