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

## NekRS-native two-fluid pressure-first splitting

The current prototype follows the ordering used by the native NekRS fluid
solver rather than an OpenFOAM-style local `rAU` correction or an exact
nested Schur complement.

For the simplified constant-alpha Stokes test, define

\[
A_k = \frac{\rho_k\gamma_0}{\Delta t}M + \mu_k K.
\]

At each step the algorithm is:

1. Build the pressure predictor from the current phase velocities,

\[
U_p = \alpha_g u_g^n + (1-\alpha_g)u_l^n.
\]

2. Solve one common pressure equation using the transient pressure response,

\[
-\nabla\cdot\left(\lambda_p\nabla p\right)
= D_w U_p,
\]

with

\[
\lambda_p =
\frac{\Delta t}{\gamma_0}
\left(
\frac{\alpha_g}{\rho_g}
+
\frac{1-\alpha_g}{\rho_l}
\right).
\]

3. Form the NekRS weak pressure-gradient load \(G_w p\).

4. Solve the FINAL phase momentum equations,

\[
A_g u_g^{n+1}=H_g+G_w p,
\]

\[
A_l u_l^{n+1}=H_l+G_w p.
\]

Thus viscosity and pressure act together inside the final phase Helmholtz
solves. There is no post-processing correction of the form
\(-\Delta t\,\nabla p/\rho_k\), no SEM-diagonal `rAU`, and no nested
\(A_k^{-1}G\) pressure Krylov operator.

This is still a pressure-splitting method rather than a monolithic exact Schur
solve. The key verification metric is therefore the final mixture continuity
error after the phase Helmholtz solves.

The log reports:

- `lambdaP`
- `divPressureRMS`: divergence of the pressure predictor
- `divPostRMS`: divergence after the final gas/liquid Helmholtz solves
- `divRatio`
- pressure iterations/residual
- total gas/liquid Helmholtz iterations and maximum component residual

For the first test, use only a few timesteps and compare `divPostRMS` with the
earlier post-correction prototype.


### Native viscous pressure-RHS correction

The periodic Stokes prototype now also mirrors the constant-property,
zero-divergence-target viscous term used inside native
`fluidSolver_t::solvePressure()`.

For each phase,

[
F_{p,k}
=
rac{1}{Delta t}u_k^n
-

u_k,
abla	imes
abla	imes u_k^n,
qquad

u_k=rac{mu_k}{ho_k},
]

using the same discrete sequence as NekRS: weak/JW-weighted curl, gather-add,
inverse lumped mass, second weak/JW-weighted curl, then gather-add and inverse
lumped mass of the complete pressure forcing.

The common pressure solve is now

[
-
ablacdotleft[
left(
rac{alpha_g}{ho_g}
+
rac{alpha_l}{ho_l}
ight)
abla p
ight]
=

ablacdotleft[
alpha_g F_{p,g}
+
alpha_l F_{p,l}
ight].
]

For this periodic constant-property test, the native boundary/surface pressure
terms and target-divergence terms are zero and are therefore not included.

The final phase momentum equations remain

[
left(
rac{ho_kgamma_0}{Delta t}M+mu_k K
ight)u_k^{n+1}
=
rac{ho_k}{Delta t}M u_k^n
+
G_w p.
]

New diagnostics include `pressureForcingRMS` and
`viscousPressureForcingRMS` so the magnitude of the native viscous correction
can be compared directly with the total pressure forcing.
