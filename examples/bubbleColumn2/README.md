# Method2: native scalar transport, volume-mixture momentum

The active case is `bubbleColumn2.udf`, using `bubbleColumn2Native.hpp` and
`bubbleColumn2Native.okl`. It solves alpha and UG with the native scalar solver,
then P/UM with the native fluid solver. **There is no UG pressure corrector.**
No NekRS core files are changed.

## Equations and native transport

UM is the volume average, not the mass average:

\[
U_m=(1-\alpha)U_l+\alpha U_g,\qquad \nabla\cdot U_m=0,
\qquad U_l=(U_m-\alpha U_g)/(1-\alpha).
\]

The reconstruction protects the denominator near alpha=1. The pressure property
is `rhoPressure=1/[(1-alpha)/rhoLiquid+alpha/rhoGas]`; user UM forces are
accelerations, which native velocity RHS construction multiplies by this density.

Native scalar advection uses a **separate UG transport velocity/Urst buffer**.
The fluid solver retains its own UM buffers. The native scalar solver forms
`UG.grad(alpha)` and each `UG.grad(UGi)` from current fields, then its normal
`makeForcing/sumMakef` applies the live native EXT coefficients to complete
operator histories. The case does not predict the gas advector before forming
those operators, does not cancel/reconstruct UM scalar advection, and does not
maintain a separate user convection EXT history.

Alpha retains the explicit compression source:

\[
\partial_t\alpha+U_g\cdot\nabla\alpha
=-\alpha\nabla\cdot U_g+\kappa_\alpha\nabla^2\alpha.
\]

This is the split form of conservative gas continuity. The compression source
uses the trace of the assembled UG gradient, and native EXT also handles its
history. **This replaces the previous custom dealiased conservative-flux
operator; the two spatial discretizations are not identical.** Discrete gas
conservation should be checked in runs with nonuniform alpha/curved geometry.

Cubature follows the v26 native default formula from `parseGeneral.hpp`:
`cubN=round(1.5*(N+1)-1)-1`. For P5 it is 7 (eight GL points per direction).
The case selects cubature advection internally. Do not put
`cubaturePolynomialOrder` in the input; the case rejects that override. The
existing interpolation-table compatibility helper only supplies a missing
native interpolation matrix if the installed table lacks this combination.

## Drag and other forces

The drag coefficient Ki is evaluated from the current available phase fields
using the existing Schiller-Naumann closure. Gas drag is

\[
\lambda^n U_l^{EXT}-\lambda^n U_g^{n+1},
\quad \lambda^n=K_i^n/(\rho_g+c_n),
\quad U_l^{EXT}=\sum_j\beta_j U_l^{n-j}.
\]

UL history uses the **same live native scalar EXT coefficients**, including
startup and variable dt. Native scalar source EXT must not extrapolate this
already predicted liquid load twice. Therefore only this gas drag component
is held at its current `lambda*UL_EXT` value across source-history slots.
Other UG source components retain normal native EXT. The implicit lambda is
snapshotted at source assembly so later properties callbacks cannot change
its pairing with the explicit liquid load. Ki itself is lagged, not predicted.

The full nonlinear **UM drag source uses native fluid EXT**, with no special
history freezing. Gravity, pressure, gas stress, mixture drift stress and the
physical phase-stress/native-baseline correction remain active. No lift,
turbulent dispersion or wall-lubrication model is added by this simplification.

VM remains enabled in the supplied input, as in the preceding case:
`Cvm=0.5`, no VM for 100 steps, ramp over 10000 further steps. Gas added inertia
uses `c=ramp*Cvm*rhoLiquid` in `rhoGas+c`; its explicit liquid-acceleration
source and the complete UM VM source now use normal native source EXT. The
completed-step material-acceleration history still uses derivative order 1
by default. Set `virtualMassEnabled=0` for a VM-off comparison. These settings
are not a claim that the previously observed VM instability is fixed.

The move from scalar EXT1 plus selected user EXT histories to **native EXT for
all explicit scalar sources** also changes the time treatment of direct UG
pressure and viscous/compression sources. This is intentional: it follows the
requested native source/convection path. It is not a fully coupled phase solve.

## Inlet, regularization and outlet

- Inlet alpha and axial UG are spatially uniform: 0.016667 and 0.3 m/s.
  Transverse UG is zero; inlet UMz=alphaInlet*UGinlet (stationary inlet liquid).
- Shared inlet/wall rim nodes retain zero velocity to reconcile the no-slip
  wall with inlet Dirichlet values. There is **no smooth near-wall ramp** in
  the inlet interior. Alpha remains uniform at the inlet, including the rim.
- Initial alpha/UG still use the preceding axial plume/tanh initialization.
  This change concerns the inlet profile, not a frozen UG or a uniform-domain IC.
- The additional user alpha filter is removed from the active path. Native
  HPFRT remains selected in the supplied input; scalar diffusion remains 1e-5.
  Alpha clipping to [0,1] is retained and can be disabled with `alphaClipEnabled=0`.
- There is no outlet damping/ramping region. Scalar baseline diffusion is
  restored uniformly from input on each properties evaluation, before any
  selected native AVM contribution. UM viscosity has no outlet multiplier.
- Native AVM is supported with UG as the transport velocity when selected.
  The supplied input selects HPFRT, not AVM.

## Run and diagnostics

Use the existing column mesh, a fresh state and the usual NekRS invocation:

```bash
mpiexec -np 1 "$NEKRS_HOME/bin/nekrs" --setup bubbleColumn2
```

Use the cluster's normal launcher/allocation. P5, tombo2 and dt=1e-5 are retained.
Clear this case's UDF cache after updating included files. Startup prints
`METHOD2 SIMPLE`, the automatic cubature order and the drag/transport choices.
The current implementation requires fixed mesh, four active native elliptic
scalars with transportCoeff=1, no subcycling and no NekNek.

Checkpoint scalars remain `ALPHA,UGX,UGY,UGZ`. Additional databases are:

| File series | Vector contents |
| --- | --- |
| `ug`, `ul` | Current reconstructed phase velocities |
| `gps` | Raw strong gradient of the newly solved pressure |
| `gpp` | Lagged pressure-gradient sample supplied at source assembly |

Gradient vectors use the visualization `velocity` slots and have units Pa/m.
`gpp` is one source sample, **not the complete EXT-weighted pressure acceleration**;
pressure-source history also includes its density normalization at each sample.
There is no post-pressure UG correction. The timestep log prints alpha min/max
and the actual UG CFL at `stabilityMonitorInterval`.

## Verification and historical cases

The active checks are `tests/native_transport.py` and `tests/native_boundary.py`.
They exercise actual kernels/native BDF-EXT forcing, UL prediction without
repeat extrapolation, history updates, compression and uniform inlet/rim values.
The UDF has also been syntax-checked against actual repository NekRS headers;
new kernels have been executed with OCCA Serial and translated to CUDA. A full
target MPI/GPU build/run and long-time stripe/VM stability remain unverified.

`bubbleColumn2Legacy.udf/.par` preserve the preceding implementation and its
settings. `bubbleColumn2Terms.hpp`, the old transport/filter modules and most
older tests belong to that historical path; **they are not included by the
active UDF**. The separate `method2Periodic` verification cases explicitly use
the legacy UDF so this change does not silently alter their original tests.
See `tests/README.md` for test scope. No Method1/3/4 or core solver code changes.
