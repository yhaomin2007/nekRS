# Method2 periodic verification cases

These three cases reuse the **current sibling `bubbleColumn2` implementation**,
including conservative alpha cubature, UG material convection, native fluid EXT
coefficients for those operators, implicit UG drag, explicit UM drag and the UG
current-pressure corrector. They do not alter bubbleColumn2 or NekRS core.
VM, clipping, HPFRT/AVM, explicit filters, inlet masks and outlet enhancement are
off. All four scalars and UM/P are solved. Use FP64 for these verification runs.

| Directory / session | Geometry | Initial state and purpose |
| --- | --- | --- |
| `case1` | 0.1 m cube, 4x4x4 hex, periodic in x/y/z | alpha=0.05, UG=UL=UM=(0,0,0.3) m/s, g=0. Exact constant translation. |
| `case2` | Same cube and mesh | alpha=0.05, UMz=0.05 m/s; terminal slip balances gravity, prescribed mean pressure gradient and drag in **both** phases. Exact uniform equilibrium. |
| `case3` | Radius 0.075 m, length 0.3 m, 512 hex; periodic only in z | No-slip UM/UG, zero-normal-gradient alpha at wall ID 1. Parabolic initial velocities. Adds -10 Pa/m to the mixture-hydrostatic mean gradient. A transient wall test, not an exact equilibrium. |

## Prepare and run

From a checkout of the `bubbleColumn` branch:

```bash
cd examples/method2Periodic
python3 prepare.py
cd case1
mpiexec -np 1 "$NEKRS_HOME/bin/nekrs" --setup case1
```

Run `case2` and `case3` analogously from their own directories, with `--setup
case2` and `--setup case3`. Use your normal site launcher/GPU allocation in place
of `mpiexec` on the cluster. The initial meshes are deliberately small; one GPU
is sufficient. Do not use a restart for these baselines.

`prepare.py` uses only Python's standard library. It generates the `.re2` meshes
and copies the three live method2 OKL files plus the periodic kernel/boundary
files into each run directory. Run it again after changing shared sources.
The host UDF includes the sibling method2 implementation directly; keep this
folder alongside `examples/bubbleColumn2`. NekRS does not reliably invalidate
UDF caches when included headers change: clear the affected run directory's
`.cache` before rebuilding after such changes. Generated files are git-ignored.

Defaults: P5 / cubature10, tombo2, dt=1e-5 s, 10000 steps (0.1 s), checkpoint every
1000 steps, diagnostics every 100 steps plus the first step. Pressure/velocity
and scalar residual tolerances are 1e-10. Scalar baseline diffusion is 1e-8 m2/s.
UG baseline diffusion is explicitly subtracted, retaining the existing physical
gas stress rather than adding a second gas viscosity. Alpha retains its small
baseline diffusion; for uniform alpha its contribution is zero. The physical
phase-stress / native-baseline correction in UM is active. No forces are frozen
as a manufactured cancellation and no velocity is restored during timestepping.

## Pressure decomposition and the equilibrium

The native periodic field is **p'**, not a periodic representation of a linear
pressure ramp:

\[
p = G_0 z + p',\qquad \nabla p=G_0\mathbf e_z+\nabla p'.
\]

Only the known mean gradient is added by the new source wrapper:

\[
f_{g,z}^{G_0}=-G_0/\rho_g,\qquad
f_{m,z}^{G_0}=-\left[(1-\alpha)/\rho_l+\alpha/\rho_g\right]G_0.
\]

Both are **accelerations** in the native source interface; do not multiply the
UM source by rhoPressure again. Native velocity RHS construction applies that
factor. The UG pressure corrector replaces only its captured grad(p') load, so
it preserves this separately prescribed mean pressure load.

For case2, rho_m=(1-alpha)*rho_l+alpha*rho_g and G0=rho_m*g_z. A bisection solve
using the same Schiller-Naumann Ki as method2 finds signed w=UGz-ULz from

\[
K_i(|w|)w=(\rho_g-\rho_m)g_z,
\quad U_{g,z}=U_{m,z}+(1-\alpha)w,
\quad U_{l,z}=U_{m,z}-\alpha w.
\]

With rho_l=998.2, rho_g=1.2 kg/m3, mu_l=1e-3 Pa s, d_b=3e-3 m,
alpha=0.05 and g_z=-9.81 m/s2:

- G0 = **-9303.3135 Pa/m**.
- w = **0.2830977116194415 m/s**.
- UGz = **0.3189428260384694 m/s**.
- ULz = **0.03584511441902793 m/s**.

Both phase accelerations and UM acceleration are zero. The actual phase drag
and gravity/pressure source kernels remain enabled; the test checks their
balance, not a replacement forcing that forcibly holds the answer fixed.

For case3 the initial velocities above are multiplied by
`1-(x*x+y*y)/(0.075*0.075)` and G0 additionally includes `drivingGradient=-10`
Pa/m. Thus mean initial UMz is approximately 0.025 m/s, not 0.05. Wall shear,
radially varying slip and extra pressure drive make the fields evolve. This
isolates wall/geometry effects without inlet or outlet corners. It has no
supplied analytic final two-phase profile. Change the driving gradient only
in case3; case2 guards against changing its exact equilibrium condition.

## What to inspect

`periodic_diagnostics.csv` and `METHOD2 PERIODIC` log lines contain max changes
from the initial alpha/UG/UM, volume-weighted alpha and axial velocities, actual
UG CFL and the prescribed G0. For case1/2 changes are exact-reference errors;
case3 changes are physical transients. The first two cases should retain
uniform fields, p' constant up to its gauge, and negligible grad(p').

At checkpoints, open these databases in VisIt:

| Database | Meaning |
| --- | --- |
| `caseN.nek5000` | Native UM, periodic p', alpha/UG checkpoint scalars |
| `ug.nek5000`, `ul.nek5000` | Separate gas/liquid velocity vectors |
| `gps.nek5000`, `gpp.nek5000` | Current raw / predictor grad(p') only |
| `gpt.nek5000` | Current total strong gradient: grad(p') + G0 e_z |

All gradient vectors occupy the `velocity` slots for visualization, in Pa/m,
with positive gradient sign. Case2's total z gradient should remain G0, while
its x/y gradients should remain zero. Column-specific inlet/outlet integral
reports are intentionally replaced by the periodic diagnostics here.

After running:

```bash
python3 ../check_results.py 1   # from case1; default velocity tol 1e-6, alpha 1e-8
```

Or run `python3 check_results.py 2` from this folder. The checker assesses every
recorded sample, not unrecorded intermediate steps. Extend `numSteps` after
initial checks; compare case3 with smaller dt and refined mesh before judging
its final profile. Passing case1/2 establishes a uniform-state baseline only;
it does not establish stability for perturbed fronts or the inlet column.

## Local verification

```bash
python3 prepare.py
python3 tests/verify.py
```

The verification script needs NumPy and g++. It checks native re2 record
layout, conformal topology, reciprocal periodic pairing, wall circular arcs,
positive Jacobians at P5 GLL locations, the actual new kernel bodies and the
actual method2 gas/mixture force kernels against independent phase balances.
These checks do not execute the full NekRS MPI/GPU time integration. Run all
three cases on the target installation to complete that verification.
