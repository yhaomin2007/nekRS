# Convection Picard iterations based on turbPipe

This example offers the native one-pass solver, the validated frozen-repeat
control, and a nonlinear convection fixed-point iteration. It uses v26's
existing `userConvergenceCheck` / `runInnerStep` callbacks; core code is unchanged.

## Run and select the mode

Run `nekrs picard_turbpipe` from this directory. The sibling `../turbPipe`
directory supplies the mesh and original host UDF, including recycling,
statistics and the original boundary conditions. An isolated copy needs that
sibling directory too. Ascent additionally needs the original YAML if enabled.

```ini
[CASEDATA]
picardEnabled = 1
picardConvection = 1
innerIterations = 3
picardTolerance = 1e-8
repeatTolerance = 1e-9
```

- `picardEnabled=0`: native baseline, no iteration callbacks.
- `picardEnabled=1, picardConvection=0`: original frozen-repeat control.
- `picardEnabled=1, picardConvection=1`: update momentum and scalar convection.
- `innerIterations`: maximum total passes, including the native predictor.
  A value of 1 reproduces the native solve and performs no convection corrector.
- `picardTolerance`: maximum scaled change across U, pressure and scalar between
  successive passes. Each field uses `max|new-old| / max(1,max|old|)`.
  Early termination is allowed only after a corrector (pass 2 or later).
- `repeatTolerance`: tolerance for the frozen control only.

The supplied main timestep is 0.001. Keep `advectionSubCyclingSteps=0` and
`initialGuess=previous` for all equations. Begin with the same timestep as the
validated baseline; increase the maximum pass count if diagnostics show
`status=maxPasses`. Reaching that cap accepts the last iterate, explicitly
without claiming tolerance convergence.

## Discrete convection update

Pass 1 is the unmodified native BDF/EXT predictor. Its assembled RHS is

```
F_native = H_BDF + M EXT(source) - M EXT(N).
```

For subsequent passes, the UDF forms

```
F_U(k+1) = H_U + M_U EXT(source_U) - M_U N_U(U(k)),
F_S(k+1) = H_S + M_S EXT(source_S) - M_S N_S(U(k),S(k)).
```

Here `N_U` evaluates the full `U(k).grad U(k)` product, and `N_S` evaluates
`rho_S U(k).grad S(k)`. Both advecting and advected fields come from the last
completed pass. The native SEM or cubature operators are selected by
`ADVECTION TYPE`. The vector RHS uses native `o_Jw` weights; the scalar RHS
uses native `o_LMM` weights. The scalar operator already includes its density.

This is a nonlinear fixed-point method with convection evaluated on the RHS.
It does not add a nonsymmetric advection operator to the elliptic solve, and
convergence is not guaranteed for arbitrary timestep/CFL. When converged, the
convection contribution is evaluated at the new-time solution. Sources and
velocity HPFRT retain their native explicit EXT treatment.

To recover the convection-free base, the UDF reconstructs native EXT convection
from the saved **pre-lag** solution slots and integration coefficients. On this
fixed mesh with constant transport densities, native ADV history is exactly
the native operator evaluated on those historical fields. Reconstruction uses
all active EXT slots, including initialized restart history; it never reads or
writes the private ADV arrays. The base RHS is then the saved native RHS plus
that extrapolated convection contribution. Every corrector starts from the
same base, avoiding accumulation of corrections.

## State and output

The native driver builds forcing and shifts its ADV/EXT histories once, then
lags solution histories on pass 1 only. The UDF preserves shifted older solution
slots, predictors, pressure start state, integration coefficients, inlet data,
fluid properties, explicit sources and divergence target. Between passes it
restores the replay state, then replaces only the two convection RHS terms.
The original pre-solve scalar diffusion coefficients are restored; native AVM
is recomputed after each scalar solve. AVM is allowed to change with the new
scalar and is excluded from the frozen-property equality check in Picard mode.

Recycling, statistics, checkpoint output and final-step cleanup execute only
once, after acceptance. Time advances once per timestep.

`picardInner` reports maximum/RMS successive-iterate differences and the scaled
maximum. Pass 1 reports zero by definition and is labeled `predictor`, never
`converged`. Later statuses are `iterating`, `converged` or `maxPasses`.
`historyMax` and `frozenMax` must remain exactly zero; the intentionally updated
RHS is excluded from `frozenMax` in convection mode. Frozen mode retains
`frozenInner` diagnostics and `picard_repeatability.csv`.

The case remains restricted to one fixed mesh/session, one passive scalar,
constant transport densities, native Navier-Stokes convection, separate velocity
predictor storage and no additional solver callbacks. Moving meshes, low Mach,
density splitting, flow-rate correction and subcycling are rejected. This is a
turbPipe development example, not yet a two-phase solver corrector.

## Verification

```bash
OCCA_DIR=/path/to/built/occa python tests/frozen_inner.py
python tests/device_section.py
```

If `OCCA_DIR` is unset, the test uses `../../../occa-cub-parser` as before.
The first test uses real OCCA Serial memory, production helper/step code,
native BDF/EXT sumMakef kernels, history-lag functions and inner-step driver.
Controlled nonlinear convection and elliptic/BC/AVM test doubles exercise
BDF1/2/3 startup, disabled/1/2/4 frozen passes and 2/4/12 convection passes.
An independent native RHS assembly with zero advection verifies every corrected
RHS against BDF history plus EXT sources minus mass-weighted latest convection.
Tests also verify unchanged older histories, once-only recycling/checkpoints,
early convergence and history-corruption detection. These are bookkeeping and
RHS tests, not full spatial-discretization or turbulent-flow validation.
The second test uses the native OKL extractor and verifies preserved BCs.
A full GPU/MPI run of the new convection mode is still required.
