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


## Direct comparison with the original turbPipe case

Both supplied `.par` files now use polynomial order 7, Tombo2, fixed `dt=0.001`,
1000 steps, no subcycling, `initialGuess=previous` and linear tolerances of
`1e-10`. Native HPFRT/AVM settings are identical. The original `turbPipe` UDF
still runs the native one-pass solver; it only gains the shared line exporter.
The Picard wrapper includes that same UDF, so both use the same sampler.

Use identical mesh contents, MPI rank count, partition/backend and restart or
initialization. The original random perturbation has a deterministic seed per
MPI rank, so the same partition/layout matters. The original case reads
`turbPipe.re2`; Picard reads the requested local lowercase `turbpipe.re2`.
If needed, copy `../turbPipe/turbPipe.re2` to `turbpipe.re2` in the Picard run
directory. Verify the two files are identical, not just similar geometries.

Run the cases in separate directories, for example from this directory:

```bash
(cd ../turbPipe && nekrs turbPipe)
nekrs picard_turbpipe
python compare_lines.py ../turbPipe . --steps 0 100 500 1000
```

Use the same MPI launch configuration for both runs on your machine.
Set `picardEnabled=1, picardConvection=1` for the actual Picard comparison.
For a control, `picardConvection=0` should reproduce the native profiles within
solver/roundoff tolerance. Check the step-zero comparison before interpreting
later differences.

Shared CASEDATA controls in **both** files:

```ini
lineProfilesEnabled = 1
lineProfileInterval = 100
lineProfilePoints = 201
```

Each run writes `lineProfiles_step000000.csv`, `lineProfiles_step000100.csv`,
etc., including the final accepted step even if it is off the interval.
Exports occur once per accepted physical timestep, never per Picard pass.
The supplied field checkpoint interval is also 100 steps; line export has its
own control. Existing files for the same step are replaced. Use fresh run
directories or remove old line-profile CSVs before a new comparison.

Five lines, with 201 points each, sample instantaneous `ux,uy,uz,p,s00`:

| Line | Location in the original D=1, L=20 pipe |
| --- | --- |
| `x_z25` | x diameter, y=0, z=5 |
| `x_z50` | x diameter, y=0, z=10 |
| `x_z75` | x diameter, y=0, z=15 |
| `y_z50` | y diameter, x=0, z=10 |
| `axial` | x=y=0, z along the pipe |

Positions are computed from the global mesh bounds; endpoints are inset by
`1e-6` of each span to avoid geometric boundary ambiguity. The implementation
follows gabls' collective `pointInterpolation_t::setPoints/find/eval` pattern,
with rank zero owning all requested points. Missing points abort the run.
Coordinates, physical time, step, line name and point index are included in
CSV files. These are line profiles, not azimuthal or statistical averages.

The comparison script needs NumPy and Matplotlib. It verifies matching steps,
physical times and locations, then produces five overlay PNGs per requested
step and `line_comparison/line_errors.csv` for **every** exported step.
Use `--steps all` for every export, or omit `--steps` for the final export.
`--output DIR` changes the plot/summary destination.

For each field and line, the summary gives maximum absolute difference,
line-integrated RMS difference and relative L2 difference (Picard minus native,
native as denominator). A zero reference norm with nonzero difference yields
`nan` for relative L2; absolute measures remain available. Pressure is compared
as exported, without subtracting a gauge offset. The script reports initial
profile mismatch explicitly. Agreement on these lines does not prove equality
of the entire 3D fields, and the native case is a comparison baseline rather
than an exact solution. Timestep refinement is needed to assess temporal
accuracy; long-time instantaneous turbulent trajectories may diverge.

Additional checks:

```bash
OCCA_DIR=/path/to/built/occa python tests/line_profiles.py
python tests/compare_lines.py
```

The exporter check compiles the production helper with real OCCA memory and
controlled interpolation/MPI doubles. It covers packing, five-line coordinates,
cadence, final-step and duplicate guards, disabled mode and missing points.
The plotting check verifies known metrics, rejects mismatched times/locations
and generates overlay figures from synthetic data. Full distributed sampling
and physical profiles still require actual NekRS runs.
