# Frozen inner-iteration demo based on turbPipe

This case tests the bookkeeping required for a later Picard corrector without
changing any core code. It uses the existing v26 `userConvergenceCheck` and
`runInnerStep` loop. Each timestep solves the **same frozen discrete system**
`innerIterations` times, including the native first pass. It compares every
repeat with that first pass. This stage deliberately does not update nonlinear
advection or rebuild forcing between passes: it is a repeatability control,
not yet a nonlinear implicit-convection/Picard solver or a two-phase corrector.
Increasing the pass count should not change the accepted solution beyond the
configured comparison tolerance.

## Run

Run from this directory, within the repository checkout:

```bash
nekrs picard_turbpipe
```

The case reuses `../turbPipe/turbPipe.re2` and includes the original turbPipe
host UDF, preserving its perturbed velocity IC, passive scalar, velocity
recycling, stabilized outlet and statistics. The device boundary section is
copied verbatim because v26 extracts `__okl__` sections before expanding
includes. No mesh is duplicated and the original turbPipe case is unchanged.
A standalone copy of this directory therefore also needs its sibling turbPipe.
The original optional Ascent code also needs its original YAML if enabled.

CASEDATA controls:

```ini
picardEnabled = 1
innerIterations = 3
repeatTolerance = 1e-9
```

- `picardEnabled=0`: native one-pass baseline; no demo callbacks installed.
- `picardEnabled=1, innerIterations=1`: exercise the bookkeeping, one solve.
- `picardEnabled=1, innerIterations=N`: N actual scalar/pressure/velocity passes.
- `repeatTolerance`: per-field `max|repeat-first| / max(1,max|first|)` limit.
  Post-solve AVM coefficient differences use this as an absolute tolerance.
  Immutable histories/forcing must remain exactly unchanged.

The example uses fixed dt, 20 steps, tight linear tolerances and
`initialGuess=previous` for pressure, velocity and scalar. Those initial guesses
avoid stage-1-only extrapolation and private residual-projection history.
The native AVM scalar regularization and velocity HPFRT are retained. Compare
runs using these same settings; the original turbPipe's default adaptive dt,
linear tolerances and pressure projection guess differ from this control.

For separate baseline and repeated runs, copy this directory to a sibling
`examples/picard_turbpipe_baseline`, set `picardEnabled=0` there, and run
`nekrs picard_turbpipe` in each directory with the same MPI layout and backend.
Compare final velocity, pressure and scalar fields at the same time. Use a
common restart if identical initial fields cannot otherwise be guaranteed.
The internal fieldwise repeat comparison is stronger than comparing norms
alone; it is performed on all physical nodes, excluding padding.

## What is saved, and when

1. At the sole `userSource` entry, save the full pre-lag U/scalar history
   arrays, pressure, predictors, contravariant advector, inlet workspace,
   integration coefficients and pre-solve scalar properties in UDF-owned
   device arrays. This callback adds no physical source.
2. Native `makeForcing` constructs RHS and pushes private ADV/EXT histories
   once. Native `runInnerStep(iter=1)` pushes solution histories once.
3. After the first solve, save the first result, full shifted solution
   histories, prepared RHS, explicit-source histories, and post-solve
   properties/divergence. In the replay snapshot only the current solution
   slot is replaced by its step-start value; older slots stay shifted once.
4. If another pass is needed, restore that replay state in `UDF_ExecuteStep`.
   The next native inner pass reapplies the same BCs and solves the same system.
   Neither `initStep`, `makeForcing`, `lagSolution`, `computeUrst`, nor timestep
   advancement is invoked again by the demo. Private ADV histories remain
   untouched. Scalar AVM starts each pass from the **pre-solve** coefficients
   and is naturally recomputed after the scalar solve, as in the first pass.
5. Only after the last pass does the original turbPipe step callback run:
   recycling, averages, Q criterion, Ascent and final-step cleanup occur once.
   The driver then finishes the timestep once. Intermediate checkpoint
   suppression is handled by preserving the original checkpoint request.

The user snapshots are transient and refreshed at each timestep; normal NekRS
restart loading remains responsible for reconstructing native startup history.
They are not additional checkpoint fields.

## Diagnostics

`frozenInner` log lines and `picard_repeatability.csv` report each pass:

- U, p and scalar maximum absolute difference and volume-weighted RMS against
  the first-pass result (U RMS averages all three components).
- `historyMax`: change in the older U/scalar solution slots.
- `frozenMax`: change in RHS, explicit-source histories, BDF/EXT coefficients,
  inlet workspace, advector, fluid properties or divergence target.
- `propertyMax`: change in the post-solve AVM scalar properties.

The first-pass difference is zero by definition; later passes are the test.
Errors outside the limits stop the run. Statistics are appended; use separate
run directories or remove the old CSV when beginning a new comparison.

This control excludes moving meshes, subcycling, density splitting, low-Mach
thermodynamics, constant-flow-rate correction, multiple sessions and extra
user callbacks. Updating nonlinearity inside the step is a separate next
stage and must use the saved pre-lag histories when rebuilding forcing; simply
calling native `makeForcing` again would push private history a second time.

## Local verification and limits

```bash
python tests/frozen_inner.py
python tests/device_section.py
```

The first test needs g++ and the OCCA build at `../../../occa-cub-parser`.
It uses real OCCA Serial memory and the actual v26 deviceMemory class, native
BDF/EXT sumMakef kernels, solution-history lag functions, runInnerStep driver,
and production demo helper/step wrapper. Controlled BC/AVM/elliptic test doubles
exercise BDF1/2/3 startup, seeded older histories, six timesteps, disabled/1/2/4
passes, one-time recycling, checkpoint retention and intentional history
corruption. All final field/history arrays must match the disabled baseline.
The second test uses v26's actual OKL extractor and checks the original BCs.

These tests validate the bookkeeping, not the full CFD discretization. No full
GPU/MPI turbPipe run has been performed in the development environment; the
runtime comparison is provided for that required validation.
