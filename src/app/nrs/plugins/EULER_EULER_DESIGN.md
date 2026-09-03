# eulerEuler plugin: design notes

Two-fluid (Eulerian-Eulerian) extension of nekRS, modeled on OpenFOAM's
`twoPhaseEulerFoam`, scoped down to what's implementable as a plugin (no core
solver changes) plus the Phase 0 simplification for phase-fraction transport
that was agreed on before this branch was started.

Scope for this pass (`ee_claude` branch): 2 phases, isothermal, incompressible,
constant particle/bubble diameter, drag-only interfacial coupling. Lift,
virtual mass, wall lubrication, energy transport, and N>2 phases are explicit
non-goals for now (later phases).

## OpenFOAM -> nekRS mapping

| twoPhaseEulerFoam | This plugin |
|---|---|
| `alpha.H` (alpha1 transport, MULES-limited) | nekRS `scalar_t` solving `[SCALAR ALPHA]` with a CG discretization + artificial diffusivity, then pointwise-clipped to `[alphaMin, alphaMax]` (`eulerEuler::step`, `clipAlpha` kernel) |
| `UEqns.H` (both phases' momentum predictors) | Phase 1 (continuous): nekRS's own `fluidSolver_t`, unmodified. Phase 2 (dispersed): **not yet self-advected** -- see Known limitations |
| `pEqn.H` (shared pressure, phase-weighted mobility) | nekRS's own pressure solve, unmodified (phase weighting NOT yet applied -- see Known limitations) |
| `MomentumTransferPhaseSystem::partialElimination()` | `eulerEuler::step`'s call to the `partialElimination` kernel -- closed-form 2x2 solve per node (see below) |
| `dragModels/*` (`Cd*Re` correlations) | `eulerEuler_dragKi()` in `kernels/eulerEuler.okl` -- Schiller-Naumann, Wen-Yu, Ergun, Gidaspow |

## Why a plugin, not a core module

nekRS already has a low-invasiveness extension pattern used by `lowMach` and
`RANSktau` (`src/app/nrs/plugins/`): a namespace with `setup()`/`buildKernel()`
plus physics-driver functions, wired into a case's `.udf` via
`UDF_LoadKernels()`/`UDF_Setup()`/`UDF_ExecuteStep()`. This plugin follows that
pattern exactly, which means the entire framework in this pass touches zero
files outside `src/app/nrs/plugins/` -- nothing in `master` or any other branch
is at risk.

The cost of that choice is real, though: nekRS's outer time-integration loop
(`nrs_t::initInnerStep`/`runInnerStep`/`finishInnerStep`) has no general
"outer corrector" hook comparable to OpenFOAM's PIMPLE loop -- the
`outerCorrector`/`initOuterStep`/`runOuterStep` members that exist on `nrs_t`
are specific to NEKNEK (overset-mesh) multirate stepping and are not safe to
repurpose. So this plugin cannot insert itself *inside* the pressure-velocity
solve; it can only run before/after it via UDF hooks. See Known limitations.

## The partial elimination (PEA) closed form

For exactly 2 phases the PEA linear system collapses to a per-node 2x2 solve
(no spatial coupling -- this is why it's implementable as a pointwise OKL
kernel with no elliptic solve of its own), which is the same simplification
`multiphaseEulerFoam`'s N-phase PEA reduces to when N=2:

```
(A1 + K) U1 - K U2       = A1 * U1_old
   -K U1 + (A2 + K) U2   = A2 * U2_old + alphaD*(rhoD - rhoC)*g
```

with `A1 = alphaC*rhoC*g0/dt`, `A2 = alphaD*rhoD*g0/dt`, and
`K = alphaD * Ki(alphaC, rhoC, nuC, d, |U1-U2|)` the linearized drag
coefficient. `g0/dt` stands in for OpenFOAM's `1/rAU` (the reciprocal of the
momentum equation's diagonal, i.e. an implicit-Euler-style mass/timestep
term) -- this is a simplification of the true `rAU` (which includes the
convection/diffusion operator's diagonal, not just the time derivative);
revisit if drag-dominated cases show it's too crude.

The continuous phase's own gravity/body force is assumed to already be
applied through nekRS's normal `userf`/OUDF body-force mechanism and is
intentionally **not** duplicated inside `partialElimination` -- only the
dispersed phase's reduced-gravity buoyancy term is added there, since phase 2
has no other forcing path yet.

## Known limitations (v0 framework)

1. **Operator-split, not simultaneous, pressure-drag coupling.** OpenFOAM
   solves `pEqn.H` and the PEA together within each PIMPLE corrector, so
   pressure "sees" the drag-implicit velocities in the same iteration. This
   plugin instead runs the PEA correction in `UDF_ExecuteStep`, i.e. strictly
   *after* nekRS's normal fluid+scalar step (and its pressure solve) has
   already completed. This is a splitting error whose size scales with how
   large the drag/timestep ratio is. Closing it requires blending
   `alphaC/rhoC` and `alphaD/rhoD` into the `o_lambda` mobility field built in
   `fluidSolver_t::solvePressure()` (`src/solver/fluid/fluidSolver.cpp`) --
   core-solver work, deferred to a later phase.
2. **No dispersed-phase self-advection/diffusion yet.** `eulerEuler::step`
   only applies drag + buoyancy to `o_U2`; there's no `UEqns.H`-equivalent
   convection/diffusion solve for phase 2. Right now `o_U2` only evolves
   through interfacial coupling with phase 1.
3. **Alpha clipping, not bounded transport.** Per the accepted Phase 0
   simplification, alpha is transported with nekRS's native CG scalar solver
   plus a user-tuned numerical diffusivity, then pointwise-clamped to
   `[alphaMin, alphaMax]`. This is not equivalent to OpenFOAM's MULES flux
   limiter and can introduce small local mass-conservation error near the
   clip bound. A grid-Peclet-based sizing rule for the diffusivity is
   `nu_art ~ |u|*h/(2N)` (h = element size, N = polynomial order) -- start
   there and tune per case.
4. **Compile-verification now runs in CI, but is a smoke test only.**
   `.github/workflows/ci.yml` now has an `eulerEulerTemplate` job (triggered
   automatically on push to `ee_claude`, same as `master`) that builds
   `libnekrs` -- with `eulerEuler.cpp` now correctly listed in
   `cmake/nekrs.cmake`'s `APP_SOURCES` -- and runs a short (20-step) instance
   of `examples/eulerEulerTemplate` via `examples/eulerEulerTemplate/ci.inc`.
   That harness only checks that the run completes without diverging (both
   phase velocities stay finite, alpha stays within `[0,1]`) -- there is no
   verified reference solution to regress against yet, since the physics
   hasn't been validated. A green CI run means "compiles and doesn't blow up
   for this short run," not "physically correct."

## Example cases

- `examples/eulerEulerTemplate/`: minimal smoke-test template. Reuses
  `examples/lowMach/lowMach.re2` as a placeholder mesh (unverified
  geometry, but wired into CI's `eulerEulerTemplate` ctest as a
  compiles-and-doesn't-diverge check -- see limitation #4).
- `examples/bubbleColumn/`: a real bubble-column case (narrow
  water-filled column aerated from the bottom, air as the dispersed
  phase), modeled on OpenFOAM's `twoPhaseEulerFoam/bubbleColumn`
  tutorial. Its mesh is deliberately NOT checked in (must be generated
  from its `input.box` via `genbox` -- no Fortran toolchain was
  available to do that in this sandbox) and it is NOT wired into CI, so
  treat it as an unbuilt, unrun design: geometry and BC choices follow
  from the plugin's real capabilities (see the deviations documented in
  `examples/bubbleColumn/bubbleColumn.par`'s header) but nothing about
  it has been compiled, meshed, or executed even once.

## Wiring a case (see `examples/eulerEulerTemplate/`)

1. Add the phase-fraction scalar in `.par`: `scalars = ALPHA` plus a
   `[SCALAR ALPHA]` section with `diffusivity = <artificial value>`.
2. In `.udf`'s `UDF_LoadKernels()`: `eulerEuler::buildKernel(kernelInfo);`
3. In `.udf`'s `UDF_Setup()`, after the case is otherwise set up:
   `eulerEuler::config_t cfg; cfg.rhoDispersed = ...; ...; eulerEuler::setup(cfg);`
   (optionally call `eulerEuler::setIC(...)` to set a nonzero initial
   dispersed-phase velocity).
4. In `.udf`'s `UDF_ExecuteStep()`: `eulerEuler::step(time, tstep);`

## Exporting the dispersed-phase velocity to output

`eulerEuler::o_U2()` is a private plugin buffer -- it is never registered
with nekRS's checkpoint writer (`src/core/iofld/iofldNek.cpp`'s field
registry only knows `mesh`/`velocity`/`pressure`/`temperature` plus
registered scalars), so a run's `.fld` files contain the continuous
(liquid) phase's velocity and alpha, but NOT the gas velocity, unless a
case explicitly exports it.

`examples/bubbleColumn/` shows the pattern: declare 3 extra `solver = none`
scalars (`GASUX`/`GASUY`/`GASUZ` in the `.par`'s `scalars = ...` list --
`solver = none` means nekRS never solves/advects/applies BCs to them, see
`scalarSolver.cpp`'s `compute[is] = 0`), then in `UDF_ExecuteStep`, after
`eulerEuler::step(...)`, copy each component of `o_U2` into its slot with
`nrs->scalar->o_solution("gasux").copyFrom(eulerEuler::o_U2(), mesh->Nlocal, 0, 0 * nrs->fieldOffset)`
(and similarly with `1 *`/`2 * nrs->fieldOffset` as the source offset for
y/z). nekRS then writes them to the checkpoint like any other scalar.

## Next phases (not started)

- Phase 5: core-level phase-weighted pressure mobility (remove limitation #1).
- Phase 6: dispersed-phase momentum convection/diffusion (remove limitation #2).
- Phase 7+: lift/virtual-mass/wall-lubrication closures, N>2 phases, energy.
