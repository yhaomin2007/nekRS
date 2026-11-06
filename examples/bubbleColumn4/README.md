# bubbleColumn4: alpha -> pressure -> gas -> liquid

Branch baseline: `bubbleColumn` at `961980d6`. This case implements the
pressure-first two-fluid algorithm in section 6 of the supplied AT document,
for fixed mesh, constant phase densities/viscosities, and no phase change.

## Core scheduling and rebuild

`fluidSolver_t::solve()` in `src/solver/fluid/fluidSolver.cpp` now supports a
paired `userPressureSolve` / `userVelocitySolve` override. With neither set,
the native single-phase pressure/velocity path is unchanged. Partial overrides
or simultaneous native fluid elliptic solvers are rejected.

The native timestep does the following, before its convergence decision:

1. Prepare phase BDF/EXT loads; lag histories exactly once in the native host.
2. Solve alpha only in the native scalar stage.
3. `postScalar`: bound alpha, reapply prescribed boundaries, freeze drag.
4. `fluid->solve`: solve the new shared pressure.
5. At that fixed pressure, solve the three gas components to full-stress convergence.
6. At the same pressure, solve the three liquid components to full-stress convergence.
7. Reconstruct volume velocity and measure final continuity.

`UDF_ExecuteStep` only checks completion and exports fields. It does not evolve
velocities. There are no pressure-free phase predictors, velocity projection,
pressure correction iterations, or Method3 composed-pressure stabilization.
The internal phase iterations solve the viscous vector coupling at fixed pressure.

**Rebuild and reinstall the core from this branch.** An existing Method3 install
cannot load this UDF because it lacks the new fluid callbacks. For an already
configured build:

```sh
cmake --build build --target install -j8
cd examples/bubbleColumn4
nekrs --setup bubbleColumn4 --build-only
nekrs --setup bubbleColumn4
```

The column mesh is external to the repository. Copy your existing mesh to
`../bubbleColumn2/bubbleColumn2.re2`, or change `[MESH] file` in the PAR file.
Its IDs must be 1=inlet, 2=pressure outlet, 3=wall.
The default PAR runs 20 steps at dt=1e-5, N=5, cubature order=10.

## Discrete equations

Write `c=g0/dt`, `a=alpha_g^(n+1)`, and

```
Hk = sum_j BDF_j uk_history_j / dt - EXT(uk.grad uk) + gravity + EXT(filter)
Vke = -(assembled mass)^-1 assembled((Klap + Kcross) EXT(uk))
f_g = lambda_g (EXT(ul) - EXT(ug))
f_l = -lambda_l (EXT(ul) - EXT(ug))
beta = a/rho_g + (1-a)/rho_l
F = a (Hg + Vge + fg) + (1-a) (Hl + Vle + fl)
```

Pressure is the variable-coefficient Poisson problem

```
(beta grad p, grad q) = (F, grad q) - c <prescribed UM.n, q>
```

with p=0 on outlet ID 2. The normal momentum condition on inlet/wall provides
the boundary lifting. Zero phase change implies Q=div(UM)=0.
The harmonic coefficient has no dt factor and is NOT the mass-average density.
Historical phase velocities are weighted with NEW alpha; historical mixture
velocity is not substituted for that sum.

The two phase equations are

```
(c M + Klap_k + Kcross_k) uk = M (Hk + fk - grad(p)/rho_k)
```

`Klap` is the native scalar stiffness. `Kcross` uses element-local, unaveraged
physical gradients and weak divergence to represent the transpose-gradient
stress. Raw native gradients carry Jw, so they are divided by element-local
mass before constructing stress. Three scalar Helmholtz solves serve as a
block-Jacobi preconditioner; half damping converges the full vector system.
The actual assembled free-row full-stress residual is checked every iteration.
Scalar stiffness actions pass a valid ZERO mass field to `Ax`, since these
scalar kernels are Helmholtz kernels; a null mass pointer is unsafe there.

The pressure RHS uses extrapolated full viscous stress, while momentum uses
converged implicit full stress. This is a temporal split, and the SEM pressure
stiffness is not the exact discrete Schur complement of the phase Helmholtz
operators. Consequently, pressure convergence alone does NOT guarantee zero
final mixture divergence. The code reports final weak and strong continuity;
no later projection conceals those errors. Time-refinement and mesh tests on
the real column remain required before judging this algorithm.

## Forces, boundaries, and output

- Schiller-Naumann drag is frozen EXPLICITLY after alpha with new alpha and
  extrapolated slip. The same loads enter pressure and both momenta. They are
  equal/opposite per mixture volume. This differs from Method3 implicit drag.
  Abort if `dt*(max(lambda_g)+max(lambda_l))>1`; reduce dt when this triggers.
- Each phase has its own constant kinematic viscosity, specified by the three
  scalar `diffusionCoeff` values, which must match within that phase.
- Both phase velocities are prescribed at inlet and are zero on the wall.
  The outlet uses homogeneous natural viscous traction in the converged weak
  full-stress system, with p=0. Alpha is prescribed at inlet and has zero
  diffusive normal flux at wall/outlet. No slip-wall variant is included.
- Alpha retains conservative cubature transport plus configurable numerical
  diffusion and HPF regularization. Clipping changes its integral; the signed
  `alphaClipMass` diagnostic explicitly reports that contribution.
- Virtual mass, lift, wall lubrication, turbulent dispersion and phase change
  are excluded. Fixed dt, no moving mesh, no NekNek, no subcycling, one outer stage.
- Scalars: alpha, ugx, ugy, ugz, ulx, uly, ulz. Native velocity is reconstructed
  VOLUME velocity, native pressure is physical shared pressure. Separate ug/ul
  vector files are written on checkpoints.

Every step prints `method4Pressure`, gas `method4Momentum`, then liquid
`method4Momentum`. At `monitorInterval`, `method4Continuity` reports final weak
free-row norm, strong divergence, alpha/pressure ranges, explicit drag dt,
alpha clipping contribution, both phase velocity L2 norms and phase CFLs. The momentum
residual is scaled by `1/c` and by `max(sqrt(volume), ||uk||_L2)`.

## Validation and short native cases

```sh
python3 tests/run.py
python3 tests/lifecycle.py
OCCA_BIN=/path/to/occa python3 tests/cuda_translation.py
```

`run.py` compiles the actual production pointwise kernels, core dispatch, phase
solver and pressure assembly against small numerical adapters. It checks full
stress against direct Galerkin block solves, variable-beta pressure and boundary
lifting, common latest pressure, phase isolation, and failure guards.
`lifecycle.py` executes the actual source/post-alpha callbacks for BDF2 with
EXT1/2/3, history ownership, new-alpha drag freezing and clipping mass accounting.
These are host/operator tests, not a native distributed column run.

Optional real OCCA Serial build/execution (set include/library paths for your install):

```sh
c++ -std=c++17 tests/transport.cpp -I$OCCA_DIR/include -L$OCCA_DIR/lib \
  -Wl,-rpath,$OCCA_DIR/lib -locca -o /tmp/method4-transport
/tmp/method4-transport bubbleColumn4Transport.okl bubbleColumn4Equations.okl
```

Prepare four 20-step native test directories without overwriting the main case:

```sh
python3 tests/prepare_smoke.py /tmp/method4-smoke --mesh /path/to/column.re2
cd /tmp/method4-smoke/plume
nekrs --setup bubbleColumn4 > run.log 2>&1
```

Profiles: plume with drag, plume without drag, uniform mixture, and pure-liquid
hydrostatics. All use monitorInterval=1 and checkpoint at the final step.
For each run inspect final phase/pressure fields and apply a deliberately chosen
continuity threshold, for example from your accepted reference or dt refinement:

```sh
python3 /path/to/nekRS/examples/bubbleColumn4/tests/check_log.py run.log \
  --steps 20 --max-weak-continuity 1e-6
```

`1e-6` is an example acceptance threshold, not a measured result or universal
criterion. For pure-liquid hydrostatics, add a selected `--max-liquid-l2`
tolerance. Repeat with half dt and the same physical duration; compare continuity,
pressure, liquid velocity, and alpha integral. Also check 1-rank versus multiple
ranks to exercise actual gather/scatter and boundary lifting.
