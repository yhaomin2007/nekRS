# Periodic 1D DGSEM reproduction and production-kernel comparison

This test reconstructs the **documented** Python DGSEM method. The original
Python file was not recovered. It is not a claim of byte-for-byte reproduction,
nor a claim that the historical 5.55%, 1.03%, 0.0793% errors have been reproduced.
The old profile/domain/error-norm details were not fully recorded.

## What is fixed here

- Domain [0,1], periodic in x, prescribed u=1, equation s_t+(us)_x=0.
- p=3 by default; p+1 GLL points per element and diagonal local mass.
- Separate endpoint values per element; no CG averaging.
- Weak conservative DGSEM volume term and common interface flux.
- Classical RK4; full RHS recomputed four times per step.
- central (alias center), upwind, LF. LF uses **global max(abs(u))** in Python.
  Production LF uses the local maximum normal speed; these agree in this
  constant-speed test. No variable-speed LF equivalence is claimed.
- No limiter, clipping, filter, diffusion, cubature, or dealiasing.
- Meshes 16, 32, 64; tanh delta=0.025; one full period by default.
- dt <= cfl*(h/2)*min(GLL weights), cfl=0.1. Steps align with quarter periods.
- Relative L2 = sqrt(sum(M*(s-exact)^2)/sum(M*exact^2)), using nodal GLL quadrature.
  Mass is sum(M*s); mass drift is absolute, not relative.

Initial profiles:
- sine: 0.5+0.25*sin(2*pi*x).
- square: 1 for 0.25<x<0.75, 0 elsewhere, including the two jump points.
- tanh: 0.5*[tanh((x-0.25)/delta)-tanh((x-0.75)/delta)],
  summed over three periodic images (-1,0,1).
The analytic solution is the initial profile evaluated at (x-t) mod 1.

## Run the reference

From this directory, with Python 3 and NumPy installed:

```sh
python3 reference.py
python3 reference.py --profile sine --periods 5 --output sine-five-periods
python3 reference.py --profile tanh --flux upwind --elements 16 32 64
python3 reference.py --profile square --flux central upwind lf
```

Outputs under results/<profile>_<flux>_n<elements>_p<degree>/:
- profile_000.csv, profile_001.csv, ...: x, numerical scalar, exact scalar,
  at every quarter period. **Duplicated interface coordinates are intentional.**
- history.csv: time, mass, mass drift, relative L2, minimum, maximum.
- config.json: exact run settings and final diagnostics.
- reference.dat: element-local x, initial state, initial RHS, final state,
  for comparison with the production kernels.
- results/summary.json: all run summaries.

The script checks SBP, weak/strong RHS agreement on random discontinuous
element states, mass conservation with variable nodal velocities, constant
preservation, constant-speed LF/upwind equality, RK4 temporal order, finite
solutions and smooth upwind mesh convergence. Square-wave overshoots and
undershoots are expected; boundedness is not an acceptance criterion.

## Compare the actual NekRS kernels against Python

```sh
export OCCA_DIR=/path/to/built/occa
python3 compare_kernels.py --results results --ranks 1 2
```

Requirements: a C++17 MPI compiler, MPI launcher, and built OCCA.
MPICXX and MPIEXEC may override the compiler/launcher; the launcher must accept
-np. The driver builds kernel_reference.cpp and invokes the repository's
**actual** alphaDG.okl (volume, trace, surface, lift, RK4) and native
src/core/kernels/wDivergenceVolumeHex3D.okl. It does not copy their equations
into a second C++ solver.

The harness embeds each 1D element into a Cartesian hex, with constant states
in y/z, periodic x traces, and zero-flux transverse walls. It compares every
3D node against the Python initial RHS and final state. Tolerances:
max absolute RHS/final difference <=1e-9; absolute mass drift <=1e-10.
It also checks the global RHS balance at every stage. Failure exits nonzero.
Each rank writes a CSV with element/node IDs, x, initial, RHS, final and
reference final. Driver logs are saved alongside the data.

For a configured GPU OCCA installation, an explicit single-rank example is:

```sh
python3 compare_kernels.py --results results --ranks 1 --device "{mode: 'CUDA', device_id: 0}"
```

Multi-rank GPU binding is the responsibility of the launch environment.

**Scope:** this is a standalone production-kernel test case, not a complete
nekrs .par/.udf fluid simulation. Its MPI face exchange intentionally uses
dense MPI_Allreduce with known periodic face IDs. It does **not** validate
production OGS/globalFaceIds, arbitrary mesh orientation, curved geometry,
or EE coupling. Those still require actual bubbleColumn_dg runs.
No core or production bubble-column files are changed by this test.

## Validation status when added

The execution host was unavailable, so Python, C++ compilation, OCCA, MPI and
GPU execution were **not run**. A separate JavaScript arithmetic reconstruction
of this specified p=3 upwind tanh test was run: relative L2 errors at 16/32/64
elements were about 3.76384%, 0.709341%, 0.0545550%; absolute mass drift was
<=6.2e-16. These are arithmetic checks, not results from reference.py or the
production kernels. The different historical percentages are not used as
assertions. Run the commands above to establish executable validation results.
