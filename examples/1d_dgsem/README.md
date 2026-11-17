# 1d_dgsem — periodic 1D DGSEM example

`reference.py` is the complete Python script supplied by the user on
2026-11-16, retained without algorithm or CLI changes. It replaces the
reconstructed reference in commit 8fd461e.

The reference is independent of NekRS. `generate_reference.py` runs that
**same script**, reads its NPZ snapshots, and exports initial states, initial
RHS and final states for comparison with the actual production OKL kernels.
There is no second Python time integrator in the export driver.

## Exact reference settings

- Periodic [0,1], prescribed u=1; conservative s_t+(us)_x=0.
- Default p=3, p+1 GLL points, diagonal local mass, separate element traces.
- Strong DGSEM, classical RK4; every stage reevaluates volume and face terms.
- central / upwind / LF. Python LF uses global max(abs(u)).
- No limiter, clipping, filtering, diffusion, or cubature/dealiasing.
- Sine: sin(2*pi*x), with no offset or amplitude scaling.
- Step: [0.25,0.5); each aligned element is initialized from its interior.
  The two traces at a discontinuity are intentionally different.
- Smooth: tanh pulse between 0.25 and 0.5, delta=0.025, periodic images -4..4.
- Script dt_limit = CFL*h*min(w)/max(abs(u)); default CFL=0.2.
  This is **not** the NekRS alphaDGCfl convention.
- Five output times: linspace(0, periods, 5). Each interval has its own
  ceil(duration/dt_limit) step count, with dt adjusted to hit its end exactly.
- Smooth-profile relative L2 is quadrature-weighted; step L2 is intentionally
  omitted because the point value at a jump is convention-dependent.
- Plots interpolate each element independently; nodal and sampled extrema
  are reported separately. Do not merge duplicate interface coordinates.

The reference also contains an advective-form option at the Python class
level and variable-velocity self-tests. The CLI transport runs and production
comparison are **conservative and constant-speed only**.

This example lives at `examples/1d_dgsem`, independently of the
bubble-column test directory. Python transport needs only NumPy and Matplotlib.
The optional kernel comparison deliberately uses
`../bubbleColumn_dg/alphaDG.okl` so it tests the production implementation.

## Run the original script

Requires Python 3, NumPy and Matplotlib. From this directory:

```sh
cd examples/1d_dgsem
python3 reference.py --profile smooth --elements 16 --output results/smooth16
python3 reference.py --profile step --elements 16 --output results/step16
python3 reference.py --profile sine --periods 5 --output results/sine5
```

Each command runs all three fluxes and writes profiles.png, errors.png,
metrics.csv and central.npz / upwind.npz / lf.npz. The CLI is the supplied
script's CLI: --elements takes **one integer**, and profile names are
**sine, step, smooth**. These differ from the previous reconstructed script.

## Generate the complete matching test suite

```sh
python3 generate_reference.py
```

Defaults: 16/32/64 elements, p=3, sine/step/smooth, one period, CFL=0.2,
delta=0.025. Each run invokes reference.py as a subprocess to produce its
plots, metrics and snapshots. To make a smaller suite:

```sh
python3 generate_reference.py --elements 16 --profile step --output reference-results-step
```

Directory layout:

- reference-results/smooth_n16_p3/profiles.png, errors.png, metrics.csv, *.npz
  are written by the original script.
- reference-results/smooth_n16_p3/upwind/reference.dat contains
  element-local x, initial state, initial RHS, final state (whitespace-separated).
- Each flux subdirectory also contains config.json and profile_000.csv through
  profile_004.csv. Config records the precise time step, step count, settings
  and a SHA-256 of the reference source.
- reference-results/summary.json lists the fixture configurations.

The exporter uses DG1D.rhs and initial_state from reference.py directly,
verifies the NPZ initial state, checks finiteness and absolute mass drift
<=1e-10, and preserves the exact RK4 step schedule. It never clips the data.
Old fixtures from the reconstructed reference are not compatible; regenerate
them instead of reusing them.

## Compare actual production kernels

```sh
export OCCA_DIR=/path/to/built/occa
python3 compare_kernels.py --results reference-results --ranks 1 2
```

Requires a C++17 MPI compiler, MPI launcher and built OCCA (lib/libocca.so).
MPICXX and MPIEXEC can override the compiler and launcher; the latter must
accept -np. The harness currently requires p=3 and a mesh divisible by every
requested rank count. The driver rejects stale reference source hashes.

The C++ harness calls the repository's alphaDG.okl volume, trace, surface,
lift and RK4 kernels, plus src/core/kernels/wDivergenceVolumeHex3D.okl.
It embeds the 1D problem in Cartesian hexes with constant transverse states,
periodic x traces and zero transverse wall flux. It compares **every 3D
node** with the original script's initial RHS and final state, including
the step's independent element endpoint values.

Acceptance: absolute max RHS/final difference <=1e-9 and absolute mass drift
<=1e-10; global RHS balance is checked at every stage. Failure exits nonzero.
Each rank writes CSV data; each MPI run has a saved log in its flux directory.

For a configured GPU OCCA installation, a single-rank example is:

```sh
python3 compare_kernels.py --results reference-results --ranks 1 --device "{mode: 'CUDA', device_id: 0}"
```

Multi-rank GPU binding remains the launch environment's responsibility.

## Relationship to bubbleColumn_dg

The script's **strong form** and the production kernel's **weak form** are
equivalent under the GLL summation-by-parts identity, for the same collocated
volume flux and surface flux. The production comparison checks that
equivalence numerically rather than assuming it.

Production LF uses local max normal speed; reference LF uses global max speed.
For these u=1 cases they both equal upwind. This test does not establish
variable-speed LF equivalence.

This is a standalone production-kernel test, not a full nekrs .par/.udf
fluid case. Trace exchange uses test-only dense MPI_Allreduce with known
periodic face IDs. It does **not** validate production OGS/globalFaceIds,
arbitrary face orientations, curved meshes, or the EE pressure/momentum
coupling. Neither core nor production bubble-column files are modified.

## Validation status

After relocation to `examples/1d_dgsem`, the actual Python suite was run
successfully: sine, step and smooth profiles; 16/32/64 elements; all three
fluxes (27 fixtures). It generated plots, metrics and comparison inputs.
The maximum absolute final mass drift across all fixtures was 6.94e-16.

Actual Python upwind smooth-profile errors (p=3, CFL=0.2, delta=0.025,
one period):

| Elements | Steps | Relative L2 |
| --- | --- | --- |
| 16 | 480 | 5.54770008% |
| 32 | 960 | 1.03037443% |
| 64 | 1920 | 0.07927838% |

These recover the historical rounded 5.55%, 1.03%, 0.0793% values.
The relocated production comparison driver found its reference inputs and
OCCA library, but C++ compilation stopped because the environment lacks
`mpi.h`. Production OCCA/MPI and GPU execution therefore remain unverified.
