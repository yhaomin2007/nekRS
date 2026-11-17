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
accept -np. The harness supports polynomial degree p>=1 and requires a mesh divisible by
every requested rank count. It builds a separate executable for each degree. The driver rejects stale reference source hashes.

The C++ harness calls the repository's alphaDG.okl volume, trace, surface,
lift and RK4 kernels, plus src/core/kernels/wDivergenceVolumeHex3D.okl.
It embeds the 1D problem in Cartesian hexes with constant transverse states,
periodic x traces and zero transverse wall flux. It compares **every 3D
node** with the original script's initial RHS and final state, including
the step's independent element endpoint values.

Acceptance: absolute max RHS/final difference <=1e-9 and absolute mass drift
<=1e-10; global RHS balance is checked at every stage. Failure exits nonzero.
Each rank writes CSV data. Each MPI run has a unique job directory, a live log,
and automatic postprocessing as described below.

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
The time-export update was compiled and run with OCCA Serial and Open MPI 5.
Matching MPI development headers were supplied through MPICXX. All 18 runs
passed: 16 elements, sine/step/smooth, central/upwind/LF, 1 and 2 MPI ranks,
and all five output times. Maximum all-node difference from Python was
1.36e-14; maximum absolute mass drift was 1.66e-16. GPU and production OGS
execution remain unverified. The harness still uses test-only MPI exchange.

Postprocessing also correctly rejected missing rank files, duplicate nodes
and wrong times, detected a 1e-4 scalar perturbation, and labeled an incomplete
four-snapshot job PARTIAL rather than passed.


## Time exports and postprocessing

Run the reference suite and then the kernel jobs:

```sh
cd examples/1d_dgsem
python3 generate_reference.py
export OCCA_DIR=/path/to/built/occa
python3 compare_kernels.py --ranks 1 2
```

For a shorter first run, generate only one mesh/profile in a separate folder:

```sh
python3 generate_reference.py --elements 16 --profile smooth --output quick-results
python3 compare_kernels.py --results quick-results --ranks 1 2
```

Every kernel job exports at t=0, T/4, T/2, 3T/4, T, exactly matching the
reference script's five snapshots. T is the requested number of periods
because the domain length and speed are both one. No interpolation in time
is used. Each rank exports all its local nodes, preserving separate traces
at the same physical interface coordinate. Never average those traces.

Each run gets a distinct directory, for example:
`reference-results/smooth_n16_p3/upwind/kernel_np2/run_<timestamp>_<id>/`.
The driver prints this full path before launching MPI. This avoids mixing
snapshots from different jobs or silently overwriting an earlier result.

Files inside each job directory:

| File | Contents |
| --- | --- |
| run.json | Exact configuration, reference source hash, MPI ranks, device, running/completed/failed status |
| reference.npz | A private copy of that job's Python reference snapshots |
| job.log | Live solver log, including EXPORT time, step, mass and mass drift |
| kernel.snapshot0.rank0.csv, etc. | Snapshot indices 0..4 for each MPI rank |
| kernel.rank0.csv, etc. | Original final-state and initial-RHS audit |
| postprocess/profiles.png | Kernel vs Python vs exact profiles at all exported times, plus nodal differences |
| postprocess/diagnostics.png | Mass drift and maximum kernel/Python difference vs time |
| postprocess/metrics.csv | All-node max/L2 differences, mass, extrema and transverse consistency |
| postprocess/comparison.npz | Element-local 1D slices and reference arrays for further analysis |
| postprocess/summary.json | Completeness, tolerance status and maximum errors |

Snapshot CSV columns are `step,time,element,i,j,k,x,mass,scalar`.
Element IDs are global for this Cartesian test mesh; i/j/k are local GLL
indices; mass is the 3D local quadrature weight. CSV precision is 17 digits.
A temporary file is renamed only after a full rank snapshot is written.
This permits reading completed times while the job continues.

Postprocessing runs automatically after each successful kernel job. To
regenerate plots or verify all stored jobs manually:

```sh
python3 postprocess.py --results reference-results --check
```

To inspect one running or failed job (replace the directory below with the
printed job path):

```sh
python3 postprocess.py --run path/to/run_directory --allow-partial
```

Only time instants with all rank files present are read. Partial results are
explicitly labeled PARTIAL and never reported as passed. Adding `--check`
requires all five snapshots, a completed job and agreement within tolerance.
If the scheduler kills the driver abruptly, run.json may remain running;
`--allow-partial` still permits inspection of completed exports.

The postprocessor verifies time/step agreement, reference source hash,
complete unique element/node IDs, coordinates and local masses. It compares
**every 3D node**, then selects j=k=0 for plots without averaging x traces.
Max and L2 kernel/Python differences are valid for all profiles; relative
L2 against the analytic step is intentionally omitted. The default acceptance
limits are max absolute difference <=1e-9 and absolute mass drift <=1e-10.
These exports belong to the standalone kernel job; they are not NekRS .f
checkpoints and do not add bubble-column OGS or EE-coupling validation.


## Higher polynomial order

Keep 16 elements and increase p from 3 to 7:

```sh
python3 generate_reference.py --elements 16 --degree 7 --output p7-results
python3 compare_kernels.py --results p7-results --ranks 2 --flux upwind
```

The driver compiles DG_NQ=p+1 for each degree present in the selected fixtures.
The optional --flux selector restricts kernel jobs; the Python reference still
produces all three fluxes. No solver equation or stabilization is changed.
At the same script CFL=0.2, one period needs 2240 RK4 steps at p=7 versus
480 at p=3. Higher order uses more nodes and a smaller time step.

Actual p=7 OCCA Serial / 2-rank MPI runs for sine, step and smooth profiles
passed all five time comparisons. Max kernel/Python difference was 1.32e-14.
For 16 elements and upwind after one period, smooth relative L2 decreased
from 5.54770008% (p=3) to 0.08384707% (p=7). Sine relative L2 decreased from
2.40812e-5 to 3.42038e-12 (fractions). The step still has Gibbs oscillations;
its sampled extrema at p=7 are approximately -0.08593 and 1.08598.
These are kernel/MPI tests, not production OGS or GPU validation.
