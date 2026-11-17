# Method1 with DG alpha and CG phase/mixture momentum

Independent feasibility prototype based on `examples/bubbleColumn` at
`1e05aacd`. LD is disabled. No NekRS core changes are required.

The sequence is **CG UG → DG alpha → conservative CG projection → native
CG pressure/UM**. UL remains reconstructed. The gas/mixture momentum model,
drag, VM, lift, and native pressure coupling retain Method1's formulation.
This case does not use the Fazeli internal injection source or its geometry.

## Run

```bash
cd examples/bubbleColumn_dg
gzip -dk bubbleColumn_dg.re2.gz
mpirun -np 1 nekrs --setup bubbleColumn_dg
```

The included straight mesh is a **new default test geometry**, 0.1 × 0.1 × 1 m,
centred in x/y, with 8 × 8 × 40 = 2560 hexes. `python3 generate_mesh.py`
regenerates it; `--refine 2` produces 20480 hexes. It is not assumed to be the
same as your existing HPC bubble-column mesh. For a controlled CG/DG
comparison, use the **same existing mesh, order, dt, IC and momentum options**
in both cases, with IDs 1=inlet, 2=outlet, 3=impermeable walls. Mesh path is
configurable in `[MESH] file`. Conforming, fixed hex meshes only in this version.

Default polynomial degree is 3 (momentum cubature degree 6), fluid dt=1e-5 s,
and 100000 fluid steps. Start with 100–1000 steps to verify startup. A small
dt is still needed for explicit gas momentum; DG alpha subcycling cannot
stabilize unstable UG/pressure coupling.

## Alpha discretization

`alphaDG::solution` is an independent element-local array with all GLL nodes
retained separately on each element. It is NEVER gather-scattered or clipped
as a CG field. The native ALPHA scalar is only a **CG coupling/output view**;
its elliptic solve, Dirichlet updates and forcing are disabled at runtime.
The scalar slot is allocated normally by NekRS during initialization.

For each element K, the collocated weak DGSEM discretization is

\[
M_K\dot\alpha_K=D_K^T M_K(\alpha u_g)-B_K\widehat F_n.
\]

Volume fluxes use GLL collocation (no dealiasing in this first DG version).
The native `opSEM::divergence` supplies the **unassembled weak** volume load.
The surface kernel offers `alphaDGFlux=central/upwind/lf` (default upwind;
`center` aliases central). With a shared normal speed `un=(unM-unP)/2`,
the flux is `0.5*(un*(alphaM+alphaP)+lambda*(alphaM-alphaP))`.
Here lambda is zero for central, |un| for upwind, and
max(|unM|,|unP|) for local Lax-Friedrichs. LF and upwind coincide for
continuous normal velocity. All three fluxes cancel on paired faces. Local
mass inversion uses **element** `mesh->o_LMM`, not assembled `o_invLMM`.
By default there is no additional alpha*div(UG) source: it is already in conservative
div(alpha*UG). Nonzero div(UG) is supported.

`mesh->globalFaceIds` identifies paired face nodes, including their orientation
across MPI partitions. A separate OGS handle communicates two traces (alpha
and outward normal speed). The DG volume state is not communicated by CG
averaging. Setup checks two participants per interior face node, one per
boundary face node, opposite normals and equal surface quadrature weights.
Face IDs distinguish incident faces even at edges/corners.

The alpha update uses classical RK4 with frozen newly solved UG. A metric/GLL-weight
estimate chooses substeps using `alphaDGCfl`. Its default 0.1 is conservative,
but is not a rigorous stability bound for arbitrary distorted geometries.
The whole coupled method remains first-order in time: `tombo1` is required.

## Boundary conditions and initialization

Existing startup modes are retained: 0=initial plume, 1=zero fields and ramped
alpha/UG inlet, 2=uniform initial UG and ramped inlet alpha. Default is mode 2.
The inlet alpha ramp is evaluated at RK stage times; UG is frozen at the
fluid-step value, so coupled startup accuracy is first-order.

DG alpha BCs are flux conditions:

* Inlet: prescribed ramped exterior alpha for **inflow**, interior alpha for outflow.
* Top: interior alpha for outflow; exterior `alphaDGBackflowValue` for backflow (default 0).
* Walls: exactly zero normal gas flux.

The selected flux is also used at open boundaries. Central averages interior
and prescribed exterior states on inflow; upwind/LF use the prescribed state.

CG velocity BCs retain the original case, including zero UG and UM on walls.
The CG alpha projection is not overwritten with an inlet Dirichlet value:
the prescribed inlet acts on the DG numerical flux. This preserves projected
gas volume. The independently prescribed mixture inlet remains inherited
from Method1; it is not a new tightly coupled phase boundary algorithm.

## DG/CG coupling and safeguards

After each completed alpha update:

\[
\alpha_{CG}=M_{CG}^{-1}Q^T M_K\alpha_{DG}.
\]

This projection conserves the quadrature-integrated gas volume and provides
a single-valued field for density, reconstructed UL, drag/lift/VM coefficients
and the Method1 divergence target. It does not overwrite alphaDG. Gradients
used by CG momentum are gradients of this projection. LD is disabled and
cannot be enabled by the input file in this prototype.

The native pressure target uses the actual projected alpha increment:

\[
\nabla\cdot U_M=\frac{\rho_L-\rho_G}{\rho_M}
\left[\frac{\alpha_{CG}^{n+1}-\alpha_{CG}^{n}}{\Delta t}
+U_M^{EXT}\cdot\nabla\alpha_{CG}^{n+1}\right].
\]

This retains Method1's approximate CG pressure constraint; local conservative
DG alpha transport does not make the full split two-fluid method exactly
phase-consistent or cure its possible short-wave instabilities.

Default `alphaDGLimiter=0` disables limiting. Opt-in `alphaDGLimiter=1` applies a quadrature-weighted element-mean-preserving
scaling limiter after each RK stage. It contracts nodal values toward their
element mean to keep them in [0,1], up to roundoff. It introduces nonlinear
numerical dissipation. It does **not** clip an inadmissible element average.
The run stops if a mean exceeds [0,1] by more than 1e-12, or values become
nonfinite; reduce alpha CFL or fluid dt in that case. This is not a proof of
boundedness for arbitrary compressive UG. Setting the limiter to 0 isolates
unlimited DG with the selected flux, while retaining nonfinite checks.

Classical RK4 is not SSP; enabling the limiter does not guarantee boundedness.
Unlimited central/upwind/LF DG can overshoot sharp fronts. RK4 accuracy refers
to the frozen-UG alpha subproblem, not the full split coupled scheme.

Alpha numerical diffusion must be zero; no DG diffusion operator is included.
Native alpha HPF/AVM options are not applied because the native alpha solver
is disabled. The supplied par omits alpha filtering. UG/UM retain their
original numerical diffusion/HPF settings, explicitly visible in the par.

## Parameters and output

| CASEDATA parameter | Default | Meaning |
| --- | --- | --- |
| laminarDispersionEnabled | 0 | Required; LD disabled |
| alphaDGCfl | 0.1 | Substep stability scale |
| alphaDGMaxSubsteps | 10000 | Abort if more substeps are required |
| alphaDGFlux | upwind | central (alias center), upwind, or lf |
| alphaDGIgnoreDivUg | 0 | Diagnostic compression cancellation (1=on) |
| alphaDGLimiter | 0 | Element-mean-preserving nodal bounds limiter |
| alphaDGBackflowValue | 0 | Top inflow alpha |

Default checkpoint s1 is **projected CG alpha**; s2–s4 are CG UG components.
The separate `alphaDG` output contains the actual discontinuous solution in
s1, written element-wise without averaging. Disable visualization smoothing
when examining interface jumps. Existing `diag` output retains grad(P) and
target/actual div(UM).

At every checkpoint, the separate `gasDiag` field file also contains:

| VisIt field | Quantity | Units |
| --- | --- | --- |
| velocity | Current solved gas velocity UG | m/s |
| s1 (scalar00) | Element-local strong div(UG), no interface averaging | 1/s |
| s2 (scalar01) | Strong div(UG) with native interface averaging | 1/s |
| s3 (scalar02) | -alphaDG * element-local div(UG) | 1/s |
| s4 (scalar03) | Raw element-local DG alpha | dimensionless |

These diagnostics use dedicated buffers and do not change the solution.
Output follows the existing checkpoint schedule; `diagnosticInterval`
does not control field-file frequency. The log reports global extrema of
local div(UG) and -alphaDG*div(UG) at the same output times.
Disable visualization smoothing when inspecting s1, s3 and s4.

Compare s1/s3/s4 and the z velocity component along the same line and at
successive times, including before stripes develop. For positive alpha,
negative div(UG) gives positive material compression -alpha*div(UG);
positive div(UG) gives dilution. This is the full three-dimensional
divergence, not just d(UGz)/dz. Spatial coincidence alone does not establish
causality. The compression product is a continuum diagnostic, not an extra
source and not an exact split of the collocated DG residual (which also
contains numerical face fluxes and polynomial product aliasing).

`alphaDG` log entries report substeps, estimated CFL, raw DG alpha min/max,
gas volume, per-step and cumulative gas-volume balance, projection volume
error and the count of limited element-stages. Boundary fluxes are integrated
with the same classical RK4 weights (1/6,1/3,1/3,1/6), at times (0,1/2,1/2,1) of each substep. With no mass source,
`Vnew - Vold + integrated_outward_boundary_flux` should be near roundoff.

**Fresh starts only:** this first version rejects restart files. Ordinary
checkpoints contain projected alpha and cannot reconstruct the missing DG
jumps. The raw DG visualization file is not yet wired into restart handling.

## Verification and current limits

```bash
OCCA_DIR=/path/to/built/occa python3 tests/run.py
```

The harness executes the production DG kernels and native weak-divergence
kernel with OCCA Serial at polynomial degrees 2, 3 and 7. It checks all
three spatial derivative directions, constant-state preservation, periodic
smooth advection convergence, nonzero-divergence transport, inlet/backflow
fluxes, unlimited sharp-packet mass conservation, element-mean preservation,
and RK4 boundary/global volume cancellation for all three fluxes on 1 and 2 MPI ranks. The test uses
a dense MPI trace exchange for its synthetic mesh; the production sparse OGS
mapping is separately checked at case setup. CUDA translation is checked;
this is not a GPU execution test. Full coupled NekRS MPI/GPU operation and
long-time stability require validation on the target installation.

This is a basic collocated DGSEM implementation, not COMSOL stabilization or
the Fazeli paper's IP-DG solver. Remaining work includes dealiased volume
fluxes, scalable/performance testing, exact DG restart, and a more systematic
assessment of the DG-to-CG momentum/pressure coupling.

**Historical SSPRK3 baseline (not validation of this RK4 revision):**
Development checks passed with 1/2 ranks: degree-2 periodic smooth-transport
L2 errors were 4.85184e-3 (4 elements in x) and 6.30990e-4 (8 elements),
identical across the two process counts. Mass-balance assertions used 1e-12
absolute tolerance for the unit-volume synthetic problem. UDF C++ syntax
also passed against the branch API. No full coupled GPU run has been made.

Repeat audit: degree-3 smooth-transport L2 errors were 4.54952e-4 and
2.95989e-5; degree-7 errors were 5.69873e-9 and 2.25009e-10. These matched
between 1 and 2 ranks. The degree-3 check covers the supplied case order.
These tests use affine elements; rotated/curved production meshes and the
full DG/CG pressure coupling still need integration testing.

### RK4 revision validation

The RK4/flux changes were reviewed and their scalar kernel arithmetic was
checked in a JavaScript runtime (RK4 order, paired flux cancellation, wall
and inflow/outflow behavior). The updated OCCA/MPI harness has **not** been
run for this revision because the editing session had no execution host.
Run `OCCA_DIR=/path/to/built/occa python3 tests/run.py` on the target install.
No full coupled MPI/GPU validation is claimed for this revision.

The lightweight arithmetic audit can be repeated with `node tests/arithmetic.cjs`.
It extracts the RK4 and surface-flux bodies from `alphaDG.okl`; it does not
validate OCCA compilation, device execution, or the full DG spatial operator.

## Standalone 1D DGSEM example

The periodic reference script and production-kernel comparison now live in
[`../1d_dgsem`](../1d_dgsem/README.md). Run them from `examples/1d_dgsem`.
The comparison continues to use this case's `alphaDG.okl` directly.

## Diagnostic test: suppress gas-velocity compression

Set `alphaDGIgnoreDivUg = 1` in [CASEDATA] (default 0). Each RK4 RHS then
adds +alphaDG*div(UG) to the conservative DG residual, targeting

    alpha_t + div(alpha*UG) = alpha*div(UG)

which is pure advective transport at the continuum level. UG divergence
is computed once per fluid step from the newly solved, frozen UG using
element-local strongDivergence(avg=false); alpha is recomputed at every
RK stage. Flux selection, boundary fluxes, limiter and zero diffusion
remain unchanged. This cancels the volume compression contribution;
collocated polynomial product aliasing and DG face terms mean it is not
an exact discrete replacement by UG dot grad(alpha) for general fields.

This is a diagnostic nonconservative model, not gas-volume conservation.
Logs report sourceStep and sourceCumulative using the RK4 weights, and
stepBalance/cumulativeBalance subtract those source integrals. With the
switch off, sources are zero and the original conservative update remains.
gasDiag s3 still shows -alphaDG*div(UG), i.e. the compression term being
cancelled, not a source applied in the switched equation.

The Method1 pressure target continues to use the actual projected alpha
increment. Thus the altered alpha also affects the coupled pressure/velocity
response: an improvement implicates compression/coupling but does not alone
prove that div(UG) was the original cause. Compare fresh runs with identical
settings except this switch; existing DG restart limitations still apply.
