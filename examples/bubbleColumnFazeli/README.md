# Fazeli bubble column with Method1

Independent physical-benchmark adaptation of Fazeli, Rhebergen & Abukhdeir,
*Laminar dispersion force effects on two-fluid modelling and simulation of
bubble column hydrodynamics*, IJMF **197** (2026), 105590,
[DOI: 10.1016/j.ijmultiphaseflow.2025.105590](https://doi.org/10.1016/j.ijmultiphaseflow.2025.105590).
Parameters and boundary conditions follow Section 5, Table 2, Fig. 4 and
Eqs. (3)–(10), (16)–(18). The authors' public
[NGSolve solver](https://zenodo.org/records/18135226) is a secondary implementation
reference; differences between its sample inputs and the final paper are
identified below.

This case is self-contained and leaves `examples/bubbleColumn` unchanged.
Its Method1 snapshot comes from commit `7bdd776c`: **UG → alpha with new UG →
native pressure/UM**. UL is reconstructed; no liquid scalar solve, reset of UM,
final phase correction, Picard iteration, or NekRS core changes are introduced.
The paper uses a linearly lagged backward-Euler IP-DG formulation. This case
uses continuous SEM and Method1 splitting, so it is **not a numerical
reproduction or a validated match to the paper's results**.

## Run

From this directory, with the existing branch-compatible NekRS v26 installation:

```bash
# Restore the included 9000-element mesh (or run python3 generate_mesh.py).
gzip -dk bubbleColumnFazeli.re2.gz
mpirun -np 1 nekrs --setup bubbleColumnFazeli
```

The included compressed mesh and the Python generator produce identical RE2
bytes. The generator needs only Python's standard library. To refine all three
directions by two, run `python3 generate_mesh.py --refine 2`; this overwrites the
local RE2 with 72000 elements. The source box stays aligned. Do not compare
meshes without also checking the time step.

Start with `endTime = 0.1` or `1` for a local smoke run. The delivered reference
configuration requests 250 s, `dt = 1e-4` s, first-order time integration,
polynomial order 3 and cubature order 6. These are starting SEM settings, not
the paper's mesh or a demonstrated converged discretization. Native UM
convection remains part of the native pressure/velocity solve; all phase
convection is supplied by the existing Method1 cubature implementation.

## Physical configuration

All quantities are dimensional SI. The paper's vertical coordinate is y; this
case rotates it to **z**: reference (x,y,z) corresponds to NekRS (x,z,y).

| Quantity | Value |
| --- | --- |
| Liquid-domain width × depth × height | 0.20 × 0.05 × 0.45 m |
| Water / gas density | 1000 / 1.2 kg/m³ |
| Water dynamic viscosity | 8.9e-4 Pa·s |
| Surface tension | 0.072 N/m |
| Bubble diameter | 0.003 m |
| Gravity | (0,0,-9.81) m/s² |
| Gas injection | 48 L/h = 1.333333333333e-5 m³/s |
| Drag | Tomiyama Eq. (3), C=16 as in the public solver |
| Lift | Tomiyama Eqs. (4),(5), evaluated from local slip |
| Laminar dispersion | Modified Panicker MP-LD, Cdis=3.2 |
| Virtual mass | Cvm=0.5 |
| Turbulence model | None |

The nominal glass container height is 0.50 m; only the 0.45 m water region is
simulated. Free-surface displacement and the air headspace are omitted.
The calculated initial/injection slip speed is obtained from Eq. (16) by
solving the force balance with the same Tomiyama drag as the gas equation.
It is printed at startup; it is not forced to the approximate experimental
0.2 m/s quoted in the text.

The drag uses finite `Cd*Re`, including the zero-slip limit. The public solver
uses C=16 despite the paper's discussion of contaminated water. That choice
is stated explicitly here. At the nominal terminal slip, the deformation
branch controls the correlation. The lift's horizontal diameter follows the
formula **as printed below Eq. (5)**, `dH=dp*(1+0.163*Eo^0.757)`; the public
sample uses Eo directly for lift. At these 3 mm conditions, the low-Eo lift
branch is capped by `0.288*tanh(0.121*Re)` in either implementation. Larger
bubbles would require resolving this convention before comparison.

## Injection-zone ambiguity

The paper states Q=48 L/h and S=0.3704 s⁻¹ but does not specify all source-box
extents unambiguously. Those values imply a volume of approximately 3.6e-5 m³.
The public `3D_sample.geo` instead defines a 0.04 × 0.02 × 0.05 m box, whose
4e-5 m³ volume would give 53.3376 L/h at S=0.3704. It cannot satisfy both
published numbers simultaneously.

The delivered **explicit geometric assumption** is the centred box

```
x = [0.08, 0.12] m
y = [0.01, 0.04] m
z = [0.01, 0.04] m
```

This yields 3.6e-5 m³ and S=0.37037037037 s⁻¹, matching the rounded paper rate.
These extents are inferred, not claimed to be a recovered original mesh.
They are editable in `[CASEDATA]`; the runtime rejects a non-aligned source
box or incorrect discrete volume. To use the public sample's source geometry
after the coordinate rotation instead, set `sourceYMin=0`, `sourceYMax=0.05`,
`sourceZMin=0.01`, `sourceZMax=0.03`. The code then uses S=1/3 s⁻¹ to preserve
48 L/h.

The element-box indicator is mass-projected to shared CG nodes and normalized
by its discrete integral, so the **assembled volume source integrates to Q**.
Source boundary nodes are smoothed by this projection; the CG representation
is not an exactly discontinuous indicator at the box surface.

## Initial and boundary conditions

Initially alpha=0, UL=UM=0 and UG=(0,0,Uslip). Pressure is initialized to
`rhoL*g*(0.45-z)`, with zero top gauge pressure. Gas is introduced by the
internal source from t=0; there is no bottom gas inlet or inlet ramp.

| Boundary ID | Location | alpha | UG | UM / reconstructed UL |
| --- | --- | --- | --- | --- |
| 1 | Bottom | 0 | (0,0,Uslip) | UM=0, hence UL=0 |
| 2 | Top | zero normal gradient | zero normal gradient | Native zero-Neumann velocity, zero gauge pressure |
| 3 | Four vertical walls | zero normal gradient | (0,0,Uslip) | UM=alpha*rhoG*UG/rhoM, hence UL=0 |

**The paper prescribes nonzero vertical gas velocity on walls.** The wall
value therefore differs from the zero-UG wall experiments in the original
case. The UDF refreshes native UM Dirichlet data after solving alpha so the
mass-weighted wall value uses the current alpha.

The top is a native open-boundary approximation: pressure zero and homogeneous
velocity Neumann data. It does not reproduce the paper's entire DG
phase-specific advective traction boundary operator, especially under backflow.
The explicit liquid stress uses zero natural top traction.

## Source and stress consistency

`referenceCase.hpp/.okl` adds the following terms to the Method1 snapshot:

\[
\partial_t\alpha+\nabla\cdot(\alpha u_g)=S,
\qquad
\partial_t\rho_m+\nabla\cdot(\rho_m u_m)=\rho_g S.
\]

The explicit target passed to the native pressure solver is therefore

\[
\nabla\cdot u_m =
\frac{(\rho_l-\rho_g)
 [\mathrm{BDF}(\alpha)+u_m^{\mathrm{EXT}}\cdot\nabla\alpha]
 +\rho_g S}{\rho_m}.
\]

Only `rhoG*S/rhoM` is added to the old target: BDF(alpha) already contains the
volume injection. Omitting that mass term, or adding another complete S,
would be inconsistent.

Gas primitive momentum receives
`rhoG*S*(Uinj-UG)/(alpha*(rhoG+Cvm*rhoL))`. Its injection relaxation is
semi-implicit, with alpha protected by the inherited 1e-6 denominator floor;
this avoids a singular explicit time-step restriction at startup. **Drag
remains explicit.** Mixture acceleration receives
`rhoG*S*(Uinj-UM)/rhoM`. The continuous terms recover the paper's conservative
injected mass and momentum; the phase/mixture time levels remain split.

Brennen's mixture viscous term is `div(tauL)`, with Newtonian stress formed
from reconstructed UL. It is supplied explicitly as a weak SEM load
`-integral grad(w):tauL`, assembled and mass-inverted. The native UM viscosity
is consequently **zero**, to avoid counting a second, approximate UM stress.
This does not remove the liquid viscosity: `muLiquid` still controls tauL and
drag. No gas molecular viscous stress is present in this reference model.
Explicit viscous, drag and LD/acoustic terms still restrict the usable dt.

Alpha and gas scalar numerical diffusion are zero by default; no HPFRT, AVM,
SVV, added alpha stabilizer, alpha clipping, or divergence filter is enabled.
Constitutive denominators retain the existing Method1 safeguards. The case
is a controlled reference experiment, not a promise that unstabilized
continuous SEM will inherit IP-DG stability. If numerical diffusion or a
filter is added, record it as a changed numerical experiment.

## Diagnostics and comparison

The normal checkpoint holds pressure, UM and alpha/UG scalars. The single
additional `diag` dataset retains the requested fields:

| Export | Meaning |
| --- | --- |
| velocity x/y/z | Latest strong pressure-gradient components |
| s1 | Target div(UM), captured before native fluid solve |
| s2 | Actual strong div(UM), evaluated after native fluid solve |

The log prints source volume, normalized flow, computed slip speed, alpha
min/max, gas volume and top gas outflow. For the first transient, compare gas
volume with `Q*t` before appreciable outflow. For developed-flow comparisons,
the paper uses averages over 30–250 s; reconstruct UL from checkpoints using
`UL=(rhoM*UM-rhoG*alpha*UG)/((1-alpha)*rhoL)` before comparing its liquid
velocity profiles. No time-averaging module is automatically enabled here.

Use `laminarDispersionEnabled=0` for the paper's TFM-noLD comparison and keep
all other settings identical. Evaluate timestep and mesh convergence before
interpreting stripe suppression as a physical improvement.

## Local checks

```bash
python3 tests/mesh.py
python3 tests/physics.py
# Optional: OCCA_DIR=/path/to/built/occa python3 tests/physics.py
```

Checks cover RE2 decoding, positive affine Jacobians, interior face pairing,
boundary IDs/areas, source volume at two refinements, production source and
momentum kernels, source-aware divergence, hydrostatic initialization, stress
signs, and MP-LD/Tomiyama limiting behaviour. Optional OCCA checks compile and
execute kernels in Serial and translate both changed OKL files for CUDA.
Header syntax has also been checked against the branch API. A full NekRS
MPI/GPU startup and a long bubble-column simulation have **not** been run in
the development environment; stability and agreement with experimental/DG
results remain to be established on the target machine.
