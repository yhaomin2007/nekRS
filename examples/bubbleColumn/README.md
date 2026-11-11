# Method 1: native mixture solve with two-phase velocity correction

The active case transports seven scalars, in this fixed native order:

`ALPHA -> UGX/UGY/UGZ -> ULX/ULY/ULZ -> native pressure -> native UM -> phase correction`.

There is one native stage per step. No core source files are changed. The
phase predictors never overwrite UM before its pressure/velocity solve.
The final correction changes only the current phase scalar solutions; their
corrected values enter the normal native histories at the next step.

## Phase predictors

Alpha uses conservative cubature transport with UG. Both phase material
convection products use the same cubature operator and native fluid EXT
coefficients. Scalar EXT is a separate EXT1 carrier to avoid double
extrapolation. The pressure gradient is the assembled strong gradient of the
latest completed pressure; `gasPressureGradientExtrapolation=1` extrapolates
that gradient for **both phases**, independently of convection.

With a=alpha, b=1-a, r=a/max(b,alphaFloor), and c=Cvm*rhoL,
the local virtual-mass inertia block is

```text
M = [rhoG+c,    -c        ]
    [-r*c,     rhoL+r*c  ].
```

At source time, solve M*(aG,aL)= (fG,fL), where aG/aL are the
material accelerations, and

```text
fG = -grad(p) + rhoG*g + K*(UL-UG) + div(a*tauG)/max(a,alphaFloor)
fL = -grad(p) + rhoL*g - r*K*(UL-UG) + div(b*tauL)/max(b,alphaFloor).
```

Each scalar source is its resulting acceleration minus its own extrapolated
material convection. This replaces the old extrapolated reconstructed-liquid
acceleration path: UL now has its own momentum equation and symmetric VM
reaction. The 2x2 inertia inversion is pointwise and analytic; no Picard loop
or additional global solve is used.

Physical Newtonian stresses and Schiller–Naumann drag are **explicit**.
The local VM time inertia is inverted jointly. Native scalar Helmholtz solves
provide the additional numerical diffusion specified in the scalar sections;
this is separate from physical stress. Native HPFRT remains in those sections.
The physical stress/drag still impose explicit timestep restrictions.

## Mixture solve and final correction

The native UM momentum equation keeps its own convection, variable density,
mixture viscosity, gravity and drift-stress forcing. Pressure uses the
variable 1/rhoM coefficient and the alpha-derived relative-flux divergence
target, including alpha numerical diffusion. The final phase correction
does not modify UM, pressure, their histories, or their forcing.

For the completed alpha, define wg=a*rhoG/rhoM and wl=b*rhoL/rhoM.
The mismatch is R=UM-wg*UGpredict-wl*ULpredict.
Use the cheap local response of

```text
A = (g0/dt)*M + K*[ 1, -1 ]
                    [-r,  r ].
B = inverse(A)*(1,1).
deltaUG = BG*R/(wg*BG+wl*BL)
deltaUL = BL*R/(wg*BG+wl*BL).
```

Dirichlet phase rows are removed before computing B; prescribed velocities
are untouched. With at least one phase of nonzero mixture weight free,
wg*UG+wl*UL=UM to roundoff. At fully constrained nodes, incompatible boundary
data cannot be repaired by this correction: the reported residual remains.
At a phase-absent node, a free zero-weight phase alone cannot remove mismatch.

This is an **approximate splitting of the whole native UM mismatch**.
It is not a second pressure projection or an exact inverse of the spatial
phase momentum operators. In particular, viscosity has no diagonal
approximation in A. Exact mixture consistency does not establish small
individual phase PDE residuals or prove stripe suppression.

The inlet has prescribed alpha and UG, stationary UL, and the matching
mass-weighted native UM. Initial UL is zero; a tanh plume initializes alpha
and UG. Existing wall types are retained: UGX/UGY and all UL components
Dirichlet zero, UGZ zero Neumann; native UM is no-slip. The algebraic
correction preserves Dirichlet values. It does not independently enforce a
strong zero-normal-gradient constraint on corrected UGZ; Neumann conditions
enter the next native phase Helmholtz solve naturally. Use zero UGZ wall
Dirichlet if that is the intended wall model.

## Parameters and output

The default par exposes physical properties, inlet/plume data, bubble size,
VM coefficient, pressure-gradient EXT, alpha compression and diagnostic
interval. Solver tolerance, scalar numerical diffusion, HPFRT and BCs remain
in their native solver sections. Old velocity limiters, low-alpha masks,
outlet damping, user alpha filtering, divergence filtering/ramping/inlet
masking and old diagnostic selector options are no longer read by this UDF.
Their shared helper implementations remain for historical tests.

Alpha is clipped to [0,1] after its solve with denominator floor 1e-6.
The alpha-derived reduced divergence target omits HPFRT and clipping rates,
as in the previous Method 1; therefore this is not exact discrete phase
mass conservation including those numerical operations.
`alphaCompressionEnabled=0` remains a diagnostic switch that removes
alpha*div(UG), changes the equation and loses conservative alpha transport.

`diagnosticInterval` controls order traces and componentwise global min/max:
first step, every N steps and checkpoints (0 means first/checkpoints only).
Check `mixtureDefectBefore` and `mixtureDefectAfter`, especially at boundaries.
The diagnostic checkpoint files include:

| Prefix | Field |
| --- | --- |
| ug, ul | Final corrected phase velocities |
| deltaUg, deltaUl | End-of-step phase corrections, velocity units |
| mixBefore, mixAfter | Native UM minus mass-weighted phase velocities |
| gmr, lmr | Phase physical momentum defects on free degrees of freedom, acceleration units |
| gpr, ggr, gdr, gvs, gvm, gcv | Gas pressure, gravity, drag, physical viscosity, VM and convection terms |

Gas equation term snapshots are frozen at source assembly and reported in
acceleration units after division by rhoG. Their sum is the **physical
predictor RHS**, before native numerical diffusion/HPFRT and final correction.
The log's sourceTime distinguishes them from end-of-step velocities.
The old single-UG BDF/Helmholtz balance exports do not describe this scheme
and are not registered. The source snapshots are not final phase PDE residuals.

The separate gmr/lmr diagnostics compute M*(D_BDF U_final - RHS_predictor)
plus the strong pressure-gradient change from source time to the final native
pressure, then divide the gas/liquid rows by rhoG/rhoL. They use the **frozen
source-time alpha and physical spatial terms**, the actual variable-step
native BDF coefficients and corrected final velocities. Weighted BDF histories
are saved before native lagging discards the oldest slot. Dirichlet rows are
zeroed. Scalar numerical diffusion/HPFRT and the final correction are not
subtracted: their momentum effects remain visible in these physical defects.
This is a residual of the chosen frozen-spatial-term time split, not a new
evaluation of fully nonlinear final-time phase equations.

Start a fresh run or use a checkpoint containing all seven scalars.
Old four-scalar checkpoints cannot initialize the new independent UL history.

## Validation

Run `python3 examples/bubbleColumn/tests/phase_momentum.py` for the actual
kernel algebra, pressure/drag/VM signs, bounded-density edge cases, stiff
response, partially constrained boundaries and unchanged native UM.
Set OCCA_DIR to a built OCCA checkout to additionally build and execute the
actual OCCA Serial kernels and translate them for CUDA. The other tests cover
the retained alpha transport/time-extrapolation and SEM utility operators;
several retained modules are historical and are not active UDF hooks.
METHOD1_AUDIT.md and the old OneThrough/UGTerms helpers describe the previous
four-scalar scheme; the active hooks are in bubbleColumnPhaseMomentum.hpp.

A full MPI/GPU column run requires the original mesh and a configured NekRS
installation. Serial algebra, kernel builds and header syntax checks do not
establish long-time stability of this new split.
