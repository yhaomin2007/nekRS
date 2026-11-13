# Method1: residual-based crosswind and SUPG

The default is `alphaStabilizationMode=6`: residual-dependent Codina-type
crosswind diffusion and transient SUPG, with coefficients lagged within each
timestep. No uniform artificial diffusivity is prescribed
in these modes. This implements the **approach** of consistent residual-based
stabilization offered by COMSOL, adapted to Method1's existing BDF/explicit
transport split. It is not a reproduction of COMSOL's internal solver. In particular, the crosswind coefficient is selected
from the native alpha predictor, then held fixed for a linear stabilized solve.
It is recomputed at every timestep. This is a semi-implicit adaptation, not
COMSOL's fully converged nonlinear crosswind discretization.

## Correspondence and scope

COMSOL documents consistent streamline stabilization (SUPG for scalar
advection–diffusion), nonlinear crosswind stabilization, a lower gradient limit,
and a polynomial-order reduction. Its transport interfaces offer Codina as an
alternative to Do Carmo–Galeão. **This module uses a Codina-type transverse
tensor, not COMSOL's default Do Carmo–Galeão/Hughes–Mallet operator.**

The streamline term acts on the transient transport residual, rather than
adding a constant longitudinal diffusivity. It uses the conservative SUPG test
perturbation `tau UG.grad(w)`; it does not add a reaction-test `div(UG)*w`
term. The time scale below is a disclosed BDF-step-based approximation, not
COMSOL's dynamic subgrid time-scale implementation or an exact GLS replica.

Native explicit convection, HPFRT, baseline implicit D0, boundary maps, clipping,
phase prediction/correction, momentum and pressure remain unchanged. The UDF
corrects only current alpha after the native scalar stage, before clipping.
It reads native forcing/history and applies the native Helmholtz operator;
it does not overwrite forcing/history, rerun scalar time integration, or change
NekRS core source files. Additional GPU memory and operator work are required.

## Discrete residual and weak equation

Let `h=g0/dt`, `B=sum_j coeffBDF[j]*alpha_old[j]/dt`, and `C_EXT` be the
already selected conservative negative transport RHS from the existing fluid
EXT history. Source-time UG is frozen for stabilization direction and tau.

```
R(alpha) = h alpha - B - C_EXT - div(D0 grad(alpha))
```

The diffusion in this sensor is a full **element-local strong** derivative.
It is not the assembled weak Helmholtz residual. It includes diffusion and
the nonzero gas divergence through conservative `C_EXT`. The added
stabilization itself is excluded from this baseline residual. HPFRT remains
in the original native RHS, and is not interpreted as physical alpha transport
in this sensor. Thus a nonzero R can also reflect the existing regularization.

With `b_native` the saved unassembled native alpha RHS, the equation solved is

```
H alpha + sum_e G_e^T W_e [Dcw_predictor Pperp grad(alpha) + tau UG R(alpha)] = b_native
H = h M + K(D0)
```

The new operator is assembled with the native weak divergence and
gather/scatter. Gradients used in the flux are local, divided by their own JW;
they are not averaged across unequal elements before forming the weak load.

This is SUPG of the **existing time-discrete IMEX transport equation**. Current
alpha is implicit in the BDF, diffusion and stabilization residual. Physical
convection is still extrapolated. It is not a fully implicit final-time
convection discretization. Original advective timestep restrictions remain.

## Coefficients

Evaluate the following coefficient formulas on the native alpha predictor,
using its BDF/EXT residual and gradients. For contravariant metric rows `a_r,a_s,a_t`, define

```
hmin = 2 / sqrt(sum_d |a_d|^2)
hgrad = 2 |grad(alpha)| / sqrt(sum_d (a_d.grad(alpha))^2)
hu = 2 |UG| / sqrt(sum_d (a_d.UG)^2)
```

Zero-gradient/velocity directions use `hmin`. With polynomial degree `p`,
`heff=hmin/p`, `hstream=hu/p`, and `r_p=(sqrt(2))^(1-p)`:

```
tau = Csupg / sqrt((2/dt)^2 + (2|UG|/hstream)^2
                   + (4D0/heff^2)^2 + div(UG)^2)
g_floor = delta_alpha / (hgrad r_p)
Pe_gate = max(0, 1 - 2D0/(|UG| hgrad))
Dcw = min(Dmax, Ccw r_p hgrad |R| Pe_gate
                / (2 max(|grad(alpha)|, g_floor)))
Pperp = I - UG UG^T/(|UG|^2 + epsilon_u^2)
```

`Pe_gate=0` when `|UG|<=epsilon_u`. Dcw is nonnegative, spatially varying,
updated at every timestep, and vanishes with the predictor residual. The gradient floor
regularizes the coefficient; it is not a minimum diffusivity. `Dmax` is a cap,
not a prescribed uniform value. Near zero velocity Pperp is regularized; the
gate also turns crosswind off. The metric lengths, tau and floor are NekRS
adaptations, disclosed here for reproducibility.

## Predictor-lagged linear correction

1. The native alpha Helmholtz result supplies a predictor.
2. Compute `R_predictor`, Dcw and tau once, and freeze the coefficients for this
   timestep. The gas velocity direction also remains frozen at source time.
3. Solve the stabilized equation for current alpha, using restarted GMRES(20).
   Crosswind is implicit with the frozen nonnegative coefficient; SUPG retains
   the current BDF/diffusion residual. No finite-difference Jacobian, Newton
   iteration, line search, or coefficient update occurs inside this solve.
4. Reevaluate the actual assembled equation residual after each GMRES cycle.
   Accept only when the true free-row RMS reaches the requested tolerance.
5. Route the final applied weak source to the Method1 divergence target once,
   then perform the existing alpha postprocessing/clipping.

For a free-row direction v, the homogeneous operator is applied directly:

```
A v = H v + sum_e G_e^T W_e [Dcw_predictor Pperp grad(v)
                           + tau UG (h v - div(D0 grad(v)))]
```

Known histories/transport are zero only in this operator action; they are
retained in the actual affine residual. Two-pass mass-weighted orthogonalization
and right scaling by h are used. No trial operator action modifies alpha or the
native forcing/history. All correction directions have zero Dirichlet rows.

The convergence norm is the volume-weighted RMS of the assembled residual
divided by assembled lumped mass, with Dirichlet rows zeroed. The target is
`AbsTol + RelTol * initialRms`. The implementation requires the existing unit
alpha transport coefficient, zero Neumann alpha loads, no Robin condition,
and no custom scalar implicit linear term.

The previous fully nonlinear wrapper differentiated a nonsmooth, strongly
state-dependent crosswind coefficient through the current BDF residual. A
sharp-front two-element problem stalled at RMS about 1.44e-3, including with a
separate dense nonlinear solver. Predictor-lagging removes that feedback; it
changes the coefficient time treatment rather than merely loosening a line
search. The regression now converges in two linear iterations (RMS about
2.47e-9), and also passes with a fixed nonzero alpha inlet.

This does not establish unconditional stability, bound preservation, or stripe
suppression. The coefficient lags the corrected solution within a timestep, so
accuracy must be assessed with timestep refinement. Linear nonconvergence still
stops with step, mode, iteration count, initial/true/projected residuals, target,
and dt. No unconverged alpha is silently accepted and no fallback turns off
stabilization.

## Controls

| CASEDATA key | Default | Meaning |
| --- | --- | --- |
| alphaStabilizationMode | 6 | 0 off; 2 residual crosswind; 5 SUPG; 6 both |
| alphaCrosswindScale | 1 | Dimensionless Ccw |
| alphaStreamlineScale | 1 | Dimensionless Csupg |
| alphaLowerGradientChange | 1e-5 | Alpha change delta_alpha for gradient floor |
| alphaMaxResidualDiffusivity | 1e-3 | Dcw cap, m²/s |
| alphaDirectionEpsilon | 1e-6 | Velocity regularization, m/s |
| alphaStabilizationRelTol | 1e-5 | Relative true linear RMS tolerance |
| alphaStabilizationAbsTol | 1e-7 | Absolute true linear RMS tolerance, 1/s |
| alphaStabilizationMaxIterations | 200 | Maximum total GMRES iterations, restarted every 20 |

Modes 1/3/4 retain the earlier constant-coefficient experiments, selected
explicitly and documented in `ALPHA_STABILIZATION.md`. The default par file
contains no constant crosswind or streamline diffusivity controls. Mode 0
bypasses the residual correction entirely. All native tolerances are retained.

## Boundaries and Method1 divergence target

No added surface load is supplied. Added weak flux has natural zero normal
boundary flux on free Neumann boundaries. Its reconstructed pointwise normal
flux need not vanish separately at every boundary node. Native inlet Dirichlet
rows replace their weak equations and can exchange gas volume through reactions.

The accepted **current** source `-M^-1 assemble(Kcw alpha + SUPG)` is routed
once into the existing Method1 alpha-diffusion contribution to the UM divergence
target. It is not extrapolated again or inserted into explicit source history.
Baseline D0 continues using the native pre-clipping alpha solution.

The raw added weak load annihilates a constant test and has zero integral to
roundoff before boundary row replacement. Crosswind alone has nonnegative
weak stiffness energy for fixed coefficients. SUPG is a residual term and does
not have the same simple diffusion energy identity. Final gas-volume balance
also includes advection, baseline diffusion, HPFRT, Dirichlet reactions and
retained clipping. This module does not guarantee positivity or exact final
gas-volume conservation, and does not remove the effect of clipping.

## Diagnostics

At the existing diagnostic interval, logs report total GMRES iterations and cycles,
initial/final/target free-row RMS, Dcw maximum, tau maximum, gas volume, net raw
stabilization and clipping volume change. The CSV is
`bubbleColumn_alpha_residual.csv`. A large fraction of values at Dmax indicates
that the cap is controlling the model; inspect the field before calibration.

Checkpoint `ars` contains the accepted **pre-clipping** linear solution evaluation
with the frozen source-time velocity; it is not reevaluated using subsequently
corrected phase velocities. `arsGrad` contains its assembled gradient vector.
The ordinary case output retains final postprocessed alpha. Dcw is selected
from s12 and predictor gradients; it is not recomputed from the final s9 field.
The existing CSV `iterations` column now means total linear GMRES iterations.

| VisIt field | Meaning |
| --- | --- |
| s1 | Accepted pre-clipping alpha |
| s2 | Source-time strong div(UG) |
| s3 | Source-time UG.grad(alpha) |
| s4 | alpha div(UG) |
| s5 | Dcw, m²/s |
| s6 | tau, seconds (not streamline diffusivity) |
| s7 | Assembled crosswind RHS, 1/s |
| s8 | Assembled SUPG RHS, 1/s |
| s9 | Local strong baseline R, 1/s |
| s10 | Total applied stabilization RHS, 1/s |
| s11 | Assembled mass-normalized free-row linear equation residual, 1/s |
| s12 | Native predictor strong residual used to select Dcw, 1/s |

## Verification and references

`tests/alpha_residual.py` executes the production residual kernels together
with real native Serial SEM gradients and weak divergences on unequal hexes.
It checks BDF/EXT residual signs, zero-residual consistency, variable
coefficients, high-order reduction, caps, directional switches, padded
offsets, conservation, crosswind energy, manufactured and sharp-front linear solves,
fixed coefficient checks, affine/homogeneous operator consistency, and CUDA
translation. `tests/alpha_linear_direction.py` executes the actual host GMRES
direction against a nonsymmetric operator, forcing multiple restart cycles and
checking true convergence and its fixed inlet component. The UDF also passes a header syntax check against the
native interfaces. These are not a full MPI/GPU column run or evidence that
the stripes are eliminated. Compare profiles, onset time, smearing, volume
budget and wall-clock cost at identical physical times.

- [COMSOL 6.4: Stabilization Techniques](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_modeling.19.090.html)
- [COMSOL: Crosswind Diffusion alternatives](https://doc.comsol.com/6.4/doc/com.comsol.help.cfd/cfd_ug_chemsptrans.11.141.html)
- [COMSOL: residual and lower-gradient settings](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_modeling.19.088.html)
- [Codina (1993): A discontinuity-capturing crosswind-dissipation for the finite element solution of the convection-diffusion equation](https://doi.org/10.1016/0045-7825(93)90213-H)
- [Marras et al.: high-order SEM residual-based stabilization, accepted manuscript](https://faculty.nps.edu/fxgirald/Homepage/Publications_files/Marras_et_al_JCP_accepted.pdf)
