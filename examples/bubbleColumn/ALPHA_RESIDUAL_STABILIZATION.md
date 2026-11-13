# Method1: residual-based crosswind and SUPG

The default is `alphaStabilizationMode=6`: nonlinear Codina-type crosswind
diffusion and transient SUPG. No uniform artificial diffusivity is prescribed
in these modes. This implements the **approach** of consistent residual-based
stabilization offered by COMSOL, adapted to Method1's existing BDF/explicit
transport split. It is not a reproduction of COMSOL's internal solver.

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

With `b_native` the saved unassembled native alpha RHS, the nonlinear equation is

```
H alpha + sum_e G_e^T W_e [Dcw Pperp grad(alpha) + tau UG R(alpha)] = b_native
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

For contravariant metric rows `a_r,a_s,a_t`, define

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
updated at every nonlinear evaluation, and vanishes with R. The gradient floor
regularizes the coefficient; it is not a minimum diffusivity. `Dmax` is a cap,
not a prescribed uniform value. Near zero velocity Pperp is regularized; the
gate also turns crosswind off. The metric lengths, tau and floor are NekRS
adaptations, disclosed here for reproducibility.

## Nonlinear correction

The native alpha result seeds a damped matrix-free Newton solve of the full
stabilized weak equation on free rows. Each direction uses up to 20 GMRES
vectors, two-pass mass-weighted orthogonalization, and right scaling by h.
The Jacobian action is a finite difference of the complete residual. GMRES
targets 5% of the current nonlinear residual. A backtracking line search
requires residual descent or attainment of the nonlinear tolerance.

The convergence norm is the volume-weighted RMS of the assembled residual
divided by assembled lumped mass, with Dirichlet rows zeroed. The target is
`AbsTol + RelTol * initialRms`. Native inlet alpha values are held fixed by
zero-row corrections. No extra Dirichlet boundary callbacks are invoked.
The implementation requires the existing unit alpha transport coefficient,
no Robin condition, and no custom scalar implicit linear term.

Nonlinear stabilization can fail to converge in strongly under-resolved
conditions; it is not unconditionally stable or bound preserving. Failure to
find a descent step or meet the tolerance stops with an explicit error rather
than accepting an unconverged stabilized equation. Reduce dt or stabilization
scales and inspect the exported residual if this occurs. A deliberately sharp
two-element stress problem exposed this difficulty; successful manufactured
tests do not establish robustness for the bubble column.

## Controls

| CASEDATA key | Default | Meaning |
| --- | --- | --- |
| alphaStabilizationMode | 6 | 0 off; 2 residual crosswind; 5 SUPG; 6 both |
| alphaCrosswindScale | 1 | Dimensionless Ccw |
| alphaStreamlineScale | 1 | Dimensionless Csupg |
| alphaLowerGradientChange | 1e-5 | Alpha change delta_alpha for gradient floor |
| alphaMaxResidualDiffusivity | 1e-3 | Dcw cap, m²/s |
| alphaDirectionEpsilon | 1e-6 | Velocity regularization, m/s |
| alphaStabilizationRelTol | 1e-5 | Relative nonlinear RMS tolerance |
| alphaStabilizationAbsTol | 1e-7 | Absolute nonlinear RMS tolerance, 1/s |
| alphaStabilizationMaxIterations | 30 | Maximum Newton iterations |

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

At the existing diagnostic interval, logs report Newton and GMRES iterations,
initial/final/target free-row RMS, Dcw maximum, tau maximum, gas volume, net raw
stabilization and clipping volume change. The CSV is
`bubbleColumn_alpha_residual.csv`. A large fraction of values at Dmax indicates
that the cap is controlling the model; inspect the field before calibration.

Checkpoint `ars` contains the accepted **pre-clipping** nonlinear evaluation
with the frozen source-time velocity; it is not reevaluated using subsequently
corrected phase velocities. `arsGrad` contains its assembled gradient vector.
The ordinary case output retains final postprocessed alpha.

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
| s11 | Assembled mass-normalized free-row nonlinear residual, 1/s |

## Verification and references

`tests/alpha_residual.py` executes the production residual kernels together
with real native Serial SEM gradients and weak divergences on unequal hexes.
It checks BDF/EXT residual signs, zero-residual consistency, variable
coefficients, high-order reduction, caps, directional switches, padded
offsets, conservation, crosswind energy, a manufactured Newton/GMRES solve,
and CUDA translation. `tests/alpha_newton_direction.py` executes the actual
host GMRES direction against a nonsymmetric nonlinear Jacobian, checking its
linear tolerance and fixed inlet component. The UDF also passes a header syntax check against the
native interfaces. These are not a full MPI/GPU column run or evidence that
the stripes are eliminated. Compare profiles, onset time, smearing, volume
budget and wall-clock cost at identical physical times.

- [COMSOL 6.4: Stabilization Techniques](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_modeling.19.090.html)
- [COMSOL: Crosswind Diffusion alternatives](https://doc.comsol.com/6.4/doc/com.comsol.help.cfd/cfd_ug_chemsptrans.11.141.html)
- [COMSOL: residual and lower-gradient settings](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_modeling.19.088.html)
- [Codina (1993): A discontinuity-capturing crosswind-dissipation for the finite element solution of the convection-diffusion equation](https://doi.org/10.1016/0045-7825(93)90213-H)
- [Marras et al.: high-order SEM residual-based stabilization, accepted manuscript](https://faculty.nps.edu/fxgirald/Homepage/Publications_files/Marras_et_al_JCP_accepted.pdf)
