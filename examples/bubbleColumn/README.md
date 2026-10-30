# Density-averaged method1: conservative UG transport of alpha

Alpha and UG have no native mixture convection or matching RHS cancellation: their device-only advection compute flags are disabled before the first timestep. Host compute stays enabled, retaining all four Helmholtz solves, diffusion, HPFRT and boundary conditions. Native alpha/UG advection histories are zero-initialized and remain zero. The alpha RHS directly supplies conservative div(alpha*UG): alphaTransportForm=1 forms the product at GLL nodes then applies assembled opSEM::strongDivergence; alphaTransportForm=2 retains conservative cubature differentiation (default). Both use the same frozen UG and time treatment. The former aliases the nodal product; neither form guarantees boundedness. UGX/UGY/UGZ receive only the user cubature -UG.grad(UG) convection in their explicit RHS. UG remains transported velocity, not QG. Gas drag and effective virtual-mass inertia remain semi-implicit; pressure and liquid acceleration remain lagged.

For alphaTransportForm=2, set cubaturePolynomialOrder >= 2*N because the conservative cubature kernel differentiates the sampled product. With alphaTransportForm=1 this extra requirement is lifted; native/UG cubature still requires sufficient quadrature for its own operator. The example N=7 uses order 14 (15 Gauss points). Scalar explicit histories use EXT1; native fluid/BDF integration is unchanged.

Fresh starts have only the axial plume: f(z)=0.5*(1-tanh((z-H)/delta)), alpha=alphaInlet*f, UGz=gasInitialVelocity*f. Initial liquid is stationary and mixture velocity follows the density-weighted relation. initialPlumeHeight and initialPlumeThickness remain configurable. Restart files take precedence. Uniform initialization selectors were removed.

Inlet alpha and UG are uniform (alphaInlet and gasInletVelocity). Shared inlet/wall rim nodes retain wall precedence for velocity. The tapered inlet option was removed.

UG reconstruction and pressure/drag/gravity/VM sources are not switched off at low alpha. Division by alpha/beta remains protected with alphaFloor; magnitude clipping is a separate optional control. The old low-alpha cutoff and smooth-mask par options no longer control this case.

## Mixture divergence

mixtureDivergenceEnabled=1 supplies a reduced target using alpha conservative transport and numerical diffusion only; 0 imposes zero divergence. divergenceRampEnabled and divergenceRampSteps retain the cosine startup ramp. Optional divergence filtering remains available.

`divergenceInletThickness = delta` applies a spatial inlet weight to the final pressure target, assuming the inlet is at `z=0`: zero for `z <= delta/2`, `0.5*(1-cos(pi*(2*z/delta-1)))` for `delta/2 < z < delta`, and one for `z >= delta`. Both value and first derivative match at the transition endpoints. Set the thickness to `0` to disable the layer (the default for older par files). The example uses `0.01` mesh units: a zero region up to `z=0.005`, followed by smooth recovery to `z=0.01`. The weight is applied after filtering, extrapolation and temporal ramp on every pass, including Picard correctors. Raw divergence comparisons remain unchanged; exported applied target includes this weight. This is a diagnostic modification of mixture continuity, not a correction of the underlying boundary reconstruction.

Divergence uses the standalone user module explicitFilter.hpp / explicitFilter.okl, ported from the supplied explicitFilterApply. It uses the original Legendre bubble basis (1, x, P_k-P_(k-2)), whose higher modes vanish at endpoints. divergenceFilterModes selects 1..N-1 high modes; divergenceFilterStrength in [0,1] sets their parabolic attenuation. The module directly applies F tensor F tensor F in-place to raw divergence. It preserves element corner values and compatible face traces; this is not the former ordinary-Legendre filter. Strength 0 is identity. Filtered divergence is assembled before optional extrapolation and ramp and supplied to pressure; raw comparisons stay unfiltered.

To reuse the module in another v26 case, include explicitFilter.hpp, call userExplicitFilter::registerKernels(kernelInfo) during UDF_LoadKernels, then call userExplicitFilter::explicitFilterApply(tag, mesh, modes, strength, field). Register the kernel for the case polynomial degree; fields may be padded or scalar slices with at least mesh Nlocal entries. The module performs element-local filtering only; the caller handles gather-scatter as needed. Matrices are cached by tag and rebuilt when mesh, degree, modes or strength changes.

The reduced alpha RHS is `R_transport_diff = EXT[-div_h(alpha*UG) + explicit_diffusion_compensation] - M^-1 K_D alpha_solved`. It is not recovered from native o_JwF. The user source caches alphaSource after optional subtractAlphaDiffusion compensation, before native HPFRT/forcing assembly. A separate user history shifts once per timestep and uses the same scalar EXT coefficients (EXT1 by default). Post-solve property evaluations do not overwrite it. With subtractAlphaDiffusion=0, only conservative gas transport and the implicit numerical diffusion remain.

Native weak diffusion uses the original implicit alpha solution, before explicit filtering/clipping. No HPFRT source, filter/clipping-rate correction, or Dirichlet-node BDF override is included. This does not disable these operations in the alpha solve itself: they still affect the evolved fields, but their equivalent rates are omitted from this divergence RHS. The weak diffusion operator is evaluated at boundary nodes as well; the former boundary temporal override is removed.

The raw target is `d_raw=(rhoLiquid-rhoGas)/rhoM * (R_transport_diff + A_UMused(alpha_final))`. The separate enabled user advection mask evaluates the mixture material term on final alpha even though native alpha/UG transport is disabled. UMused is the step-start advector used for fluid relUrst, not the newly pressure-corrected velocity. The density prefactor uses final alpha with its usual density clamp.

The full BDF alternative remains an exported diagnostic only; no par selector chooses it. Div s1 is the reduced transport/diffusion target, s2 is the full BDF target, s3 is the actual filtered/extrapolated/ramped pressure target, and s4 is s1-s2. In particular, s1 and s2 need not agree: s2 includes the time changes caused by HPFRT, alpha postprocessing and imposed boundaries. Their difference is no longer solely a numerical RHS-recovery error. The s5 velocity-lag diagnostic is unchanged.

This reduced target deliberately does not enforce the full discrete density continuity identity for all alpha regularization/postprocessing changes. Divergence filtering, startup ramp and optional history extrapolation can further change the target. Pressure still does not correct UG within the same timestep.

## UG pressure-gradient explicit filter

Set gasPressureGradientExplicitFilterModes and gasPressureGradientExplicitFilterStrength to apply the standalone bubble-basis filter separately to all three selected pressure-gradient components. The example uses one mode at strength 0.05; strength 0 disables it. Sequence: compute raw strong/weak gradients, copy the selected field, spatial filter, shared-node assembly, optional existing temporal relaxation, then optional inlet mask. Raw diagnostic gradients stay unchanged. Each evaluation starts from raw gradients, preventing repeated filter application to cached history. This filters the gradient entering UG forcing only; it does not filter pressure or the native mixture momentum pressure term.

## Alpha explicit filter

alphaExplicitFilterModes and alphaExplicitFilterStrength configure the same standalone bubble-basis module on alpha after each scalar solve, before density/divergence/pressure evaluation. Strength 0 disables it; the example enables one mode at strength 0.05. Existing HPFRT regularization remains independent and can be combined with this filter. The sequence is solve alpha, explicit filter, shared-node assembly, reapply prescribed scalar boundaries, then optional clipping. UG is not explicitly filtered.

Divergence uses weak diffusion on the original implicit alpha solution; the computed g0/dt postprocessing-rate diagnostic is excluded from its RHS. Conservation CSV records filter and clipping volume changes separately and accounts for both in corrected cumulative balances. The filter is not a positivity limiter or generally volume-preserving on curved elements. Filtering acts once per scalar advancement; in the current one-pass case this is once per timestep.

## Verification

Serial checks cover the reduced transport/diffusion RHS with EXT1/EXT2, its predicted difference from full BDF when HPF/filter/clipping act, and source history management; masked native alpha/UG advection and direct gas RHS; gas/VM scaling and dilute forcing; plume initialization; uniform/rim boundary consistency; diagnostic kernels and conservative transport. Full GPU/MPI integration must be checked in the user's NekRS installation. Core files are not modified.

METHOD1_AUDIT.md records the earlier pre-simplification snapshot and is retained as historical analysis.

Obsolete alpha native/cancellation comparison columns were removed from conservation CSV. Mixture material advection remains in divergence reconstruction, not in the alpha transport RHS; UM also continues to affect solved UG through physical coupling.

## Source-sign checks

The source_signs.py regression exercises actual case kernels with physical limiting cases: hydrostatic liquid pressure must accelerate gas upward; implicit drag must reduce slip energy; common phase acceleration must be unchanged by VM; gas viscous stress must damp a sinusoidal velocity mode; positive drift-stress divergence must enter mixture momentum negatively; an upward moving decreasing-alpha front must generate positive mixture expansion. It also checks the native pressure target and velocity-RHS sign convention against an analytic expansion correction. These local tests do not establish stability or correctness of the complete GPU/MPI pressure solve and lagged two-phase coupling.

## Continuity coupling lag diagnostic

The div output adds scalar04 (VisIt s5), in 1/s:
`delta_d = (rhoLiquid-rhoGas)/rhoM * (UM_used-UM_new) dot strongGrad(alpha_new)`.
UM_used is copied at userSource entry from the actual fluid o_U used by
nrs_t::computeUrst to construct o_relUrst. In the current implementation this
is the step-start velocity, not fluid o_Ue. UM_new is the completed fluid solve
before the optional emergency velocity limiter. Alpha is the current final
postprocessed scalar; density uses the same [0,1] alpha clamp as the raw target.
The gradient is a fresh assembled opSEM::strongGrad; this diagnostic has no
filter, ramp, or feedback into the equations and is computed even with the
mixture divergence target disabled. Initial output is zero.

At divergenceComparisonInterval, the log line divUmCouplingLag and
bubbleColumn_divergence_coupling.csv report global min/max, volume-weighted
mean and RMS. Compare its magnitude and location with the existing raw target
s1 and applied target s3. Large values identify a potentially significant
lagged-velocity contribution, but do not establish causality. This nodal
continuum-form diagnostic is not the exact difference of native cubature
advection operators or the measured final divergence/projection residual.

## Mixture divergence formula selector

`mixtureDivergenceForm=1` retains the previous reduced material-alpha formula:
`d=(rhoL-rhoG)/rhoM * (EXT[-div(alpha*UG)+diffusion_compensation] + diffusion_implicit + A_UMused(alpha_final))`.

`mixtureDivergenceForm=2` (example par setting) uses the relative conservative flux:
`d=(rhoL-rhoG)/rhoL * (EXT[div(alpha*(UMused-UG))+diffusion_compensation] + diffusion_implicit)`.
The C++ default is 1 for existing par files. Only values 1 and 2 are accepted.
The relative flux is constructed once at the alpha source time using the same
alpha, step-start UM, and frozen UG time level. It follows alphaTransportForm:
1 uses assembled nodal-product SEM divergence; 2 uses conservative cubature.
The saved source receives the same scalar EXT coefficients as alpha transport.
Native weak implicit diffusion still uses the pre-filter/clipping solved alpha.
Configured explicit alpha diffusion compensation is included; HPF/filter/
clipping-rate sources and boundary BDF overrides remain excluded.

Form 2 replaces the separate gas-transport and UM-material-advection terms;
it does not add an extra UM.grad(alpha) term or use rhoM in its prefactor.
Since form 1 uses final alpha in the mixture material term while form 2 keeps
the entire relative flux at one explicit time level, they are not identical
discrete/time-level rearrangements. Neither is an implicit pressure corrector.
Filtering, assembly, history extrapolation, startup ramp and the master switch
are unchanged. Div s1 now means the selected formula's raw target; s2 remains
the full BDF comparison, s3 the applied target, s4=s1-s2 and s5 the original
nodal material-advection lag diagnostic (not a complete form-2 residual).

## User-side block Picard iteration

`bubbleColumnPicard.hpp` composes the existing source, post-scalar, properties,
pressure and accepted-step callbacks using v26's inner-step loop. No core
files are changed. The example par selects coupled mode:

```ini
picardMode = 2
innerIterations = 6
picardAcceptAtCap = 0
picardRelativeTolerance = 1e-6
picardAlphaTolerance = 1e-8
picardUgTolerance = 1e-7
picardUmTolerance = 1e-7
picardPressureTolerance = 1e-3
picardDivergenceTolerance = 1e-6
picardRepeatTolerance = 1e-9
```

Modes:

- 0: old one-pass path; no iteration hooks are installed. This is also the C++
  default for existing par files without picardMode.
- 1: frozen-repeat control. Every pass solves from the same starting fields,
  coefficients, pressure-filter cache and native forcing. All four scalar
  fields, pressure and applied divergence are compared with pass 1.
- 2: the native first pass is the predictor. Later passes replace scalar user
  forcing with current-iterate conservative alpha transport, UG material
  convection and gas forces. UM physical force is refreshed after the scalar
  block. UM native convection and HPF retain their original EXT treatment.

One pass (innerIterations=1) performs no corrector. For comparison the example
sets all initialGuess entries to previous; iteration modes also select these
before elliptic setup. An old par with projection/extrapolated linear guesses
may differ by linear-solve error when switching to iteration. Compare matching
linear settings and choose residual tolerances tight enough to resolve the
requested Picard changes; outer iterate convergence is not linear accuracy.

The scalar base load is `native_JwF - M_LMM*EXT(original_user_scalar_source)`.
A corrector adds `M_LMM*current_user_scalar_source`, without EXT of the current
nonlinear product. The fluid base is
`native_JwF - M_Jw*EXT(original_user_mixture_force)`; it retains BDF history,
native EXT convection and HPF. preFluid adds `M_Jw*current_mixture_force` using
new alpha/UG and the preceding UM iterate. Native sumMakef's fluid load is an
acceleration load; no extra rho factor is added here. The native pressure and
velocity solvers use refreshed mixture coefficients, including variable 1/rho
in pressure. Scalars keep transportCoeff=1 and semi-implicit drag. Native
predictors (including native predictor-dependent viscous terms) remain fixed.
This is a user-side nonlinear RHS fixed-point iteration, not an Oseen/Newton
operator and not a fully implicit treatment of UM convection.

Each complete preceding iterate supplies alpha/UG transport, gas pressure,
drag and liquid acceleration for the next four-scalar block. UG does not use
new alpha from within that same scalar block. VM acceleration uses
`(UL_iter-UL_old)/dt + UL_iter.grad(UL_iter)`, retaining the case's first-order
physical-time definition and a fixed UL_old. The helper restores UL_old after
each evaluation; a Picard difference is never divided by dt as acceleration.
Pressure-gradient temporal relaxation uses the same old-time filtered anchor
for every pass; spatial filtering and the inlet mask remain active. The
accepted pressure gradient is committed from the final pressure once.

After each fresh scalar solve, configured alpha filtering/clipping and gas
limiting are applied. Per-pass cumulative counters start from the same old
step totals, so only the accepted corrections remain accumulated. Properties
and the selected divergence are updated. Form 2 uses current alpha/UG with the
preceding UM iterate in `div(alpha*(UM-UG))`; form 1 uses a separate current-UM
contravariant buffer for its material term. Neither overwrites native fluid
relUrst. Numerical diffusion remains on the pre-filter implicit alpha solution;
HPF/filter/clipping-rate sources remain excluded from divergence. Divergence
filtering, assembly, ramp and the master switch are retained. Coupled mode
requires divergenceExtrapolationEnabled=0 so a fresh target reaches pressure.

Native BDF/ADV/EXT histories shift only during normal physical-step entry.
Separate UDF histories track the ORIGINAL predictor user sources solely to
remove their exact EXT contribution from native forcing. They do not substitute
accepted corrector samples into native history; the native predictor retains
its historical predictor-source convention. Scalar and fluid EXT arrays can
have different orders/coefficients. Correctors never call initStep,
makeForcing, lagSolution or native makeAdvection. The existing alpha transport
history is not overwritten by residual or corrector evaluations.

Only accepted steps execute the parent UDF end-step work: divergence history,
VM history, conservation histories, checkpoint exports and final UM limiting.
The original checkpoint request survives intermediate suppression. As in the
old case, the emergency UM limiter acts AFTER the convergence callback; a
limited exported state need not satisfy the measured pre-limit constraint.
Restart loading uses normal native startup plus freshly initialized UDF source
histories; these are transient arrays, not additional restart variables. The
first native startup coefficients select only available samples.

`bubbleColumn_picard.csv` and `bubblePicard` log lines report separate field
changes, a scaled change and immutable-history errors. Each field is scaled by
`absoluteTolerance + relativeTolerance * max|previous field|`; all fields and
the applied divergence must have scaled change <=1 after at least two passes.
Statuses distinguish predictor, iterating, converged and maxPasses. A strict
cap stops an unconverged step; picardAcceptAtCap=1 explicitly permits acceptance
with maxPasses. Frozen mode uses an absolute repeatTolerance and must run the
requested number of passes. No within-step dt reduction is performed.

The logged strongDivMismatchRms compares an assembled SEM strong divergence
of the completed UM with its applied target. It is a history-neutral spatial
diagnostic, NOT an exact native pressure-projection or full nonlinear residual,
and is not used as a convergence criterion. BDF density/gas-balance diagnostics
and raw-vs-applied divergence remain separate. In coupled mode s1-s2 can
contain both excluded regularization effects and incomplete nonlinear
convergence; it is not merely an RHS recovery error. Picard convergence does not
restore the regularization terms deliberately excluded from continuity.

Restrictions: fixed mesh, no advection subcycling, four transported gas
scalars, no density splitting, thermodynamic low-Mach, constant-flow correction
or NekNek coupling. Existing unrelated preFluid/convergence callbacks are not
overwritten. First validate modes 0 and 1 with matched linear settings, then
compare more passes and smaller dt at equal physical times in mode 2.

Local verification: `python tests/picard_inner.py` (set OCCA_DIR if needed).
It uses actual OCCA Serial memory, v26 deviceMemory, native sumMakef kernels,
native solution-history lagging and runInnerStep with the production Picard
module/step guard. Controlled nonlinear physics and elliptic doubles exercise
BDF1/2/3, separate scalar/fluid EXT, variable mixture properties, pressure
filter anchoring, VM anchoring, current forces, one-time commits, frozen replay,
convergence/cap policies and intentional history corruption. Elliptic/spatial
physics are test doubles: full GPU/MPI bubbleColumn validation is still needed.
