# Density-averaged method1: conservative UG transport of alpha

Alpha and UG have no native mixture convection or matching RHS cancellation: their device-only advection compute flags are disabled before the first timestep. All four Helmholtz solves, diffusion, HPFRT and boundary conditions remain enabled. The native scalar loop solves alpha then UG before the native pressure/UM solve; host solve masks are never changed. Native alpha/UG advection histories are zero-initialized and remain zero. The alpha RHS directly supplies conservative div(alpha*UG): alphaTransportForm=1 forms the product at GLL nodes then applies assembled opSEM::strongDivergence; alphaTransportForm=2 retains conservative cubature differentiation (default). Both use the same frozen UG and time treatment. The former aliases the nodal product; neither form guarantees boundedness. UGX/UGY/UGZ receive only the user cubature -UG.grad(UG) convection in their explicit RHS. UG remains transported velocity, not QG. Gas drag and effective virtual-mass inertia remain semi-implicit; pressure uses the latest completed pressure or independently selected gradient EXT; liquid acceleration and other nonconvection sources retain their previous latest-sample treatment.

For alphaTransportForm=2, set cubaturePolynomialOrder >= 2*N because the conservative cubature kernel differentiates the sampled product. With alphaTransportForm=1 this extra requirement is lifted; native/UG cubature still requires sufficient quadrature for its own operator. The example N=7 uses order 14 (15 Gauss points). The complete alpha and UG convection products use native fluid EXT coefficients. Scalar EXT1 carries already-selected sources, avoiding double extrapolation; native fluid/BDF integration is unchanged.

Fresh starts have only the axial plume: f(z)=0.5*(1-tanh((z-H)/delta)), alpha=alphaInlet*f, UGz=gasInitialVelocity*f. Initial liquid is stationary and mixture velocity follows the density-weighted relation. initialPlumeHeight and initialPlumeThickness remain configurable. Restart files take precedence. Uniform initialization selectors were removed.

Inlet alpha and UG are uniform (alphaInlet and gasInletVelocity). Shared inlet/wall rim nodes retain wall precedence for velocity. The tapered inlet option was removed.

UG reconstruction and pressure/drag/gravity/VM sources are not switched off at low alpha. Division by alpha/beta remains protected with alphaFloor; magnitude clipping is a separate optional control. The old low-alpha cutoff and smooth-mask par options no longer control this case.

## Mixture divergence

mixtureDivergenceEnabled=1 supplies a reduced target using alpha conservative transport and numerical diffusion only; 0 imposes zero divergence. divergenceRampEnabled and divergenceRampSteps retain the cosine startup ramp. Optional divergence filtering remains available.

`divergenceInletThickness = delta` applies a spatial inlet weight to the final pressure target, assuming the inlet is at `z=0`: zero for `z <= delta/2`, `0.5*(1-cos(pi*(2*z/delta-1)))` for `delta/2 < z < delta`, and one for `z >= delta`. Both value and first derivative match at the transition endpoints. Set the thickness to `0` to disable the layer (the default for older par files). The example uses `0.01` mesh units: a zero region up to `z=0.005`, followed by smooth recovery to `z=0.01`. The weight is applied after filtering, extrapolation and temporal ramp on each physical timestep. Raw divergence comparisons remain unchanged; exported applied target includes this weight. This is a diagnostic modification of mixture continuity, not a correction of the underlying boundary reconstruction.

Divergence uses the standalone user module explicitFilter.hpp / explicitFilter.okl, ported from the supplied explicitFilterApply. It uses the original Legendre bubble basis (1, x, P_k-P_(k-2)), whose higher modes vanish at endpoints. divergenceFilterModes selects 1..N-1 high modes; divergenceFilterStrength in [0,1] sets their parabolic attenuation. The module directly applies F tensor F tensor F in-place to raw divergence. It preserves element corner values and compatible face traces; this is not the former ordinary-Legendre filter. Strength 0 is identity. Filtered divergence is assembled before optional extrapolation and ramp and supplied to pressure; raw comparisons stay unfiltered.

To reuse the module in another v26 case, include explicitFilter.hpp, call userExplicitFilter::registerKernels(kernelInfo) during UDF_LoadKernels, then call userExplicitFilter::explicitFilterApply(tag, mesh, modes, strength, field). Register the kernel for the case polynomial degree; fields may be padded or scalar slices with at least mesh Nlocal entries. The module performs element-local filtering only; the caller handles gather-scatter as needed. Matrices are cached by tag and rebuilt when mesh, degree, modes or strength changes.

The reduced alpha RHS is `R_transport_diff = EXT_fluid[-div_h(alpha*UG)] + explicit_diffusion_compensation - M^-1 K_D alpha_solved`. Raw transport is cached before diffusion compensation and receives native fluid EXT once. Current explicit diffusion compensation is added after transport extrapolation; it is not extrapolated or recovered from native o_JwF. Property evaluations do not overwrite the cached selected transport. With subtractAlphaDiffusion=0, only conservative gas transport and implicit numerical diffusion remain.

Native weak diffusion uses the original implicit alpha solution, before explicit filtering/clipping. No HPFRT source, filter/clipping-rate correction, or Dirichlet-node BDF override is included. This does not disable these operations in the alpha solve itself: they still affect the evolved fields, but their equivalent rates are omitted from this divergence RHS. The weak diffusion operator is evaluated at boundary nodes as well; the former boundary temporal override is removed.

The raw target is `d_raw=(rhoLiquid-rhoGas)/rhoM * (R_transport_diff + A_UMused(alpha_final))`. The separate enabled user advection mask evaluates the mixture material term on final alpha even though native alpha/UG transport is disabled. UMused is the step-start advector used for fluid relUrst, not the newly pressure-corrected velocity. The density prefactor uses final alpha with its usual density clamp.

The full BDF alternative remains an exported diagnostic only; no par selector chooses it. Div s1 is the reduced transport/diffusion target, s2 is the full BDF target, s3 is the actual filtered/extrapolated/ramped pressure target, and s4 is s1-s2. In particular, s1 and s2 need not agree: s2 includes the time changes caused by HPFRT, alpha postprocessing and imposed boundaries. Their difference is no longer solely a numerical RHS-recovery error. The s5 velocity-lag diagnostic is unchanged.

This reduced target deliberately does not enforce the full discrete density continuity identity for all alpha regularization/postprocessing changes. Divergence filtering, startup ramp and optional history extrapolation can further change the target. Pressure still does not correct UG within the same timestep.

## UG pressure gradient

UG always uses the raw assembled strong gradient. The old weak-gradient
selection, pressure spatial filter, temporal relaxation and inlet mask settings
have been removed. The raw weak gradient is still exported for diagnostics only.
`gasPressureGradientExtrapolation=0` uses grad(p^n), the latest completed pressure
available before this step's pressure solve. `=1` uses the native fluid EXT
weights on raw strong-gradient histories. This is independent of convection EXT.

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
`d=(rhoL-rhoG)/rhoM * (EXT_fluid[-div(alpha*UG)] + diffusion_compensation + diffusion_implicit + A_UMused(alpha_final))`.

`mixtureDivergenceForm=2` (example par setting) uses the relative conservative flux:
`d=(rhoL-rhoG)/rhoL * (EXT_fluid[div(alpha*(UMused-UG))] + diffusion_compensation + diffusion_implicit)`.
The C++ default is 1 for existing par files. Only values 1 and 2 are accepted.
The relative flux is constructed once at the alpha source time using the same
alpha, step-start UM, and frozen UG time level. It follows alphaTransportForm:
1 uses assembled nodal-product SEM divergence; 2 uses conservative cubature.
The saved raw flux receives the same native fluid EXT coefficients as alpha transport.
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

## Fixed solve order and time extrapolation

The order is alpha -> UG -> pressure -> UM, once per physical step, with no
Picard iteration or deferred gas solve. Native scalar solve order is checked
against ALPHA, UGX, UGY, UGZ. No core or method2 code is changed.

At source assembly, nonlinear products are computed from a single physical-time
state and stored as separate histories. The complete conservative alpha flux
divergence and UG material convection are extrapolated using exactly the current
native fluid `o_coeffEXT`, including startup order and variable-step weights.
Velocity factors are not independently predicted first. The relative-flux
divergence target uses the same weights; current explicit diffusion compensation
and the solved implicit diffusion are added separately. HPF, filtering and
clipping remain excluded from this reduced divergence target.

The native scalar EXT array remains an independent EXT1 carrier. Thus each
pre-extrapolated convection/pressure source enters the native BDF forcing once;
other explicit sources and native scalar HPF retain latest-sample treatment.
Implicit diffusion, implicit gas drag and native UM convection are unchanged.
Pressure history is independent of convection history, so the pressure option
does not switch convection EXT off or apply EXT a second time.

Histories advance only during source assembly. Setup/property/diagnostic
evaluations cannot advance them or replace the selected gradient. No user
history files are written: after restart, missing slots are filled by constant
continuation of the first available sample, then replaced by real samples over
the next few steps. A restarted trajectory need not match uninterrupted EXT
during this short warmup.

`oneThroughLogInterval=1` logs every step; default 10 logs step 1 and each tenth
step; 0 disables. Rank 0 logs ALPHA_UG_PENDING, ALPHA_UG_DONE,
PRESSURE_UM_BEGIN and PRESSURE_UM_DONE, all with the enabled mask (1,1,1,1).
Native scalar solver residuals appear between the first two markers in the
checked alpha/UG order; pressure and UM residuals appear between the latter two.

Verification: `python3 tests/time_extrapolation.py` compiles actual C++ history,
pressure selection and convection routing against controlled CPU arrays. The
remaining spatial/operator regressions are separate; full GPU/MPI integration
is still required.

## UG equation diagnostics

Set these CASEDATA options (enabled by default for this diagnostic run):

```ini
ugTermDiagnosticsEnabled = 1
ugTermDiagnosticsInterval = 10
```

The interval selects rank-0 log/CSV statistics: step 1, each Nth step, and every
checkpoint. Use 1 for every step; 0 means step 1/checkpoints only. Setting enabled
to 0 disables runtime snapshots, reductions and exports. Enabled diagnostics
retain source/solved snapshots every step so checkpoint timing cannot miss them;
they use additional device memory and weak-operator evaluations (about 76
padded scalar fields, or 608 bytes per padded node in FP64).

All equation terms below are **accelerations** (m/s^2 for dimensional inputs),
with their signed contribution to the RHS, except the positive LHS BDF fields.
With c_VM=C_VM*rhoLiquid*virtualMassEnabled and rhoEff=rhoGas+c_VM, the
implemented normalized equation is:

```text
BDF(UG) = pressure + convection_EXT + gravity + drag_liquid + VM_liquid
        + gas_stress + diffusion_compensation + native_regularization
        + drag_implicit + diffusion_implicit_weak
BDF(UG) = BDF_diagonal - BDF_history
```

The gas-side VM inertia is already absorbed into rhoEff; VM_liquid is the
explicit liquid-acceleration contribution, not a second complete virtual-mass
force. Do not add the extra comparison fields (raw convection, net drag,
separate BDF pieces, defects or postprocess rate) to rhs_sum.

Each checkpoint writes the following independent vector databases. Open the
**velocity** field in each database; it contains the three components of that
term, not a velocity solution.

| Database | Log/CSV term | Definition |
| --- | --- | --- |
| gpr.nek5000 | pressure | -gasPressureEnabled * gradP_used/rhoEff; selected raw strong or EXT |
| gcv.nek5000 | convection_EXT | -EXT_fluid[UG dot grad(UG)], actually used |
| ggr.nek5000 | gravity | rhoGas/rhoEff * gravity |
| gdl.nek5000 | drag_liquid | lambda_source * UL_source |
| gvm.nek5000 | VM_liquid | c_VM/rhoEff * liquidAcceleration_source |
| gvs.nek5000 | gas_stress | div(alpha*tauGas)/(max(alpha,alphaFloor)*rhoEff) |
| gdc.nek5000 | diffusion_compensation | Configured positive weak-stiffness compensation at source time |
| gus.nek5000 | user_rhs | Actual UG user source copied before native regularization |
| ghp.nek5000 | native_regularization | Actual native explicit source minus cached user source; HPFRT in the current par |
| gdi.nek5000 | drag_implicit | -lambda_source * UG_solved |
| gnd.nek5000 | diffusion_implicit_weak | -M_assembled^-1 K_D UG_solved |
| gdt.nek5000 | BDF_time | Actual BDF derivative before gas postprocessing |
| grs.nek5000 | rhs_sum | user_rhs + native_regularization + drag_implicit + diffusion_implicit_weak |
| grb.nek5000 | balance_free | BDF_time - rhs_sum on free equation rows; zero on Dirichlet rows |
| gse.nek5000 | source_defect | user_rhs minus the sum of the seven explicit physical/transport terms |
| gpc.nek5000 | postprocess_rate | (g0/dt)*(UG_after_postprocessing - UG_solved); diagnostic only |
| gcr.nek5000 | convection_raw | -UG dot grad(UG) at the source time before EXT; comparison only |
| gdn.nek5000 | drag_net | drag_liquid + drag_implicit; comparison only |
| gbd.nek5000 | BDF_diagonal | (g0/dt)*UG_solved; comparison only |
| gbh.nek5000 | BDF_history | Sum of old-state BDF loads, cached before native history lagging |
| gmk.nek5000 | mask only | Per-component vector: 1=free equation row, 0=Dirichlet row; dimensionless |
| gcf.nek5000 | source coefficients | scalar00=source alpha; scalar01=source lambda (1/s); scalar02=rhoEff (kg/m^3) |

Source fields are captured **immediately after addExplicitSources**, including
the selected pressure gradient and already extrapolated convection. Later
properties, pressure/UM or VM-history evaluations cannot overwrite them.
Implicit diffusion, drag and BDF terms use the actual solved UG **before**
alpha/UG postprocessing. The native regularization is obtained from the actual
native explicit buffer after makeExplicit, not from a second filter application.
Sources are mass-weighted and assembled at shared nodes to match the weak
equation, including element-local native regularization.

The implicit diffusion field is a mass-normalized **weak residual**, not a
pointwise Laplacian at constrained boundaries. Inspect gmk when interpreting
boundary peaks. balance_free applies to the current zero-Neumann UG boundaries;
nonzero Neumann or Robin loads would require additional boundary-load accounting.
It is a spatial weak-equation diagnostic, not the elliptic solver's internal
preconditioned convergence norm.

UGTERMS log lines contain timestep, stage, term and MPI-global min/max for x/y/z.
bubbleColumn_ug_terms.csv stores the same ranges plus source_time. Source-stage
records are emitted before the gas solve, solved-stage records before pressure/UM,
and checkpoint-stage records cover all exported vectors. If any component has a
nonfinite value on any rank, nonfinite=1 is reported and that vector's range is
printed as NaN; the original field is preserved for inspection.

Start by inspecting source_defect (should be roundoff-level), then pressure,
gas_stress, VM_liquid, convection_EXT versus convection_raw, and drag_net.
Compare BDF_time with rhs_sum only where gmk=1. A nonzero postprocess_rate
identifies changes introduced after solving rather than by the Helmholtz equation.

Local verification: python3 tests/ug_term_diagnostics.py compiles the actual
diagnostic kernels together with the actual physical source kernels, and the
whole diagnostic host header over controlled CPU/IO doubles. It checks source
recomposition, VM/drag/pressure switches, low alpha, weak balance, component
masks, nonfinite values, postprocessing and BDF-history capture, CSV rows, and
all checkpoint names (including source alpha/drag/inertia coefficients). These checks do not replace full NekRS MPI/GPU validation.


Diagnostic setup allocates buffers only: in v26, UDF_Setup precedes scalar
elliptic-solver construction. Dirichlet masks are therefore read lazily at the
first postScalar capture after UG solves, validated before dereference, and
cached for the fixed mesh. The diagnostic regression starts with null solver
pointers and separately checks a missing solver table before constructing
solvers, so it exercises the native initialization order.
