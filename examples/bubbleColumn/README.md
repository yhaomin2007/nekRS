# Density-averaged method1: conservative UG transport of alpha

Alpha has no native mixture convection or matching RHS cancellation: its device-only advection compute flag is disabled before the first timestep. Host compute stays enabled, retaining the alpha Helmholtz solve, diffusion, HPFRT and boundary conditions. Native alpha advection histories are zero-initialized and remain zero. The alpha RHS directly supplies conservative div(alpha*UG): alphaTransportForm=1 forms the product at GLL nodes then applies assembled opSEM::strongDivergence; alphaTransportForm=2 retains conservative cubature differentiation (default). Both use the same frozen UG and time treatment. The former aliases the nodal product; neither form guarantees boundedness. UGX/UGY/UGZ retain native mixture convection plus identical RHS cancellation, replaced by cubature UG.grad(UG). UG remains transported velocity, not QG. Gas drag and effective virtual-mass inertia remain semi-implicit; pressure and liquid acceleration remain lagged.

For alphaTransportForm=2, set cubaturePolynomialOrder >= 2*N because the conservative cubature kernel differentiates the sampled product. With alphaTransportForm=1 this extra requirement is lifted; native/UG cubature still requires sufficient quadrature for its own operator. The example N=7 uses order 14 (15 Gauss points). Scalar explicit histories use EXT1; native fluid/BDF integration is unchanged.

Fresh starts have only the axial plume: f(z)=0.5*(1-tanh((z-H)/delta)), alpha=alphaInlet*f, UGz=gasInitialVelocity*f. Initial liquid is stationary and mixture velocity follows the density-weighted relation. initialPlumeHeight and initialPlumeThickness remain configurable. Restart files take precedence. Uniform initialization selectors were removed.

Inlet alpha and UG are uniform (alphaInlet and gasInletVelocity). Shared inlet/wall rim nodes retain wall precedence for velocity. The tapered inlet option was removed.

UG reconstruction and pressure/drag/gravity/VM sources are not switched off at low alpha. Division by alpha/beta remains protected with alphaFloor; magnitude clipping is a separate optional control. The old low-alpha cutoff and smooth-mask par options no longer control this case.

## Mixture divergence

mixtureDivergenceEnabled=1 supplies the target calculated from the actual alpha equation RHS; 0 imposes zero divergence. divergenceRampEnabled and divergenceRampSteps retain the cosine startup ramp. Optional divergence filtering remains available.

Divergence uses the standalone user module explicitFilter.hpp / explicitFilter.okl, ported from the supplied explicitFilterApply. It uses the original Legendre bubble basis (1, x, P_k-P_(k-2)), whose higher modes vanish at endpoints. divergenceFilterModes selects 1..N-1 high modes; divergenceFilterStrength in [0,1] sets their parabolic attenuation. The module directly applies F tensor F tensor F in-place to raw divergence. It preserves element corner values and compatible face traces; this is not the former ordinary-Legendre filter. Strength 0 is identity. Filtered divergence is assembled before optional extrapolation and ramp and supplied to pressure; raw comparisons stay unfiltered.

To reuse the module in another v26 case, include explicitFilter.hpp, call userExplicitFilter::registerKernels(kernelInfo) during UDF_LoadKernels, then call userExplicitFilter::explicitFilterApply(tag, mesh, modes, strength, field). Register the kernel for the case polynomial degree; fields may be padded or scalar slices with at least mesh Nlocal entries. The module performs element-local filtering only; the caller handles gather-scatter as needed. Matrices are cached by tag and rebuilt when mesh, degree, modes or strength changes.

The RHS is recovered from native scalar o_JwF by removing its BDF history load. This preserves actual UG-flux/EXT/source and HPF contributions without any alpha mixture-convection cancellation. Native weak diffusion is evaluated on the pre-clipping solved alpha, then the BDF filtering/clipping-rate correction is added. At imposed alpha Dirichlet nodes, boundary evolution replaces the interior equation. For the continuity constraint only, evaluate mixture material advection of updated alpha using a separate enabled user mask (independent of the disabled native alpha transport mask), multiply by (rhoLiquid-rhoGas)/rhoM and assemble to obtain the target.

The BDF alternative remains an exported diagnostic only; there is no longer a selectable mixtureDivergenceMethod. Raw recovered-RHS, raw BDF, actual pressure target and their signed difference are exported in the div checkpoint.

The raw target is d_raw=(rhoLiquid-rhoGas)/rhoM * (D_t^BDF alpha_postprocessed + A_UMpred(alpha_postprocessed)). The recovered-RHS and independent BDF kernels are checked against each other with nonzero UM material advection and native alpha convection disabled, including weak diffusion, HPFRT, filtering/clipping and imposed boundary corrections. The final target may differ from this raw continuity identity because of divergence filtering, ramp or optional history extrapolation.

The pressure target still uses predicted mixture advection before the pressure-corrected U exists. Thus recovering the alpha RHS consistently does not make the whole pressure/gas coupling implicit or establish exact final total-mass continuity. Pressure does not correct UG within the same step.

## UG pressure-gradient explicit filter

Set gasPressureGradientExplicitFilterModes and gasPressureGradientExplicitFilterStrength to apply the standalone bubble-basis filter separately to all three selected pressure-gradient components. The example uses one mode at strength 0.05; strength 0 disables it. Sequence: compute raw strong/weak gradients, copy the selected field, spatial filter, shared-node assembly, optional existing temporal relaxation, then optional inlet mask. Raw diagnostic gradients stay unchanged. Each evaluation starts from raw gradients, preventing repeated filter application to cached history. This filters the gradient entering UG forcing only; it does not filter pressure or the native mixture momentum pressure term.

## Alpha explicit filter

alphaExplicitFilterModes and alphaExplicitFilterStrength configure the same standalone bubble-basis module on alpha after each scalar solve, before density/divergence/pressure evaluation. Strength 0 disables it; the example enables one mode at strength 0.05. Existing HPFRT regularization remains independent and can be combined with this filter. The sequence is solve alpha, explicit filter, shared-node assembly, reapply prescribed scalar boundaries, then optional clipping. UG is not explicitly filtered.

Divergence recovery uses weak diffusion on the original implicit alpha solution and adds g0/dt*(alpha_postprocessed-alpha_solved), including filtering and clipping. Conservation CSV records filter and clipping volume changes separately and accounts for both in corrected cumulative balances. The filter is not a positivity limiter or generally volume-preserving on curved elements. Filtering acts once per scalar advancement; in the current one-pass case this is once per timestep.

## Verification

Serial checks cover actual forcing recovery with HPF, weak diffusion, clipping and imposed boundary correction; native advection cancellation; gas/VM scaling and dilute forcing; plume initialization; uniform/rim boundary consistency; diagnostic kernels and conservative transport. Full GPU/MPI integration must be checked in the user's NekRS installation. Core files are not modified.

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
