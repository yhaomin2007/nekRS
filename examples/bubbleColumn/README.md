# Density-averaged method1: conservative UG transport of alpha

Alpha has no native mixture convection or matching RHS cancellation: its device-only advection compute flag is disabled before the first timestep. Host compute stays enabled, retaining the alpha Helmholtz solve, diffusion, HPFRT and boundary conditions. Native alpha advection histories are zero-initialized and remain zero. The alpha RHS directly supplies conservative cubature div(alpha*UG). UGX/UGY/UGZ retain native mixture convection plus identical RHS cancellation, replaced by cubature UG.grad(UG). UG remains transported velocity, not QG. Gas drag and effective virtual-mass inertia remain semi-implicit; pressure and liquid acceleration remain lagged.

At degree N, set cubaturePolynomialOrder >= 2*N because the conservative kernel differentiates the sampled product. The example N=7 uses order 14 (15 Gauss points). Scalar explicit histories use EXT1; native fluid/BDF integration is unchanged.

Fresh starts have only the axial plume: f(z)=0.5*(1-tanh((z-H)/delta)), alpha=alphaInlet*f, UGz=gasInitialVelocity*f. Initial liquid is stationary and mixture velocity follows the density-weighted relation. initialPlumeHeight and initialPlumeThickness remain configurable. Restart files take precedence. Uniform initialization selectors were removed.

Inlet alpha and UG are uniform (alphaInlet and gasInletVelocity). Shared inlet/wall rim nodes retain wall precedence for velocity. The tapered inlet option was removed.

UG reconstruction and pressure/drag/gravity/VM sources are not switched off at low alpha. Division by alpha/beta remains protected with alphaFloor; magnitude clipping is a separate optional control. The old low-alpha cutoff and smooth-mask par options no longer control this case.

## Mixture divergence

mixtureDivergenceEnabled=1 supplies the target calculated from the actual alpha equation RHS; 0 imposes zero divergence. divergenceRampEnabled and divergenceRampSteps retain the cosine startup ramp. Optional divergence filtering remains available.

Divergence uses the standalone user module explicitFilter.hpp / explicitFilter.okl, ported from the supplied explicitFilterApply. It uses the original Legendre bubble basis (1, x, P_k-P_(k-2)), whose higher modes vanish at endpoints. divergenceFilterModes selects 1..N-1 high modes; divergenceFilterStrength in [0,1] sets their parabolic attenuation. The module directly applies F tensor F tensor F in-place to raw divergence. It preserves element corner values and compatible face traces; this is not the former ordinary-Legendre filter. Strength 0 is identity. Filtered divergence is assembled before optional extrapolation and ramp and supplied to pressure; raw comparisons stay unfiltered.

To reuse the module in another v26 case, include explicitFilter.hpp, call userExplicitFilter::registerKernels(kernelInfo) during UDF_LoadKernels, then call userExplicitFilter::explicitFilterApply(tag, mesh, modes, strength, field). Register the kernel for the case polynomial degree; fields may be padded or scalar slices with at least mesh Nlocal entries. The module performs element-local filtering only; the caller handles gather-scatter as needed. Matrices are cached by tag and rebuilt when mesh, degree, modes or strength changes.

The RHS is recovered from native scalar o_JwF by removing its BDF history load. This preserves actual UG-flux/EXT/source and HPF contributions without any alpha mixture-convection cancellation. Native weak diffusion is evaluated on the pre-clipping solved alpha, then the BDF filtering/clipping-rate correction is added. At imposed alpha Dirichlet nodes, boundary evolution replaces the interior equation. For the continuity constraint only, evaluate mixture material advection of updated alpha using a separate enabled user mask (independent of the disabled native alpha transport mask), multiply by (rhoLiquid-rhoGas)/rhoM and assemble to obtain the target.

The BDF alternative remains an exported diagnostic only; there is no longer a selectable mixtureDivergenceMethod. Raw recovered-RHS, raw BDF, actual pressure target and their signed difference are exported in the div checkpoint.

The pressure target still uses predicted mixture advection before the pressure-corrected U exists. Thus recovering the alpha RHS consistently does not make the whole pressure/gas coupling implicit or establish exact final total-mass continuity. Pressure does not correct UG within the same step.

## Alpha explicit filter

alphaExplicitFilterModes and alphaExplicitFilterStrength configure the same standalone bubble-basis module on alpha after each scalar solve, before density/divergence/pressure evaluation. Strength 0 disables it; the example enables one mode at strength 0.05. Existing HPFRT regularization remains independent and can be combined with this filter. The sequence is solve alpha, explicit filter, shared-node assembly, reapply prescribed scalar boundaries, then optional clipping. UG is not explicitly filtered.

Divergence recovery uses weak diffusion on the original implicit alpha solution and adds g0/dt*(alpha_postprocessed-alpha_solved), including filtering and clipping. Conservation CSV records filter and clipping volume changes separately and accounts for both in corrected cumulative balances. The filter is not a positivity limiter or generally volume-preserving on curved elements. Filtering acts once per scalar advancement; in the current one-pass case this is once per timestep.

## Verification

Serial checks cover actual forcing recovery with HPF, weak diffusion, clipping and imposed boundary correction; native advection cancellation; gas/VM scaling and dilute forcing; plume initialization; uniform/rim boundary consistency; diagnostic kernels and conservative transport. Full GPU/MPI integration must be checked in the user's NekRS installation. Core files are not modified.

METHOD1_AUDIT.md records the earlier pre-simplification snapshot and is retained as historical analysis.

Obsolete alpha native/cancellation comparison columns were removed from conservation CSV. Mixture material advection remains in divergence reconstruction, not in the alpha transport RHS; UM also continues to affect solved UG through physical coupling.
