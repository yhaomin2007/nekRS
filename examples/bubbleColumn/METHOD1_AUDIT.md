# Method1 equation and implementation audit

Audited physics snapshot: remote `bubbleColumn` commit `22d958cb36405749cbbb55474ff61e704b108658`, 2026-10-13. This review concerns `examples/bubbleColumn`, not method2. Repository par defaults are described below; the user's actual runtime par, log and nodal fields for the latest failed run were not supplied. No solver equations were changed during this audit.

## 1. Continuum equations and definitions

Assume incompressible phases, constant phase densities, no phase change, alpha = gas volume fraction, beta = 1-alpha. Gas/liquid velocities are ug/ul. Define

\[
\rho_m=\beta\rho_l+\alpha\rho_g,\qquad
\mathbf U=\frac{\beta\rho_l\mathbf u_l+\alpha\rho_g\mathbf u_g}{\rho_m},
\qquad
\mathbf u_l=\frac{\rho_m\mathbf U-\alpha\rho_g\mathbf u_g}{\beta\rho_l}.
\]

The physical phase continuity equations are

\[
\partial_t\alpha+\nabla\cdot(\alpha\mathbf u_g)=0,
\qquad
\partial_t\beta+\nabla\cdot(\beta\mathbf u_l)=0.
\]

Their mass-weighted sum gives

\[
\partial_t\rho_m+\nabla\cdot(\rho_m\mathbf U)=0,
\qquad
\nabla\cdot\mathbf U=
\frac{\rho_l-\rho_g}{\rho_m}
(\partial_t\alpha+\mathbf U\cdot\nabla\alpha).
\]

Thus density-averaged U is generally not divergence-free. This identity has the correct sign in both case divergence kernels. It is not by itself a complete discrete pressure algorithm.

For drag coefficient Ki and virtual mass c=Cvm*rho_l, the intended gas material-velocity equation is

\[
(\rho_g+c)D_g\mathbf u_g=
-\nabla p+\rho_g\mathbf g-K_i(\mathbf u_g-\mathbf u_l)
+cD_l\mathbf u_l+\alpha^{-1}\nabla\cdot(\alpha\boldsymbol\tau_g).
\]

Here Dg = partial_t + ug.dot(grad) and Dl = partial_t + ul.dot(grad). The gas stress uses the Newtonian Stokes form. In the physical velocity equation there is no additional ug*div(ug) term: it cancels with phase continuity when converting conservative phase momentum to material velocity. Adding it separately would change the equation. Artificial alpha diffusion/filters break exact phase mass/momentum equivalence unless their fluxes are treated consistently.

Summing physical phase momentum cancels internal drag and virtual mass forces. With consistent total mass continuity,

\[
\rho_mD_m\mathbf U=-\nabla p+\rho_m\mathbf g
+\nabla\cdot(\alpha\boldsymbol\tau_g+\beta\boldsymbol\tau_l)
-\nabla\cdot\boldsymbol\tau_d,
\]

\[
\boldsymbol\tau_d=
\frac{\alpha\rho_g\rho_m}{\beta\rho_l}
(\mathbf u_g-\mathbf U)\otimes(\mathbf u_g-\mathbf U).
\]

The implemented drift coefficient agrees with this identity away from clamps/phase-activity masks. Using mu_m=beta*mu_l+alpha*mu_g with a native U stress operator is a closure approximation; it is not an exact replacement for the two separate phase stresses when slip and alpha gradients exist.

## 2. Equations actually advanced with current defaults

Defaults: polynomial degree 7; dt=1e-5; BDF2; scalar EXT1; alphaTransportForm=2; mixtureDivergenceMethod=0; virtual mass enabled with Cvm=0.5; implicit numerical diffusivity 1e-5 for all four scalars; HPFRT enabled; alpha clipping to [0,1]. The divergence ramp lasts 1000 steps. Divergence filtering is off in the repository par.

Define BDF derivative B alpha_new = (g0*alpha_new - sum(b_j*alpha_history_j))/dt. On unconstrained scalar degrees of freedom the alpha equation is schematically

\[
M B\alpha^{n+1}+K_{D_\alpha}\alpha^{n+1}
=-M C_{g,h}(\alpha^n,\mathbf u_g^n)+M H_\alpha^n,
\]

where Cg,h is the case conservative cubature divergence and H is native HPF relaxation. Dirichlet lifting/constraints are part of the native solve. Post-solve clipping creates a further correction to this equation.

The code places +native gas advection in the alpha explicit forcing to cancel the -native advection already included by scalar sumMakef. After mass assembly, matching EXT history leaves the conservative source once. This cancellation was independently tested with unequal shared-node masses and EXT1/EXT2. There is no identified double-counting defect in this path.

The gas component equation has implicit time, artificial diffusion and frozen drag reaction. Its advective term is native cubature advective-form transport with the frozen latest UG. Explicit sources use lagged pressure, reconstructed liquid velocity, lagged liquid material acceleration and physical gas stress. In schematic form,

\[
B\mathbf u_g^{n+1}+\lambda_d\mathbf u_g^{n+1}
-L_{\nu_{g,num}}\mathbf u_g^{n+1}
=-A_{g,h}(\mathbf u_g^n)
+\chi\frac{-\nabla p^n+\rho_g\mathbf g+K_i\mathbf u_l^n+c\mathbf a_l^{lag}}{\rho_g+c}
+\mathbf S_{\tau_g}^n+\mathbf H_g^n.
\]

The frozen reaction is lambda_d=chi*Ki/(rho_g+c). The code uses chi=1 when clamped alpha >= alphaFloor, zero otherwise. The stress addition requires alpha > alphaFloor. Numerical diffusion is additional to the explicit physical gas stress, because subtractUgDiffusion defaults to zero.

VM is semi-implicit for gas acceleration. Its liquid acceleration is evaluated at completed steps with a first-order difference plus strong ul.dot(grad ul), then lagged. BDF2 gas time integration does not make the entire coupled VM treatment second order. With scalar EXT1, the whole scalar advancement is generally not second order despite using BDF2 for time derivatives.

Native mixture momentum uses updated density/viscosity, native explicit mixture advection and explicit gravity/drift acceleration. Its pressure correction enforces the supplied target divergence. It is not a coupled pressure correction of both gas and liquid momenta.

## 3. Exact execution order

`src/app/nrs/nrs.cpp` calls the following paths:

1. Extrapolate native fluid/scalar states; compute mixture contravariant velocity.
2. `userSource`: prepare latest-state gas advector; build old-state alpha/gas/fluid explicit forces; native alpha-advection cancellation.
3. Native advection and HPF; construct scalar/fluid BDF and EXT forcing.
4. Lag solution buffers; apply time-new Dirichlet boundaries.
5. Solve alpha and all three UG components.
6. `postScalar`: clip alpha.
7. `userProperties`: optionally damp/clip UG, recompute sources with new scalars and pre-fluid mixture/pressure state; update density and diffusion; construct both divergence candidates.
8. Select raw divergence, optionally filter/reassemble, apply optional divergence history/ramp.
9. Solve fluid pressure and mixture velocity. UG receives no correction from this new pressure in the current step.
10. Completed-step UDF updates liquid-acceleration history and diagnostics.

This is a partitioned, lagged coupling. No outer convergence of alpha/UG/U/p is performed in this case.

## 4. Confirmed continuity mismatches

### Default method0 is not the discrete alpha update

For conservative transport, current method0 constructs

\[
d_0=k(\alpha^{n+1})[
-C_{g,h}(\alpha^{n+1},\mathbf u_g^n)
+A_{m,h}(\alpha^{n+1};\mathbf U^{EXT})
+L_{strong}(D_\alpha\nabla\alpha^{n+1})],
\quad k=(\rho_l-\rho_g)/\rho_m.
\]

But alpha actually used Cg,h(alpha_old,UG_old), implicit weak stiffness diffusion, and explicit HPF. Method0:

- substitutes new-alpha transport for the explicitly advanced old-alpha transport;
- omits HPF and post-solve clipping contributions;
- replaces the implicit weak diffusion load by divergence of an assembled strong gradient;
- does not reconstruct optional explicit diffusion subtraction in post-solve properties.

The strong diffusion reconstruction and weak stiffness are not identical globally, particularly at interelement interfaces and constrained boundaries. These are implementation inconsistencies, not proof of which one dominates the failed run.

### Method1 improves time consistency but is not fully pressure-consistent

Method1 uses the actual BDF alpha change, including whatever diffusion/filter/clipping altered the stored field:

\[
d_1=k(\alpha^{n+1})[B\alpha^{n+1}+A_{m,h}(\alpha^{n+1};\mathbf U^{EXT})].
\]

It still uses predicted mixture advection before U_new is pressure-corrected. Even in a continuum analogy, if the target uses U_predict but the final continuity uses U_new, the total mass residual is

\[
R_m=(\rho_g-\rho_l)(\mathbf U^{new}-\mathbf U^{predict})\cdot\nabla\alpha,
\]

assuming the other operators match. A sharp alpha gradient amplifies this mismatch. Therefore switching to method1 alone is a diagnostic improvement, not a proof of exact mixture mass conservation.

Filtering and startup ramping intentionally modify the target identity. They can suppress perturbations, but require monitoring conservation and target-vs-reconstructed residuals.

## 5. Confirmed high-degree cubature aliasing

Default parsing gives cubature polynomial degree 10, i.e. cubNq=11 for N=7. The current form2 kernel separately interpolates alpha/UG, forms their flux samples and differentiates those products with the 11-point Gauss differentiation matrix.

Two degree-7 fields can have a degree-14 product. An 11-point differentiation matrix differentiates the degree-10 interpolant of those samples, not the degree-14 product. This is different from differentiating the factors separately or using a consistent weak flux form with boundary contributions.

Reproduction uses the actual interpolation, conservative-product and projection kernels on an affine element:

\[
\alpha(r)=0.02[1+0.3P_7(r)],\qquad
u_{g,x}(r)=0.3[1+0.3P_7(r)],\qquad u_{g,y}=u_{g,z}=0.
\]

Both factors are positive and representable. Results from `tests/audit_cubature_degree.py`:

| cubNq | Relative error in projected transport load | Integrated boundary balance |
|---|---:|---|
| 11 (default) | 0.120680950 | passes |
| 15 | 3.46e-15 | passes |

Previous low-degree regression fields did not excite this error. Boundary conservation alone does not prove pointwise/load accuracy or dealiasing. I should not have described the current operator as fully dealiased for arbitrary degree-7 inputs.

For this particular product-differentiation implementation, cubNq >= 2*N+1 resolves the product polynomial on an affine element. For N=7 the diagnostic setting is `cubaturePolynomialOrder=14` (15 points). Curved metric/quadrature accuracy remains a separate issue. This does not establish boundedness or coupled stability, and it increases work for native operators as well.

## 6. Phase masks, boundaries and initialization

Turning off `gasVelocityLowAlphaCutoffEnabled` preserves velocity reconstruction and bypasses smooth low-alpha velocity damping. It does NOT disable chi in the gas pressure/drag/gravity/VM source, the alphaFloor stress division guard, or the drift-stress masks. Crossing alphaFloor therefore still changes gas acceleration abruptly. The current reaction/source switches are not a smooth phase-activity model.

`alphaTransportForm=1/2` includes full compression and ignores both `alphaCompressionMode` and `alphaCompressionInletThickness`. Thus repository form2 with thickness=0.01 does NOT suppress inlet compression. It cannot be compared to a prior split run with that term locally disabled without acknowledging the changed equation.

Inlet alpha is fixed; UG and mixture velocity share the same wall-distance smoothstep weight. Their relation correctly gives zero liquid inlet velocity. Wall UG/U are zero; alpha has zero Neumann walls. Outlet UG/alpha/U use zero Neumann. Rim wall precedence is consistent.

Mode2 initialization keeps UG uniform throughout the domain while only alpha has the axial plume. It does not match the new radial inlet taper or no-slip wall values. There is no explicit compatible hydrostatic initial-pressure construction in this UDF. These startup adjustments may seed disturbances, but their dominance has not been established.

The outlet pressure callback retains `-0.5*|U|^2*s0`. Its density normalization should be checked against the intended physical-pressure convention before using it as a quantitative backflow-pressure model. It is not identified as the early inlet instability cause.

## 7. What the tests and observations establish

The repaired norm monitor avoids the core offset-shadowing defect. The case CFL uses the native contravariant/GLL-spacing definition. Native automatic targetCFL remains based on mixture U, not UG. CFL constrains transport stability, not every lagged interphase feedback.

Uniform runs succeeding, and plume transport succeeding with mixture divergence disabled, support investigating the pressure/divergence feedback first. They do not establish the exact trigger. The observed horizontal alpha/UG bands and scalar extrema do not provide a temporal causal ordering; nodal data and early-time pressure/divergence fields are still needed.

Existing serial regression checks exercise individual kernels, cancellation, boundary values and routing. They do not test the full coupled GPU/MPI solve. The new high-degree audit demonstrates a gap in the previous coverage.

## 8. Prioritized corrections and controlled comparisons

1. Make source-based divergence consistent with the alpha update actually advanced: frozen/extrapolated transport history, native weak diffusion, HPF, and explicit accounting for clipping/Dirichlet effects. Use BDF reconstruction as an independent diagnostic, not label either current path exact.
2. Correct form2's product-differentiation aliasing, or first test cubaturePolynomialOrder=14 at N=7 to isolate this known mechanism. Retain old modes for comparison.
3. Measure final total-mass residual using corrected U, rather than only checking fluid divergence against its supplied target. Quantify the predicted-to-corrected velocity contribution near the front.
4. Separate velocity cutoff from force activation explicitly in diagnostics/options; evaluate a consistent smooth phase-activity treatment rather than assume the existing cutoff switch removed all phase transitions.
5. Match initialized velocity to inlet/wall profiles and review pressure initialization, then run a controlled matrix varying one feature at a time.

Do not assume an implicit convection solve alone fixes these mismatches. Do not conclude that every oscillation is a CG spatial-resolution problem. The confirmed defects warrant correction before relying on further filter tuning as validation.

## Code map

- `bubbleColumn.udf`: parameters, initial conditions, callbacks, output/history updates.
- `bubbleColumnEquations.okl`: velocity reconstruction, material acceleration history, phase properties, drift/gas stress, drag/VM sources, divergence formulas.
- `bubbleColumnTransport.okl`: interpolation, products/differentiation, projection and conservative alpha operator.
- `bubbleColumnTerms.hpp`: preparation/forcing, native advection cancellation, diffusion/stress, divergence selection/filtering, monitoring.
- `bubbleColumnBoundary.oudf`: inlet, wall and outlet data.
- `src/app/nrs/nrs.cpp`: native time ordering and pressure/divergence callbacks (read only).
- `src/solver/scalar/scalarSolver.cpp` and scalar `sumMakef.okl`: BDF/EXT and weak implicit scalar solve (read only).
- `src/solver/fluid/fluidSolver.cpp`: variable-density native projection, gradient-of-divergence terms and velocity solve (read only).
- `src/platform/par/parseGeneral.hpp`: default cubature degree (read only).
