# Method1 alpha stabilization: first-stage implementation

This implements the supplied handoff in **bubbleColumn / Method1**. No core,
momentum, drag, pressure algorithm, phase correction, convection selection,
boundary map, baseline diffusion or HPFRT setting is changed. This is an
unvalidated numerical stabilization experiment, not a physical dispersion model.

## Implemented model

The added flux is q = [Dcw I + (Dstream-Dcw) n n^T] grad(alpha), with
n = UG/sqrt(|UG|^2 + epsilon^2). It is added as +div(q) on the alpha RHS.
Both coefficients are constant in space and time; the orientation follows the
actual source-time gas velocity, including in the plume interior. There is no
front, alpha-gradient, low-alpha or shear activation mask.

| CASEDATA parameter | Default | Meaning |
| --- | --- | --- |
| alphaStabilizationMode | 1 | 0 off; 1 crosswind; 3 streamline artificial diffusion; 4 both |
| alphaCrosswindDiffusivity | 2e-5 | Dcw in m^2/s, active in modes 1 and 4 |
| alphaStreamlineDiffusivity | 2e-5 | Dstream in m^2/s, active in modes 3 and 4 |
| alphaDirectionEpsilon | 1e-6 | Positive velocity regularization in m/s |

Mode 0 does not alter the scalar forcing or mixture divergence target and is
the baseline comparison. Modes 2 (adaptive) and 5 (SUPG) are rejected explicitly.
Streamline artificial diffusion is **not SUPG/GLS**. Residual, modal and shear
sensors are diagnostic/deferred work, not adaptive diffusivity selectors.

For vertical nonzero UG, crosswind mainly damps horizontal gradients and
streamline mainly damps vertical gradients. At UG=0, crosswind is isotropic,
whereas streamline vanishes. Near zero velocity the regularized transverse
tensor is not an exact projector. Equal active coefficients give exactly
isotropic added diffusion, independent of velocity, a useful comparison test.

The default native alpha diffusion remains D0=1e-5 m^2/s and is implicit.
The default added crosswind coefficient is the handoff's first nonzero sweep
value. Existing HPFRT is retained so that its change does not confound the test.
Sweep Dcw = 0, 2e-5, 5e-5, 1e-4 with mode 1; compare identical physical times.
Mode 0 is the strict baseline; mode 4 with Dcw=Dstream gives isotropic added
diffusion of that amount. A zero Dcw in mode 1 also adds no stabilization.

## Weak SEM operator and boundaries

Native gradientVolume is evaluated **without gather/scatter**. Its result is
element-local JW*grad(alpha); the flux kernel divides by that element's JW,
then forms q at GLL nodes. Native weak divergence supplies +G^T W q. We negate
it, assemble with gather/scatter and divide by assembled lumped mass:

Fstab = -M^-1 sum_e G_e^T W_e D_e G_e alpha.

This preserves constants and, for fixed UG/nonnegative coefficients, obeys
alpha^T M Fstab <= 0. Averaging the gradient before forming q would generally
break this symmetric weak stiffness construction; the averaged gradient is
used only for visualization diagnostics.

No artificial surface load is added. Thus alpha Neumann walls and outlet have
natural zero added diffusive normal flux in the weak equation. This does not
mean a separately reconstructed pointwise q.n must vanish at every wall node.
The existing inlet Dirichlet condition remains imposed by the native scalar
solver. Its constrained-node reactions can exchange gas volume; homogeneous
Neumann added diffusion plus prescribed Dirichlet alpha is not a promise of
zero net diffusion exchange at a Dirichlet inlet.

With all boundary rows free, the raw assembled added load integrates to zero
to roundoff. Actual gas-volume changes also include inlet/outlet advection,
baseline diffusion, Dirichlet reactions, HPFRT and the retained nodal clipping.
No new clipping or explicit alpha filter is introduced. A fully bound-preserving
and globally conservative transport method remains future work.

## Time integration and Method1 continuity

Fstab is sampled at the completed source-time state. Its own history uses the
native fluid EXT coefficients, including native startup and variable timesteps.
The scalar's existing EXT1 carrier passes this result without double EXT.
This is **explicit added diffusion**; D0 alone remains implicit.

The same selected Fstab is added once to the alpha source and to
o_alphaExplicitDiffusion. Existing Method1 continuity reconstruction therefore
includes it with precisely the same time weights as alpha. It is not added to
the already extrapolated conservative transport history or to UM momentum.

At setup, the module computes a geometry/polynomial-order-aware upper bound
L on the unit-diffusion stiffness spectral radius. It is the maximum element
trace of M_e^-1/2 K_e M_e^-1/2, globally reduced. Tensor diffusion is bounded
above by max(Dcw,Dstream)*I, so theta = dt*max(Dcw,Dstream)*L bounds the
added diffusion's dt-scaled eigenvalues. It is reported as
explicitDiffusionBoundDt. This is conservative and may considerably overestimate
the actual eigenvalue. For isolated BDF2/EXT2 negative-real diffusion, theta
below 4/3 is a sufficient bound; it is NOT a stability guarantee for the full
advection/diffusion/coupled EE system. No timestep is silently changed.

Reduce dt when the added diffusivity or spatial resolution is increased.
No spatial stabilization can guarantee stability under an excessive explicit
diffusion timestep, nor does the weak dissipative sign guarantee positivity.

## Output

At diagnosticInterval (first step, every N steps, and checkpoints), the log and
bubbleColumn_alpha_stabilization.csv record alpha extrema, gas volume, current
advective boundary flux, net raw and applied-EXT stabilization rates, raw energy
production, clipping volume change, a split gas-volume budget defect and theta.
EnergyRaw should be nonpositive to roundoff for the current frozen tensor.
NetStab(raw,EXT) should be near roundoff on the full unmasked weak load. This is
not the volume balance of the final Dirichlet-constrained/clipped solution.

Checkpoint astGrad stores the current assembled grad(alpha) as a vector.
The FP64 ast file stores these scalar fields (VisIt s1..s11):

| VisIt | Meaning |
| --- | --- |
| s1 | Final alpha |
| s2 | Current assembled div(UG) |
| s3 | Current UG.grad(alpha) |
| s4 | Current alpha*div(UG) |
| s5 | Active Dcw |
| s6 | Active Dstream |
| s7 | Current raw weak stabilization RHS |
| s8 | Source-time stabilization RHS actually applied, after EXT |
| s9 | Current regularized transverse |Pperp grad(UGz)|, in 1/s |
| s10 | Baseline split residual, before subtracting stabilization |
| s11 | Split residual after subtracting applied stabilization |

The split residual uses the actual pre-lag native BDF history and selected
conservative transport EXT. Baseline diffusion uses the same pre-clipping
implicit alpha solution as Method1's divergence target. Final alpha is used
for the BDF current value. Therefore s11 still contains HPFRT, boundary
substitution, clipping and linear-solver effects; it is not a nonlinear
final-time physical residual or the scalar Helmholtz solver residual.
splitBudgetDefect is its volume integral. Current boundary flux is an
instantaneous diagnostic; it is not substituted for EXT transport in that
time-discrete budget. On restart, missing user EXT histories use the existing
constant-continuation policy; transient restart discrepancies are expected.

## Verification and limits

Run tests/alpha_stabilization.py. It checks production kernels, real native
weak gradient/divergence execution, unequal element masses/shared-node
assembly, dissipation, zero net weak load, constant preservation, tensor
directions/regularization and equal-coefficient isotropy. It also checks
source/mixture-target routing, explicit history use and the geometric bound.

Serial kernels and CUDA translation do not establish long-time stripe
suppression. Compare alpha profiles, plume width, conservation, onset time and
cost against the baseline and isotropic-diffusion runs before calibration.
