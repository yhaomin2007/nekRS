# Two-fluid inlet--outlet pipe with native alpha scalar

This case reuses `examples/turbPipe/turbPipe.re2` (boundary 1 inlet,
boundary 2 outlet, boundary 3 wall). It has no periodic boundary and no
constant-flow-rate control.

The liquid and gas inlets are uniform plug profiles with `uz=0.10` and
`uz=0.15`, respectively. The liquid wall is no-slip, while the gas wall uses
slip/no-penetration (`zeroDirichletN/zeroNeumann`). Both interior velocity
fields start from rest, while alpha starts uniformly at `0.05`, so inlet
startup and the associated pressure evolution can be observed directly.

Each physical timestep uses one segregated pass: native alpha transport with
the gas velocity, gas momentum, liquid momentum, and one shared-pressure
correction. There are no nonlinear two-fluid inner iterations.

Gas volume fraction is the ordinary NekRS scalar named `alpha`. Setting
`nativeAlphaScalar = true` binds that scalar's transport velocity to the gas
velocity, including `o_U`, `o_Ue`, and `o_relUrst`. NekRS therefore supplies
the standard passive-scalar advection, implicit diffusion, boundary treatment,
and elliptic solve. The two-fluid coordinator aliases the scalar solution as
`alpha_g`, applies its conservative bound correction, and sets
`alpha_l = 1-alpha_g`.

The current native-scalar verification mode deliberately requires exactly one
scalar named `alpha`. It is intended to isolate alpha transport before the
general multi-species interface is designed.
