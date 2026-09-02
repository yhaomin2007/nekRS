# Two-fluid inlet--outlet pipe with native alpha scalar

This case reuses `examples/turbPipe/turbPipe.re2` (boundary 1 inlet,
boundary 2 outlet, boundary 3 wall). It has no periodic boundary and no
constant-flow-rate control.

The liquid and gas have matching density (`1000`) and viscosity (`1.0e-3`).
Both inlets use the same uniform plug profile, `uz=0.10`, and both phases use
no-slip walls. Both interior velocity fields start from rest, while alpha
starts uniformly at `0.05`; drag and gravity are disabled. This deliberately
single-fluid-like baseline isolates native alpha transport and the shared
pressure correction before phase-property and boundary-condition differences
are introduced.

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
