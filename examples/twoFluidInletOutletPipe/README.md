# Two-fluid inlet--outlet pipe with native alpha scalar

This case reuses `examples/turbPipe/turbPipe.re2` (boundary 1 inlet,
boundary 2 outlet, boundary 3 wall). It has no periodic boundary and no
constant-flow-rate control.

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
