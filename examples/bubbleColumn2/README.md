# bubbleColumn2: uploaded QG working baseline

Primary fields are native volume velocity `uv`, pressure, `ALPHA`, and
`QGX/QGY/QGZ`, with `qg=alpha*ug` and `ul=(uv-qg)/(1-alpha)`.

The seven user-uploaded files are preserved verbatim in commit `4f79c46`.
The following repair retains the uploaded `.par`, `.usr`, and both boundary
files, all switches, HPFRT, outlet damping, gas masks, alpha/vector clipping,
phase-velocity exports, stability monitors and conservation CSV columns.
Gas pressure remains lagged; no new-pressure gas correction is added.

## Numerical repairs

- Cancel the exact native scalar-advection kernel for alpha and all QG
  components, including cubature and predicted contravariant velocity, rather
  than a pointwise approximation. Conservative flux divergences are unchanged.
  Nonzero scalar subcycling is rejected because its forcing ordering differs.
- Substitute `ul=(uv-qg)/beta` in gas drag: `lambda=Ki/(rhoGas*beta)`
  is implicit, while `alpha*lambda*uv` is explicit. The mixture reaction rate
  `alpha*(1-rhoGas/rhoLiquid)*lambda` is also put on its native Helmholtz
  diagonal. This remains segregated drag and retains the native `Ap` pressure
  mobility; it is not a block pressure/drag correction.
- Add the volume-mixture physical two-phase stress minus the native base stress
  as a lagged correction. Extra native viscosity from outlet damping remains.
  Existing explicit physical gas stress and scalar numerical diffusion remain.
- Initialize virtual-mass scratch before its first use, avoiding undefined
  data even when virtual mass is disabled.
- Use `alphaInitial` as the background plume fraction. With the supplied value
  zero, initialization is identical to the uploaded case, including initially
  stationary liquid and a localized, initially non-solenoidal volume velocity.

Clipping and masks retain their original nonconservative changes and accounting
in the CSV. No global redistribution limiter is introduced. Gas stress and
mixture stress corrections remain explicit and may restrict the timestep.

## Validation status

CPU-translated pointwise-kernel syntax/algebra and baseline-preservation checks
are available during development. Full NekRS MPI/GPU compilation and short-run
validation are required before judging stability or permissible timestep.

## Mixture viscous correction switch

`[CASEDATA] mixtureViscousCorrectionEnabled = 0.0` disables the explicit
physical two-phase stress minus native base stress correction. Set it to `1.0`
to recover that correction from commit `1179116`. The option defaults to zero
even when omitted. Native implicit viscosity, outlet damping, gas stress,
drag, filters, masks, diagnostics and the lagged gas pressure remain unchanged.
