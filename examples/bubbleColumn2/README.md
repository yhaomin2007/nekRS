# bubbleColumn2

One-pass Eulerian two-phase scaffold using the phase-volume averaged velocity

`uv=(1-alpha)*ul+alpha*ug`

as the native nekRS fluid velocity. For incompressible phases without phase
change, adding the two phase-volume equations gives the exact constraint

`div(uv)=0`.

An empty `userDivergence` callback is registered so that NekRS zero-fills
`fluid->o_div` before every pressure solve. The callback adds no source, so no
nonzero divergence or divergence filter is used in this example.

## Boundary map and initialization

The column axis and upward inlet-flow direction are `+z`; gravity acts in
`-z`. The future mesh must expose these physical boundary IDs:

| ID | Patch | Volume velocity | `alpha` | Gas velocity |
|---:|---|---|---|---|
| 1 | inlet | `uvz=alphaInlet*ugInlet` | fixed inlet value | fixed vertical value |
| 2 | outlet | zero normal gradient | zero normal gradient | zero normal gradient |
| 3 | wall | no slip | zero normal gradient | no slip |

At a fresh start, `alpha` and `ugz` share the smooth plume profile

`f(z)=0.5*(1-tanh((z-initialPlumeHeight)/initialPlumeThickness))`.

The background fraction is `alphaInitial`, blended into the plume as
`alpha=alphaInitial+(alphaInlet-alphaInitial)*f(z)`. The initial volume
velocity is zero (divergence free in the interior); liquid counterflow follows
from reconstruction. The inlet starts the prescribed volume flux on advancement.
Restart fields are not overwritten.

## Reconstruction and alpha transport

Liquid velocity and slip are reconstructed from

`ul=(uv-alpha*ug)/(1-alpha)`,

`ur=ug-ul=(ug-uv)/(1-alpha)`.

The alpha source cancels the exact native scalar advection kernel (including
its cubature choice) and replaces it by the assembled conservative divergence
of `alpha*ug`. Scalar subcycling is rejected because this replacement uses the
non-subcycled source ordering. No core solver changes are required.

After scalar advancement, an optional overshoot triggers a bounded global
projection into `[0,1-alphaFloor]`. It preserves the transported SEM quadrature
gas volume and leaves Dirichlet nodes unchanged. This is a nonlocal redistribution,
not a local flux-corrected transport scheme. Infeasible gas volumes stop the run
rather than silently discarding mass. The projection uses host transfers and
collective reductions only when bounds are violated.

The scalar `diffusionCoeff` and `transportCoeff` values are read directly from
their four `.par` sections; `userProperties()` does not overwrite them. Each
implicit numerical diffusion can be canceled with a lagged explicit volume
term using the `[CASEDATA]` switches

`subtractAlphaDiffusion`, `subtractUgxDiffusion`,
`subtractUgyDiffusion`, and `subtractUgzDiffusion`.

For a switch value of one, the corresponding explicit source receives

`-div(diffusionCoeff*grad(s))`.

Thus the new-time diffusion remains implicit for conditioning while its
previous-time contribution is removed from the intended equation. Set a switch
to zero to retain that scalar's numerical diffusion. This is an IMEX deferred
correction: it does not algebraically cancel new-time or boundary diffusion and
may reduce the stabilization obtained from a large `diffusionCoeff`.

## Volume-mixture momentum and pressure

Dividing each phase momentum equation by its constant phase density, then
adding with phase-volume weights, gives the pressure mobility

`Ap=(1-alpha)/rhoLiquid+alpha/rhoGas`.

The code stores `rhoEffective=1/Ap` in the nekRS fluid-density property, so the
native pressure projection uses `div(Ap*grad(p))` while enforcing zero
divergence.

The volume-mixture convective flux contains the kinematic drift tensor

`Tdrift=alpha*(1-alpha)*ur*ur`

which is evaluated equivalently as

`Tdrift=alpha/(1-alpha)*(ug-uv)*(ug-uv)`.

Unlike mass-weighted mixture momentum, interphase forces do not cancel after
the phase equations are divided by their densities. If `agI` is the gas
interphase acceleration, the retained volume-mixture contribution is

`alpha*(1-rhoGas/rhoLiquid)*agI`.

NekRS retains an implicit variable-viscosity stress operator based on

`nuEffective=(1-alpha)*muLiquid/rhoLiquid+alpha*muGas/rhoGas`,

`muEffective=rhoEffective*nuEffective`.

The exact volume-mixture viscous acceleration is reconstructed from the two
phase stresses,

`Vexact=div((1-alpha)*tauLiquid/rhoLiquid+alpha*tauGas/rhoGas)`.

To avoid double counting, the explicit mixture RHS receives only

`Vcorrection=Vexact-div(tauNative)/rhoEffective`.

Here all three stresses use the deviatoric Newtonian form
`mu*(grad(u)+grad(u)^T-2/3*div(u)*I)`. The `.par` file enables NekRS's
`navierStokes+variableViscosity` stress formulation so that `tauNative` matches
the operator being corrected. The correction is time-lagged while the base
effective diffusion remains implicit.

## Drag and virtual mass

Schiller--Naumann drag uses the reconstructed physical slip and constant
`bubbleDiameter`. It is semi-implicit in the gas-velocity scalar equations:
`lambdaD*uv` is explicit and `lambdaD*ug` is placed on the Helmholtz diagonal.
The mixture drag is also split: `c*lambdaD*ug` is explicit and
`rhoEffective*c*lambdaD*uv` is implicit, where `c=alpha*(1-rhoGas/rhoLiquid)`.
The source-stage rate is frozen and multiplied by the refreshed effective density.
This is segregated semi-implicit drag, not a block-coupled phase response;
pressure mobility remains `Ap` and no new-pressure gas correction is added.

`dragEnabled` and `virtualMassEnabled` are independent numeric switches in
`[CASEDATA]`. Drag defaults on; the explicitly lagged virtual-mass
approximation defaults off.

The gas equation still uses the previous/extrapolated pressure because scalars
are solved before mixture pressure in the one-pass NekRS ordering. Its phase-stress acceleration is included explicitly as
`div(alpha*tauGas)/(alpha*rhoGas)` where `alpha>alphaFloor`; below that threshold,
the absent-phase stress acceleration is zero. Numerical scalar diffusion remains
separate from this physical stress. The default retains `1e-5` numerical diffusion
for alpha and all gas components, with subtraction switches off.
Lift, turbulent dispersion, and wall lubrication remain disabled.
