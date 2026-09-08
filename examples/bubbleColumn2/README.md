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

The liquid is initially stationary and the volume velocity is initialized as
`uvz=alpha*ugz`. Restart fields are not overwritten.

## Reconstruction and alpha transport

Liquid velocity and slip are reconstructed from

`ul=(uv-alpha*ug)/(1-alpha)`,

`ur=ug-ul=(ug-uv)/(1-alpha)`.

nekRS transports passive scalars with `uv`. The alpha source adds the relative
gas flux so that the intended equation remains

`d(alpha)/dt+div(alpha*ug)=0`.

Because `div(uv)=0`, the pointwise correction is

`Salpha=-(ug-uv).grad(alpha)-alpha*div(ug)`.

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

The effective viscosity is presently an interpolation based on
`(1-alpha)*nuLiquid+alpha*nuGas`. This and the missing alpha-weighted gas
viscous-stress term remain modeling approximations in the one-pass scaffold.

## Drag and virtual mass

Schiller--Naumann drag uses the reconstructed physical slip and constant
`bubbleDiameter`. It is semi-implicit in the gas-velocity scalar equations:
`lambdaD*uv` is explicit and `lambdaD*ug` is placed on the Helmholtz diagonal.

`dragEnabled` and `virtualMassEnabled` are independent numeric switches in
`[CASEDATA]`. Drag defaults on; the explicitly lagged virtual-mass
approximation defaults off.

The gas equation still uses the previous/extrapolated pressure because scalars
are solved before mixture pressure in the one-pass NekRS ordering. Lift,
turbulent dispersion, and wall lubrication remain disabled.
