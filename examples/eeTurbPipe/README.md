# eeTurbPipe

Initial Euler-Euler pressure/velocity verification case based on `examples/turbPipe`.

For this first stage the two phases are deliberately identical:

- `alpha_g = 0.1`, `alpha_l = 0.9` (fixed; no alpha transport equation)
- identical phase density and viscosity
- identical velocity boundary conditions
- no interphase drag/lift/virtual-mass/dispersion/wall-lubrication terms
- no-slip pipe wall for both phases

The required reduction test is

`u_l == u_g == u_single`

and therefore

`u_m = alpha_l*u_l + alpha_g*u_g == u_single`,

with the same pressure field as the standard single-phase turbPipe case, within numerical tolerance.

This case intentionally reuses the turbPipe mesh.  Copy or symlink `../turbPipe/turbPipe.re2` to `eeTurbPipe.re2` before running if the launcher requires a case-local mesh file.

The first implementation stage is pressure/velocity-only.  Do not add an alpha equation until this reduction test passes.
