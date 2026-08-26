# Uniform drag relaxation

This is the first two-fluid regression case.  It uses a fully periodic box,
zero gravity, `u_l=0`, and `u_g=1`.  With no pressure gradient or external
force, Schiller--Naumann drag must reduce the slip while conserving

`alpha_l rho_l u_l + alpha_g rho_g u_g`.

Generate the mesh and run in the usual way:

```bash
genbox <<<'input.box'
nekrs --setup twoFluidDrag.par
```

The printed mixture momentum should remain constant to temporal and linear
solver tolerances, while `abs(u_g-u_l)` decreases monotonically.
