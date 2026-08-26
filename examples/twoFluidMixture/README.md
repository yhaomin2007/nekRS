# Shared-pressure mixture projection

Both phases start with `u_x=sin(x)`, so the initial mixture velocity has a
nonzero divergence.  The common pressure projection should reduce

`div(alpha_l u_l + alpha_g u_g)`

to the pressure/velocity splitting error. Run with:

```bash
genbox <<<'input.box'
nekrs --setup twoFluidMixture.par
```

The case prints the mixture-divergence norm after every step.
