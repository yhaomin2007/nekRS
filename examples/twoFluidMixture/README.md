# Shared-pressure mixture projection

Both phases start with `u_x=sin(x)`, so the initial mixture velocity has a
nonzero divergence. Momentum solves and drag are disabled. The common pressure
projection alone must reduce

`div(alpha_l u_l + alpha_g u_g)`

below the regression tolerance. Run with:

```bash
genbox <<<'input.box'
nekrs --setup twoFluidMixture.par
```

The case prints the mixture-divergence norm after every step.
