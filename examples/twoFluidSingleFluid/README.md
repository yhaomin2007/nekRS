# Two-fluid single-fluid equivalence

This regression sets `K=0`, uses identical phase properties, and initializes
both phases with the same periodic divergence-free Taylor--Green field. The two
phase velocities must remain identical and the mixture-continuity residual must
remain below the regression tolerance.

```bash
genbox <<<'input.box'
nekrs --setup twoFluidSingleFluid.par
```
