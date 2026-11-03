#pragma once

namespace bubbleColumn2
{
// NekRS-LS stores the Helmholtz coefficients separately from solve().
// Select this API when available; older NekRS passes them to solve directly.
template<class Solver, class Memory>
inline auto solveGasPressureCorrection(Solver *solver,
    const Memory &diff, const Memory &lambda, const Memory &rhs,
    Memory delta, int)
    -> decltype(solver->coeff0HLM(diff), solver->coeff1HLM(lambda),
                solver->solve(rhs, delta), void())
{
  solver->coeff0HLM(diff);
  solver->coeff1HLM(lambda);
  solver->solve(rhs, delta);
}

template<class Solver, class Memory>
inline auto solveGasPressureCorrection(Solver *solver,
    const Memory &diff, const Memory &lambda, const Memory &rhs,
    Memory delta, long)
    -> decltype(solver->solve(diff, lambda, rhs, delta), void())
{
  solver->solve(diff, lambda, rhs, delta);
}
} // namespace bubbleColumn2
