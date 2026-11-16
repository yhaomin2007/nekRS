#!/usr/bin/env python3
"""
Periodic 1D DGSEM + classical RK4.

Conservative: s_t + (u*s)_x = 0
Advective:    s_t + u*s_x   = 0

GLL collocation; no limiter, clipping, filtering or added diffusion.
Independent mathematical reference, not a NekRS/OOGS test.

Outputs:
  profiles.png : time evolution and exact solution
  errors.png   : final nodal errors
  metrics.csv  : error, extrema and mass balance
  *.npz        : numerical snapshots

Dependencies: numpy, matplotlib
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import Legendre

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gll_operators(p):
    """GLL nodes, weights and nodal differentiation matrix."""
    P = Legendre.basis(p)
    r = np.r_[-1.0, P.deriv().roots(), 1.0]
    w = 2.0 / (p * (p + 1) * P(r)**2)

    bary = np.array([
        1.0 / np.prod(r[i] - np.delete(r, i))
        for i in range(p + 1)
    ])
    D = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        for j in range(p + 1):
            if i != j:
                D[i, j] = bary[j] / bary[i] / (r[i] - r[j])
        D[i, i] = -D[i].sum()

    B = np.zeros_like(D)
    B[0, 0], B[-1, -1] = -1.0, 1.0
    defect = w[:, None] * D + D.T * w[None, :] - B
    if np.max(np.abs(defect)) > 1e-10:
        raise RuntimeError("GLL summation-by-parts check failed")
    return r, w, D


class DG1D:
    def __init__(self, elements, degree, flux, form="conservative",
                 variable_velocity=False):
        self.K = elements
        self.p = degree
        self.h = 1.0 / elements
        self.r, self.w, self.D = gll_operators(degree)
        self.x = (
            np.arange(elements)[:, None]
            + 0.5 * (self.r[None, :] + 1.0)
        ) * self.h
        self.mass = np.broadcast_to(
            0.5 * self.h * self.w, self.x.shape
        )
        self.u = np.ones_like(self.x)
        if variable_velocity:
            self.u += 0.2 * np.sin(2 * np.pi * self.x)
        self.flux = flux
        self.form = form

    def integral(self, s):
        return np.sum(self.mass * s)

    def rhs(self, s):
        # Shared +x-oriented interface flux: e right -> e+1 left.
        sm = s[:, -1]
        sp = np.roll(s[:, 0], -1)
        a = 0.5 * (
            self.u[:, -1] + np.roll(self.u[:, 0], -1)
        )

        if self.flux == "central":
            lam = np.zeros_like(a)
        elif self.flux == "upwind":
            lam = np.abs(a)
        elif self.flux == "lf":
            lam = np.full_like(a, np.max(np.abs(self.u)))
        else:
            raise ValueError(self.flux)

        fhat = 0.5 * a * (sm + sp) - 0.5 * lam * (sp - sm)
        f = self.u * s

        if self.form == "conservative":
            result = -(2.0 / self.h) * (f @ self.D.T)
        else:
            result = -(2.0 / self.h) * self.u * (s @ self.D.T)

        # Strong DG surface correction, using LOCAL mass.
        result[:, 0] += (
            2.0 / (self.h * self.w[0])
            * (np.roll(fhat, 1) - f[:, 0])
        )
        result[:, -1] -= (
            2.0 / (self.h * self.w[-1])
            * (fhat - f[:, -1])
        )
        return result

    def evolve(self, s, duration, cfl):
        # This is a script-specific CFL, NOT the NekRS CFL.
        dt_limit = (
            cfl * self.h * self.w.min()
            / np.max(np.abs(self.u))
        )
        steps = max(1, int(np.ceil(duration / dt_limit)))
        dt = duration / steps
        s = s.copy()

        for _ in range(steps):
            k1 = self.rhs(s)
            k2 = self.rhs(s + 0.5 * dt * k1)
            k3 = self.rhs(s + 0.5 * dt * k2)
            k4 = self.rhs(s + dt * k3)
            s += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

            if not np.all(np.isfinite(s)):
                raise RuntimeError("Nonfinite solution; reduce CFL")
        return s


def exact_profile(x, time, kind, delta):
    z = np.mod(np.asarray(x) - time, 1.0)

    if kind == "sine":
        return np.sin(2 * np.pi * z)

    if kind == "step":
        return ((z >= 0.25) & (z < 0.5)).astype(float)

    # Periodized tanh pulse; delta <= 0.1.
    result = np.zeros_like(z)
    for k in range(-4, 5):
        result += 0.5 * (
            np.tanh((z - 0.25 - k) / delta)
            - np.tanh((z - 0.5 - k) / delta)
        )
    return result


def initial_state(dg, kind, delta):
    if kind != "step":
        return exact_profile(dg.x, 0, kind, delta)

    # Jump-aligned mesh: each element receives its interior trace.
    # Opposite sides of a discontinuity intentionally have different values.
    centers = (np.arange(dg.K) + 0.5) * dg.h
    values = ((centers >= 0.25) & (centers < 0.5)).astype(float)
    return np.broadcast_to(values[:, None], dg.x.shape).copy()


def self_tests():
    """Distinguish conservative transport from pure advective transport."""
    rng = np.random.default_rng(42)

    for flux in ("central", "upwind", "lf"):
        dg = DG1D(8, 3, flux)
        assert np.max(np.abs(dg.rhs(np.ones_like(dg.x)))) < 1e-10
        random_state = rng.random(dg.x.shape)
        assert abs(dg.integral(dg.rhs(random_state))) < 1e-10

        cons = DG1D(8, 3, flux, variable_velocity=True)
        assert np.max(np.abs(cons.rhs(1.0 / cons.u))) < 1e-10
        assert abs(cons.integral(cons.rhs(random_state))) < 1e-10

        adv = DG1D(
            8, 3, flux, form="advective", variable_velocity=True
        )
        assert np.max(np.abs(adv.rhs(np.ones_like(adv.x)))) < 1e-10

    print("PASS: SBP, constant preservation, mass cancellation, "
          "variable-velocity steady states")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=16)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--periods", type=int, default=1)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.025)
    parser.add_argument(
        "--profile", choices=("sine", "step", "smooth"),
        default="smooth"
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    if min(args.elements, args.degree, args.periods) < 1:
        parser.error("elements, degree and periods must be positive")
    if not np.isfinite(args.cfl) or args.cfl <= 0:
        parser.error("CFL must be finite and positive")
    if not 0 < args.delta <= 0.1:
        parser.error("delta must be in (0, 0.1]")
    if args.profile == "step" and args.elements % 4:
        parser.error("step test requires elements divisible by 4")

    self_tests()
    args.output.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, float(args.periods), 5)
    fluxes = ("central", "upwind", "lf")
    colors = ("#2468b4", "#d15b24", "#238965")
    fig, axes = plt.subplots(
        5, 3, figsize=(12, 11), sharex=True, sharey=True
    )
    error_fig, error_ax = plt.subplots(figsize=(9, 4))
    records = []
    all_values = []

    for column, (flux, color) in enumerate(zip(fluxes, colors)):
        dg = DG1D(args.elements, args.degree, flux)
        state = initial_state(dg, args.profile, args.delta)
        initial_mass = dg.integral(state)
        snapshots = []

        # Polynomial interpolation for plotting each element independently.
        rq = np.linspace(-1, 1, 101)
        interpolation = (
            np.polynomial.legendre.legvander(rq, dg.p)
            @ np.linalg.inv(
                np.polynomial.legendre.legvander(dg.r, dg.p)
            )
        )
        xplot = (
            np.arange(dg.K)[:, None] + 0.5*(rq[None, :] + 1)
        ) * dg.h

        for row, time in enumerate(times):
            if row:
                state = dg.evolve(
                    state, time - times[row - 1], args.cfl
                )
            snapshots.append(state.copy())

            exact = exact_profile(
                dg.x, time, args.profile, args.delta
            )
            dense_state = state @ interpolation.T
            all_values.extend([
                float(dense_state.min()), float(dense_state.max())
            ])
            mass_error = dg.integral(state) - initial_mass

            # A step's value at its exact jump is convention-dependent.
            # Report L2 only for smooth profiles.
            relative_l2 = ""
            if args.profile != "step":
                relative_l2 = np.sqrt(
                    dg.integral((state-exact)**2)
                    / dg.integral(exact**2)
                )

            records.append(dict(
                flux=flux, time=time,
                relativeL2=relative_l2,
                nodalMin=state.min(), nodalMax=state.max(),
                sampledMin=dense_state.min(),
                sampledMax=dense_state.max(),
                massError=mass_error
            ))

            ax = axes[row, column]
            xx = np.linspace(0, 1, 4001)
            ax.plot(
                xx, exact_profile(xx, time, args.profile, args.delta),
                "k-", lw=1.3, label="Exact"
            )
            for e in range(dg.K):
                ax.plot(
                    xplot[e], dense_state[e], color=color, lw=1.1,
                    label="DG" if e == 0 else None
                )
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 1)
            if row == 0:
                ax.set_title(flux.upper())
                ax.legend(fontsize=8)
            if column == 0:
                ax.set_ylabel(f"t = {time:g}\nscalar s")
            if row == 4:
                ax.set_xlabel("x")

        np.savez(
            args.output / f"{flux}.npz",
            x=dg.x, times=times, solution=np.array(snapshots)
        )
        error_ax.plot(
            dg.x.ravel(), (state-exact).ravel(),
            ":" if flux == "lf" else "-",
            label=flux, color=color
        )
        print(records[-1])

    lo, hi = min(all_values), max(all_values)
    margin = 0.08 * max(hi-lo, 1)
    axes[0, 0].set_ylim(lo-margin, hi+margin)

    fig.suptitle(
        f"1D DGSEM: {args.profile} | u=1 | "
        f"{args.elements} elements | p={args.degree}\n"
        f"Classical RK4 | CFL={args.cfl} | no limiter or added diffusion"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output / "profiles.png", dpi=180)
    plt.close(fig)

    error_ax.set(
        xlabel="x", ylabel="DG - exact",
        title=f"Final nodal error at t={times[-1]:g}"
    )
    error_ax.legend()
    error_ax.grid(alpha=0.2)
    error_fig.tight_layout()
    error_fig.savefig(args.output / "errors.png", dpi=180)
    plt.close(error_fig)

    with (args.output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    largest_balance_error = max(abs(r["massError"]) for r in records)
    print(f"Maximum mass-balance error: {largest_balance_error:.3e}")
    print(f"Results saved to: {args.output.resolve()}")
    print("LF equals upwind here because u=1.")
    print("For step data, nodal error at the exact jump depends on "
          "the chosen point-value convention.")
    print("This reference does not validate NekRS/OOGS/GPU execution.")


if __name__ == "__main__":
    main()
