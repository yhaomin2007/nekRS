#!/usr/bin/env python3
"""Run the supplied reference.py unchanged; export its snapshots for OKL tests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from reference import DG1D, initial_state

SOURCE_ID = "user-supplied-dg1d-2026-11-16"
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--profile", nargs="+", choices=["sine", "step", "smooth"],
                        default=["sine", "step", "smooth"])
    parser.add_argument("--periods", type=int, default=1)
    parser.add_argument("--cfl", type=float, default=.2)
    parser.add_argument("--delta", type=float, default=.025)
    parser.add_argument("--output", type=Path, default=Path("reference-results"))
    args = parser.parse_args()
    if (min(args.elements)<1 or args.degree<1 or args.periods<1
            or not np.isfinite(args.cfl) or args.cfl<=0
            or not 0<args.delta<=.1):
        parser.error("Invalid elements, degree, periods, CFL or delta.")
    if "step" in args.profile and any(n%4 for n in args.elements):
        parser.error("Step profile requires elements divisible by 4.")
    source_hash = hashlib.sha256((HERE/"reference.py").read_bytes()).hexdigest()
    output = args.output.resolve()
    summaries = []
    for name in args.profile:
        for ne in args.elements:
            folder = output/f"{name}_n{ne}_p{args.degree}"
            # All profiles, plotting, interpolation, time integration and
            # metrics are produced by the user's actual script.
            subprocess.run([
                sys.executable, str(HERE/"reference.py"),
                "--elements", str(ne), "--degree", str(args.degree),
                "--profile", name, "--periods", str(args.periods),
                "--cfl", repr(args.cfl), "--delta", repr(args.delta),
                "--output", str(folder)], check=True)
            for flux in ("central", "upwind", "lf"):
                dg = DG1D(ne, args.degree, flux)
                initial = initial_state(dg, name, args.delta)
                with np.load(folder/f"{flux}.npz") as data:
                    x = data["x"].copy()
                    times = data["times"].copy()
                    solution = data["solution"].copy()
                np.testing.assert_array_equal(x, dg.x)
                np.testing.assert_array_equal(solution[0], initial)
                if not np.isfinite(solution).all():
                    raise RuntimeError("Reference contains nonfinite states.")
                if max(abs(float(dg.integral(s)-dg.integral(initial)))
                       for s in solution) > 1e-10:
                    raise RuntimeError("Reference mass drift exceeded 1e-10.")
                # Match evolve()'s actual interval-wise step calculation.
                dt_limit = args.cfl*dg.h*dg.w.min()/np.max(np.abs(dg.u))
                durations = np.diff(times)
                counts = [max(1, int(np.ceil(t/dt_limit))) for t in durations]
                dts = [float(t/n) for t, n in zip(durations, counts)]
                # Integer periods give exactly equal quarter intervals.
                # Reject an unequal schedule instead of approximating it.
                if any(n != counts[0] for n in counts) or any(t != dts[0] for t in dts):
                    raise RuntimeError("Kernel harness requires a uniform RK4 step schedule.")
                fixture = folder/flux
                fixture.mkdir(parents=True, exist_ok=True)
                initial_rhs = dg.rhs(initial)
                np.savetxt(fixture/"reference.dat", np.column_stack((
                    dg.x.ravel(), initial.ravel(), initial_rhs.ravel(),
                    solution[-1].ravel())), fmt="%.17e")
                for index, (time, s) in enumerate(zip(times, solution)):
                    np.savetxt(fixture/f"profile_{index:03d}.csv",
                               np.column_stack((dg.x.ravel(), s.ravel())),
                               delimiter=",", header="x,scalar", comments="")
                config = dict(
                    source_id=SOURCE_ID, reference_sha256=source_hash,
                    elements=ne, degree=args.degree, profile=name, flux=flux,
                    periods=args.periods, delta=args.delta, cfl=args.cfl,
                    times=times.tolist(), steps=sum(counts), dt=dts[0],
                    initial_mass=float(dg.integral(initial)),
                    final_mass=float(dg.integral(solution[-1])),
                    form="conservative", velocity="constant-one")
                (fixture/"config.json").write_text(json.dumps(config, indent=2)+"\n")
                summaries.append(config)
    (output/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    print("Exported the supplied script's exact element-local initial/RHS/final data.")
    print("Run compare_kernels.py to test production kernels; it has not run here.")


if __name__ == "__main__":
    main()
