#!/usr/bin/env python3
"""Compare time-resolved production-kernel exports with the supplied DGSEM script."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reference import exact_profile, gll_operators


def process(run, allow_partial=False, tolerance=1e-9):
    run = Path(run)
    meta = json.loads((run/"run.json").read_text())
    c, ranks = meta["config"], meta["ranks"]
    source_hash = hashlib.sha256(Path(__file__).with_name("reference.py").read_bytes()).hexdigest()
    if c["reference_sha256"] != source_hash:
        raise ValueError(f"{run}: reference source hash mismatch")
    if not allow_partial and meta["status"] != "completed":
        raise ValueError(f"{run}: job status is {meta['status']}; use --allow-partial to inspect")
    K, p = c["elements"], c["degree"]
    nq = p+1
    z, w, _ = gll_operators(p)
    with np.load(run/"reference.npz") as data:
        x, times, reference = data["x"], data["times"], data["solution"]
    if x.shape != (K, nq) or reference.shape != (5, K, nq):
        raise ValueError("Unexpected reference array shape")
    np.testing.assert_allclose(times, c["times"], rtol=0, atol=1e-14)
    if c["steps"] % 4:
        raise ValueError("Export schedule must have four equal step intervals")
    mass1d = np.broadcast_to(w/(2*K), x.shape)
    expected_x = (np.arange(K)[:, None]+.5*(z+1))/K
    np.testing.assert_allclose(x, expected_x, rtol=0, atol=1e-13)
    initial_mass = float(np.sum(reference[0]*mass1d))
    records, states, indices = [], [], []
    for index, time in enumerate(times):
        paths = [run/f"kernel.snapshot{index}.rank{rank}.csv" for rank in range(ranks)]
        if not all(path.exists() for path in paths):
            if allow_partial:
                continue  # only read an instant once all ranks have published it
            raise ValueError(f"{run}: missing rank export at snapshot {index}")
        rows = np.concatenate([np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
                               for path in paths])
        required = {"step", "time", "element", "i", "j", "k", "x", "mass", "scalar"}
        if not required.issubset(rows.dtype.names or ()):
            raise ValueError("Unexpected CSV columns")
        if any(not np.isfinite(rows[name]).all() for name in required):
            raise ValueError("Nonfinite exported value")
        ids = []
        for name, upper in (("element", K), ("i", nq), ("j", nq), ("k", nq)):
            value = rows[name].astype(int)
            if np.any(rows[name] != value) or np.any(value<0) or np.any(value>=upper):
                raise ValueError(f"Invalid {name} index")
            ids.append(value)
        e, i, j, k = ids
        key = ((e*nq+k)*nq+j)*nq+i
        if len(key) != K*nq**3 or len(np.unique(key)) != len(key):
            raise ValueError("Duplicate or missing element-local nodes across ranks")
        np.testing.assert_allclose(rows["time"], time, rtol=0, atol=1e-12)
        if np.any(rows["step"] != index*(c["steps"]//4)):
            raise ValueError("Snapshot step does not match reference schedule")
        np.testing.assert_allclose(rows["x"], x[e, i], rtol=0, atol=1e-13)
        expected_mass = w[i]*w[j]*w[k]/(8*K)
        np.testing.assert_allclose(rows["mass"], expected_mass, rtol=1e-12, atol=1e-15)
        values = rows["scalar"]
        difference = values-reference[index, e, i]
        mass = float(np.sum(rows["mass"]*values))
        error_l2 = float(np.sqrt(np.sum(rows["mass"]*difference**2)))
        ref_norm = float(np.sqrt(np.sum(rows["mass"]*reference[index, e, i]**2)))
        # Only select a transverse slice AFTER checking all 3D nodes.
        full = np.empty((K, nq, nq, nq))
        full[e, k, j, i] = values
        line = full[:, 0, 0, :]
        transverse_error = float(np.max(np.abs(full-line[:, None, None, :])))
        exact = exact_profile(x, float(time), c["profile"], c["delta"])
        exact_l2 = ""
        if c["profile"] != "step":
            exact_l2 = float(np.sqrt(np.sum(mass1d*(line-exact)**2)/np.sum(mass1d*exact**2)))
        records.append(dict(
            snapshot=index, time=float(time), maxAbsDifference=float(np.max(np.abs(difference))),
            absoluteL2Difference=error_l2,
            relativeL2Difference=error_l2/ref_norm if ref_norm else "",
            relativeL2Exact=exact_l2, mass=mass, massError=mass-initial_mass,
            referenceMassError=float(np.sum(mass1d*reference[index])-initial_mass),
            nodalMin=float(values.min()), nodalMax=float(values.max()),
            transverseMaxDifference=transverse_error))
        states.append(line.copy())
        indices.append(index)
    if not records:
        raise ValueError(f"{run}: no complete snapshots available yet")
    complete = len(records)==5 and meta["status"]=="completed"
    out = run/"postprocess"
    out.mkdir(exist_ok=True)
    with (out/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    np.savez(out/"comparison.npz", x=x, times=times[indices],
             kernel=np.array(states), reference=reference[indices])
    rq = np.linspace(-1, 1, 101)
    interp = np.polynomial.legendre.legvander(rq, p) @ np.linalg.inv(
        np.polynomial.legendre.legvander(z, p))
    xp = (np.arange(K)[:, None]+.5*(rq+1))/K
    fig, axes = plt.subplots(len(records), 2, figsize=(12, 2.6*len(records)), squeeze=False)
    for row, (index, state) in enumerate(zip(indices, states)):
        ax, err_ax = axes[row]
        xx = np.linspace(0, 1, 2001)
        ax.plot(xx, exact_profile(xx, times[index], c["profile"], c["delta"]),
                color="black", lw=1, label="Exact")
        dense, ref_dense = state@interp.T, reference[index]@interp.T
        for element in range(K):
            ax.plot(xp[element], ref_dense[element], color="tab:orange", lw=1.4,
                    label="Python DGSEM" if element==0 else None)
            ax.plot(xp[element], dense[element], color="tab:blue", ls="--",
                    label="Production kernel" if element==0 else None)
            err_ax.plot(x[element], state[element]-reference[index, element], color="tab:red")
        ax.set_ylabel(f"t={times[index]:g}\nscalar")
        err_ax.set_ylabel("Kernel - Python")
        for axis in (ax, err_ax):
            axis.set_xlim(0, 1)
            axis.set_xlabel("x")
            axis.grid(alpha=.25)
        if row==0:
            ax.legend(fontsize=8)
    label = "COMPLETE" if complete else "PARTIAL"
    fig.suptitle(f"{label}: {c['profile']} / {c['flux']} / N={K}, p={p}, ranks={ranks}")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out/"profiles.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    t = [r["time"] for r in records]
    axes[0].plot(t, [r["massError"] for r in records], "o-", label="Kernel")
    axes[0].plot(t, [r["referenceMassError"] for r in records], "x--", label="Python")
    axes[0].set_ylabel("Mass change (signed)")
    axes[0].legend()
    axes[1].plot(t, [r["maxAbsDifference"] for r in records], "o-")
    axes[1].set_ylabel("Max |kernel - Python|")
    for ax in axes:
        ax.set_xlabel("Time")
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/"diagnostics.png", dpi=180)
    plt.close(fig)
    within_tolerance = all(r["maxAbsDifference"]<=tolerance and abs(r["massError"])<=1e-10
                           for r in records)
    summary = dict(complete=complete, within_tolerance=within_tolerance,
                   passed=complete and within_tolerance, snapshots=len(records),
                   max_difference=max(r["maxAbsDifference"] for r in records),
                   max_mass_drift=max(abs(r["massError"]) for r in records))
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"{run}: {json.dumps(summary)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="One job directory containing run.json")
    parser.add_argument("--results", type=Path, default=Path("reference-results"))
    parser.add_argument("--allow-partial", action="store_true",
                        help="Plot fully written times from a running/failed job; never mark it passed")
    parser.add_argument("--check", action="store_true", help="Exit nonzero unless complete and within tolerance")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance<=0:
        parser.error("Tolerance must be finite and positive")
    runs = [args.run] if args.run else sorted(
        p.parent for p in args.results.glob("*/*/kernel_np*/run_*/run.json"))
    if not runs:
        parser.error("No job exports found. Run compare_kernels.py first.")
    summaries = [process(run, args.allow_partial, args.tolerance) for run in runs]
    if args.check and not all(s["passed"] for s in summaries):
        raise SystemExit("Comparison failed or incomplete; see postprocess/metrics.csv")


if __name__ == "__main__":
    main()
