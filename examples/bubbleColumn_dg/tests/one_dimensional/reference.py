#!/usr/bin/env python3
"""Periodic conservative DGSEM reference; no NekRS/OCCA dependency.

Reconstructed from the documented 1D method, not the recovered historical file.
See README.md for exact initial conditions and reproducibility limits.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import Legendre


def operators(p):
    P = Legendre.basis(p)
    z = np.r_[-1., P.deriv().roots(), 1.]
    w = 2. / (p * (p + 1) * P(z)**2)
    b = np.array([1. / np.prod(z[i] - np.delete(z, i))
                  for i in range(p + 1)])
    D = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        for j in range(p + 1):
            if i != j:
                D[i, j] = b[j] / (b[i] * (z[i] - z[j]))
        D[i, i] = -D[i].sum()
    return z, w, D


def common_flux(left, right, ul, ur, kind, global_speed):
    if kind == "center":
        kind = "central"
    a = .5 * (ul + ur)
    lam = {"central": 0., "upwind": np.abs(a),
           "lf": global_speed}[kind]
    return .5 * a * (left + right) - .5 * lam * (right - left)


def rhs(s, u, h, w, D, kind):
    # Each row is a distinct element; no averaging of shared endpoint states.
    face = common_flux(s[:, -1], np.roll(s[:, 0], -1),
                       u[:, -1], np.roll(u[:, 0], -1),
                       kind, np.max(np.abs(u)))
    # Weak conservative form: M^-1 (D^T W (u*s) - B*fhat).
    r = ((u * s * w) @ D) / w
    r[:, 0] += np.roll(face, 1) / w[0]
    r[:, -1] -= face / w[-1]
    return (2. / h) * r


def rk4(s, dt, evaluate):
    k1 = evaluate(s)
    k2 = evaluate(s + .5 * dt * k1)
    k3 = evaluate(s + .5 * dt * k2)
    k4 = evaluate(s + dt * k3)
    return s + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.


def profile(x, name, delta):
    x = np.mod(x, 1.)
    if name == "sine":
        return .5 + .25 * np.sin(2*np.pi*x)
    if name == "square":
        return ((x > .25) & (x < .75)).astype(float)
    # Periodic sum of smooth top-hat packets. The image terms avoid a seam.
    s = np.zeros_like(x)
    for shift in (-1, 0, 1):
        s += .5 * (np.tanh((x - .25 + shift)/delta)
                   - np.tanh((x - .75 + shift)/delta))
    return s


def self_check(p):
    z, w, D = operators(p)
    B = np.zeros_like(D)
    B[0, 0], B[-1, -1] = -1., 1.
    np.testing.assert_allclose(w[:, None]*D + D.T*w, B, atol=2e-13)
    rng = np.random.default_rng(20261116)
    s = rng.random((5, p+1))
    u = 1. + .2*rng.random(s.shape)
    for flux in ("central", "center", "upwind", "lf"):
        r = rhs(s, u, .2, w, D, flux)
        assert abs(np.sum(r*w)*.1) < 2e-12
        face = common_flux(s[:, -1], np.roll(s[:, 0], -1),
                           u[:, -1], np.roll(u[:, 0], -1),
                           flux, np.max(np.abs(u)))
        strong = -(s*u) @ D.T
        strong[:, 0] += (np.roll(face, 1) - s[:, 0]*u[:, 0])/w[0]
        strong[:, -1] -= (face - s[:, -1]*u[:, -1])/w[-1]
        np.testing.assert_allclose(r, 10*strong, atol=2e-11)
        np.testing.assert_allclose(rhs(np.ones_like(s), np.ones_like(s),
                                      .2, w, D, flux), 0., atol=2e-12)
    np.testing.assert_allclose(rhs(s, np.ones_like(s), .2, w, D, "lf"),
                               rhs(s, np.ones_like(s), .2, w, D, "upwind"))
    # Time accuracy isolated from spatial truncation.
    errs = []
    for n in (10, 20, 40):
        value = 1.
        for _ in range(n):
            value = rk4(value, 1./n, lambda a: -a)
        errs.append(abs(value-np.exp(-1.)))
    assert min(errs[i]/errs[i+1] for i in (0, 1)) > 14.


def run(ne, p, flux, name, periods, cfl, delta, output):
    z, w, D = operators(p)
    h = 1./ne
    x = (np.arange(ne)[:, None] + .5*(z+1)) * h
    u = np.ones_like(x)
    initial = profile(x, name, delta)
    s = initial.copy()
    evaluate = lambda a: rhs(a, u, h, w, D, flux)
    initial_rhs = evaluate(s)
    mass = lambda a: float(.5*h*np.sum(a*w))
    m0 = mass(s)
    # dt*|u|/(h/2)/min(w) <= cfl; every snapshot hits its time exactly.
    nominal = cfl*.5*h*w.min()
    segments = 4*periods
    per_segment = int(np.ceil(.25/nominal))
    dt = .25/per_segment
    steps = segments*per_segment
    tag = f"{name}_{flux}_n{ne}_p{p}"
    folder = output/tag
    folder.mkdir(parents=True, exist_ok=True)
    history = []
    for segment in range(segments+1):
        time = .25*segment
        exact = profile(x-time, name, delta)
        rel = float(np.sqrt(np.sum(w*(s-exact)**2)/np.sum(w*exact**2)))
        history.append(dict(time=time, mass=mass(s), mass_drift=mass(s)-m0,
                            relative_l2=rel, minimum=float(s.min()), maximum=float(s.max())))
        np.savetxt(folder/f"profile_{segment:03d}.csv",
                   np.column_stack((x.ravel(), s.ravel(), exact.ravel())),
                   delimiter=",", header="x,scalar,exact", comments="")
        if segment < segments:
            for _ in range(per_segment):
                s = rk4(s, dt, evaluate)
            assert np.isfinite(s).all()
            assert abs(mass(s)-m0) < 1e-10
    # Whitespace-separated exchange file, one row per ELEMENT-LOCAL 1D node.
    # The production 3D kernel harness checks every transverse copy.
    np.savetxt(folder/"reference.dat",
               np.column_stack((x.ravel(), initial.ravel(),
                                initial_rhs.ravel(), s.ravel())), fmt="%.17e")
    with (folder/"history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metadata = dict(elements=ne, degree=p, flux=flux, profile=name, delta=delta,
                    periods=periods, dt=dt, steps=steps, cfl=cfl,
                    initial_mass=m0, final=history[-1])
    (folder/"config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--flux", nargs="+", choices=["central", "center", "upwind", "lf"],
                        default=["central", "upwind", "lf"])
    parser.add_argument("--profile", nargs="+", choices=["sine", "square", "tanh"],
                        default=["sine", "square", "tanh"])
    parser.add_argument("--periods", type=int, default=1)
    parser.add_argument("--cfl", type=float, default=.1)
    parser.add_argument("--delta", type=float, default=.025)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if (min(args.elements)<2 or args.degree<1 or args.periods<1
            or not 0<args.cfl<=.5 or not 0<args.delta<=.1):
        parser.error("Require elements>=2, degree>=1, periods>=1, 0<cfl<=.5, 0<delta<=.1")
    self_check(args.degree)
    summaries = [run(n, args.degree, f, name, args.periods, args.cfl,
                     args.delta, args.output)
                 for name in args.profile for f in args.flux for n in args.elements]
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    # Convergence is meaningful for smooth profiles; square-wave boundedness
    # and historical percent errors are deliberately not acceptance criteria.
    for name in ("sine", "tanh"):
        rows = sorted((r for r in summaries if r["profile"]==name and r["flux"]=="upwind"),
                      key=lambda r:r["elements"])
        for coarse, fine in zip(rows, rows[1:]):
            if coarse["elements"] != fine["elements"]:
                assert fine["final"]["relative_l2"] < coarse["final"]["relative_l2"]


if __name__ == "__main__":
    main()
