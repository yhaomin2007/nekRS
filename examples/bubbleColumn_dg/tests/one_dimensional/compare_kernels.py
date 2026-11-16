#!/usr/bin/env python3
"""Compile actual production kernels and compare against reference.py outputs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

here = Path(__file__).resolve().parent
case = here.parents[1]
repo = case.parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", default="{mode: 'Serial'}",
                        help="OCCA device properties; e.g. {mode: 'CUDA', device_id: 0}")
    args = parser.parse_args()
    configs = sorted(args.results.resolve().glob("*/config.json"))
    if not configs:
        parser.error("No reference results found. Run reference.py first.")
    if min(args.ranks)<1:
        parser.error("MPI ranks must be positive.")
    occa = Path(os.environ.get("OCCA_DIR", repo/"3rd_party/occa")).resolve()
    if not (occa/"lib/libocca.so").exists():
        parser.error("Set OCCA_DIR to a built OCCA installation (lib/libocca.so).")
    for file in configs:
        c = json.loads(file.read_text())
        if c["degree"]!=3 or any(c["elements"]%n for n in args.ranks):
            parser.error("Kernel comparison requires p=3 and elements divisible by every rank count.")
    build = args.results.resolve()/"kernel-build"
    build.mkdir(parents=True, exist_ok=True)
    exe = build/"kernel_reference"
    subprocess.run(shlex.split(os.environ.get("MPICXX", "mpicxx")) + [
        "-std=c++17", "-O2", "-DDG_NQ=4", str(here/"kernel_reference.cpp"),
        "-I"+str(occa/"include"), "-L"+str(occa/"lib"),
        "-Wl,-rpath,"+str(occa/"lib"), "-locca", "-o", str(exe)], check=True)
    env = dict(os.environ)
    env.setdefault("OCCA_CACHE_DIR", str(build/"occa-cache"))
    # Honor the user's scheduler/MPI setup; do not silently grant root execution.
    launcher = shlex.split(os.environ.get("MPIEXEC", "mpirun"))
    for file in configs:
        c = json.loads(file.read_text())
        flux = {"central": 0, "center": 0, "upwind": 1, "lf": 2}[c["flux"]]
        for ranks in args.ranks:
            command = launcher + ["-np", str(ranks), str(exe),
                str(case/"alphaDG.okl"),
                str(repo/"src/core/kernels/wDivergenceVolumeHex3D.okl"),
                str(c["elements"]), str(flux), str(c["steps"]), repr(c["dt"]),
                str(file.parent/"reference.dat"),
                str(file.parent/f"kernel_np{ranks}"), args.device]
            completed = subprocess.run(command, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (file.parent/f"kernel_np{ranks}.log").write_text(completed.stdout)
            print(completed.stdout, end="")
            completed.check_returncode()
    print("PASS: production OKL RHS/final states match Python on every 3D node.")
    print("Trace exchange uses test-only MPI_Allreduce, NOT production OGS.")

if __name__ == "__main__":
    main()
