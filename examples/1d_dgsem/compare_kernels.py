#!/usr/bin/env python3
"""Compile actual production kernels and compare against reference.py outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import shutil
import sys
import uuid
from datetime import datetime, timezone

here = Path(__file__).resolve().parent
repo = here.parents[1]
case = repo / "examples" / "bubbleColumn_dg"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("reference-results"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--flux", nargs="+", choices=["central", "upwind", "lf"],
                        default=["central", "upwind", "lf"])
    parser.add_argument("--device", default="{mode: 'Serial'}",
                        help="OCCA device properties; e.g. {mode: 'CUDA', device_id: 0}")
    args = parser.parse_args()
    configs = sorted(args.results.resolve().glob("*/*/config.json"))
    configs = [file for file in configs if json.loads(file.read_text())["flux"] in args.flux]
    if not configs:
        parser.error("No reference results found. Run generate_reference.py first.")
    if min(args.ranks)<1:
        parser.error("MPI ranks must be positive.")
    occa = Path(os.environ.get("OCCA_DIR", repo/"3rd_party/occa")).resolve()
    if not (occa/"lib/libocca.so").exists():
        parser.error("Set OCCA_DIR to a built OCCA installation (lib/libocca.so).")
    reference_hash = hashlib.sha256((here/"reference.py").read_bytes()).hexdigest()
    for file in configs:
        c = json.loads(file.read_text())
        if (c.get("source_id")!="user-supplied-dg1d-2026-11-16"
                or c.get("reference_sha256")!=reference_hash
                or c.get("form")!="conservative"
                or c.get("velocity")!="constant-one"):
            parser.error("Stale or incompatible fixture; rerun generate_reference.py.")
        if c["degree"]<1 or any(c["elements"]%n for n in args.ranks):
            parser.error("Kernel comparison requires p>=1 and elements divisible by every rank count.")
    build = args.results.resolve()/"kernel-build"
    build.mkdir(parents=True, exist_ok=True)
    executables = {}
    for degree in sorted({json.loads(file.read_text())["degree"] for file in configs}):
        exe = build/f"kernel_reference_p{degree}"
        subprocess.run(shlex.split(os.environ.get("MPICXX", "mpicxx")) + [
            "-std=c++17", "-O2", f"-DDG_NQ={degree+1}", str(here/"kernel_reference.cpp"),
            "-I"+str(occa/"include"), "-L"+str(occa/"lib"),
            "-Wl,-rpath,"+str(occa/"lib"), "-locca", "-o", str(exe)], check=True)
        executables[degree] = exe
    env = dict(os.environ)
    env.setdefault("OCCA_CACHE_DIR", str(build/"occa-cache"))
    # Honor the user's scheduler/MPI setup; do not silently grant root execution.
    launcher = shlex.split(os.environ.get("MPIEXEC", "mpirun"))
    for file in configs:
        c = json.loads(file.read_text())
        exe = executables[c["degree"]]
        flux = {"central": 0, "center": 0, "upwind": 1, "lf": 2}[c["flux"]]
        for ranks in args.ranks:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run = file.parent/f"kernel_np{ranks}"/f"run_{stamp}_{uuid.uuid4().hex[:8]}"
            run.mkdir(parents=True)
            shutil.copy2(file.parent.parent/f"{c['flux']}.npz", run/"reference.npz")
            metadata = dict(config=c, ranks=ranks, device=args.device, status="running",
                            created_utc=stamp)
            def save_status(status):
                metadata["status"] = status
                temporary = run/"run.json.tmp"
                temporary.write_text(json.dumps(metadata, indent=2)+"\n")
                temporary.replace(run/"run.json")
            save_status("running")
            command = launcher + ["-np", str(ranks), str(exe),
                str(case/"alphaDG.okl"),
                str(repo/"src/core/kernels/wDivergenceVolumeHex3D.okl"),
                str(c["elements"]), str(flux), str(c["steps"]), repr(c["dt"]),
                str(file.parent/"reference.dat"),
                str(run/"kernel"), args.device]
            print(f"Job output: {run}", flush=True)
            try:
                # Persist stdout during the job, including each EXPORT event.
                with (run/"job.log").open("w", buffering=1) as log:
                    completed = subprocess.run(command, env=env, text=True,
                                               stdout=log, stderr=subprocess.STDOUT)
                completed.check_returncode()
            except BaseException:
                save_status("failed")
                raise
            save_status("completed")
            subprocess.run([sys.executable, str(here/"postprocess.py"),
                            "--run", str(run), "--check"], check=True)
    print("PASS: production OKL RHS/final states match Python on every 3D node.")
    print("Trace exchange uses test-only MPI_Allreduce, NOT production OGS.")

if __name__ == "__main__":
    main()
