"""Run production kernels with OCCA Serial + 1/2 MPI ranks; translate CUDA."""
import os
from pathlib import Path
import subprocess
import tempfile
case=Path(__file__).resolve().parents[1]
repo=case.parents[1]
occa=Path(os.environ.get("OCCA_DIR",repo/"3rd_party/occa"))
if not (occa/"lib/libocca.so").exists():
    raise SystemExit("Set OCCA_DIR to a built OCCA installation; tests were not run.")
env=dict(os.environ,OMPI_ALLOW_RUN_AS_ROOT="1",OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);env["OCCA_CACHE_DIR"]=str(p/"cache")
    exe=p/"transport"
    for nq in (3,4,8):
        subprocess.run(["mpicxx","-std=c++17","-O2","-DDG_NQ="+str(nq),str(case/"tests/transport.cpp"),
            "-I"+str(occa/"include"),"-L"+str(occa/"lib"),"-Wl,-rpath,"+str(occa/"lib"),"-locca","-o",str(exe)],check=True)
        for ranks in (1,2):
            subprocess.run(["mpirun","-np",str(ranks),str(exe),str(case/"alphaDG.okl"),
                str(repo/"src/core/kernels/wDivergenceVolumeHex3D.okl")],env=env,check=True)
    with (p/"dg.cu").open("w") as out:
        subprocess.run([str(occa/"bin/occa"),"translate","-m","CUDA","-D","p_blockSize=64",
            "-D","p_Np=27","-D","p_Nfp=9","-D","p_Nsgeo=19","-D","p_Nvgeo=12",
            "-D","p_NXID=0","-D","p_NYID=1","-D","p_NZID=2","-D","p_WSJID=10",
            "-D","dfloat=double","-D","dlong=int",str(case/"alphaDG.okl")],stdout=out,check=True)
print("Production DG kernels: OCCA Serial/MPI tests and CUDA translation passed.")
print("MPI harness uses test-only dense trace exchange; production OGS pairing is checked at case startup.")
