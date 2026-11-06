#!/usr/bin/env python3
"""Check OCCA's actual CUDA lowering; nvcc is not required for this test."""
import os
from pathlib import Path
import re
import shutil
import subprocess
root=Path(__file__).resolve().parents[1]
occa=os.environ.get('OCCA_BIN') or shutil.which('occa')
if not occa:
    raise SystemExit('Set OCCA_BIN to the installed occa executable to run this test.')
defines={'dfloat':'double','dlong':'int','p_blockSize':256,'p_Nfaces':6,
         'p_Nfp':36,'p_Nsgeo':19,'p_NXID':0,'p_NYID':1,'p_NZID':2,'p_WSJID':10}
command=[occa,'translate','-m','CUDA']
for key,value in defines.items():
    command+=['-D',f'{key}={value}']
result=subprocess.run(command+[str(root/'bubbleColumn4Equations.okl')],
                      check=True,capture_output=True,text=True)
assert not re.search(r'\batomicSub\s*\(',result.stdout), 'CUDA floating atomicSub is unsupported'
assert re.search(r'\batomicAdd\s*\(\s*&rhs\[i\]',result.stdout), 'Expected additive pressure-surface atomic'
print('PASS: real OCCA CUDA translation uses atomicAdd for negative FP64 boundary flux, with no atomicSub')
