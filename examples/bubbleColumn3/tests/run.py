#!/usr/bin/env python3
"""Run production pointwise OKL equations as serial C++ (no NekRS install)."""
from pathlib import Path
import re
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
source=(root/'bubbleColumn3Equations.okl').read_text()
source=source.replace('@kernel','').replace('@restrict','').replace('@atomic','')
source=re.sub(r';\s*@tile\([^;\n]*?\)\)',')',source)
source=re.sub(r';\s*@(?:outer|inner)\(\d+\)', '', source)
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    (p/'equations_serial.hpp').write_text(source)
    subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I'+tmp,
                    str(root/'tests/kernels.cpp'),'-o',str(p/'test')],check=True)
    subprocess.run([str(p/'test')],check=True)

# Native v26 creates scalar elliptic solvers AFTER UDF_Setup. Do not call
# applyDirichlet there: it dereferences their boundary masks.
udf=(root/'bubbleColumn3.udf').read_text()
setup=udf.split('void UDF_Setup()\n{',1)[1].split('void UDF_ExecuteStep',1)[0]
begin=udf.split('inline void beginStep(double time)',1)[1].split('inline void postAlpha',1)[0]
assert 'applyDirichlet(' not in setup
assert begin.index('s->applyDirichlet(time)') < begin.index('reconstructMixture();convection')
print('PASS: initial phase boundary application deferred until scalar elliptic masks exist')

assert 'missingCubatureInterpolationHeader(N,C)' in udf
assert 'setupPressure();' not in setup
assert 'dragLoadKernel(' not in begin  # partner loads never enter EXT history
pressure=(root/'bubbleColumn3Pressure.hpp').read_text()
assert 'if(!pressureSolver) setupPressure();' in pressure
assert 'validateLinearSolve(pressureSolver.get()' in pressure
assert 'reconstructMixture();solvePressureAndCorrect' in udf
print('PASS: unconditional core cubature registration, lazy pressure setup, unextrapolated drag, and linear convergence guards')

# Installed v26 sha:96b3cf9 rejects `poisson` in an elliptic PAR section.
# Its internal kernel-registration option must be set by Setup0 instead.
import configparser
par=configparser.ConfigParser()
par.read_string('[ROOT]\n'+(root/'bubbleColumn3.par').read_text())
assert 'poisson' not in par['ELLIPTIC phasePressure']
setup0=udf.split('void UDF_Setup0(',1)[1].split('void UDF_LoadKernels(',1)[0]
assert 'setArgs("ELLIPTIC PHASEPRESSURE HELMHOLTZ TYPE","POISSON")' in setup0
assert 'setArgs("ELLIPTIC PHASEPRESSURE ELLIPTIC COEFF FIELD","TRUE")' in setup0
print('PASS: v26 pressure PAR compatibility and pre-registration Poisson option')
