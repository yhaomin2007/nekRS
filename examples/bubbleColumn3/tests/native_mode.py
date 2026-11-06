#!/usr/bin/env python3
"""Guard native-only dispatch and the absence of nested/custom pressure solves."""
from pathlib import Path
import configparser
root=Path(__file__).resolve().parents[1]
header=(root/'bubbleColumn3Pressure.hpp').read_text()
udf=(root/'bubbleColumn3.udf').read_text()
par=configparser.ConfigParser();par.read_string('[ROOT]\n'+(root/'bubbleColumn3.par').read_text())
assert par['CASEDATA']['pressureSolveMode']=='native'
assert 'std::string pressureSolveMode="native"' in udf
native=header.split('inline void solveNativePhasePressure(',1)[1].split('inline void solvePressureAndCorrect(',1)[0]
assert native.count('pressureSolver->solve(')==1
assert 'method3Projection::solve' not in native and 'precondition(' not in native
assert 'correctKernel(m->Nlocal,O,betaG' in native and 'correctKernel(m->Nlocal,O,betaL' in native
assert native.index('pressureSolver->solve(')<native.index('correctKernel(')
assert 'if(log)' in native and 'nativePressureAction(pressure,work)' in native
main=header.split('inline void solvePressureAndCorrect(',1)[1]
assert main.index('solveNativePhasePressure(time,step,log,before,tau);return;')<main.index('method3Projection::solve(')
assert udf.count('pressureSolver')==0 # no installation as default fluid pressure solver
print('PASS: native mode default, one native solve, latest pressure corrects both phases, custom GCR bypassed')
