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
assert 'correctKernel(' not in native and 'strongGrad(' not in native
assert native.index('pressureSolver->solve(')<native.index('solveNativePressureVelocities(time,pressure)')
assert 'if(log)' in native and 'nativePressureAction(pressure,work)' in native
main=header.split('inline void solvePressureAndCorrect(',1)[1]
assert 'solveNativePhasePressure(time,step,log,before,tau);' in main
assert 'method3Projection::solve(' not in header and 'correctKernel(' not in header
assert 'bubbleColumn3Krylov.hpp' not in udf
assert udf.count('pressureSolver')==0 # no installation as default fluid pressure solver
print('PASS: native mode default, one native solve, latest pressure corrects both phases, custom GCR bypassed')

load=header.split('inline void addNativePhasePressureLoad(',1)[1].split('inline void solveNativePressureVelocities(',1)[0]
assert 'core-wGradientVolumeHex3D' in load and '1/p.rhoGas' in load and '1/p.rhoLiquid' in load
core=(root.parents[1]/'src/solver/fluid/fluidSolver.cpp').read_text()
velocity=core.split('void fluidSolver_t::solveVelocity(',1)[1].split('void fluidSolver_t::setupEllipticSolver',1)[0]
assert 'core-wGradientVolumeHex3D' in velocity and 'ellipticSolver.at(0)->solve' in velocity
print('PASS: native velocity and both phase RHSs use the identical weak gradient kernel before Helmholtz')
