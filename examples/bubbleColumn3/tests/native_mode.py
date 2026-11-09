#!/usr/bin/env python3
"""Check native reference dispatch and native pressure loads in both paths."""
from pathlib import Path
import configparser
root=Path(__file__).resolve().parents[1]
header=(root/'bubbleColumn3Pressure.hpp').read_text()
udf=(root/'bubbleColumn3.udf').read_text()
par=configparser.ConfigParser();par.read_string('[ROOT]\n'+(root/'bubbleColumn3.par').read_text())
assert par['CASEDATA']['pressureSolveMode']=='consistent'
assert 'std::string pressureSolveMode="consistent"' in udf
native=header.split('inline void solveNativePhasePressure(',1)[1].split('inline void formPressureRhs(',1)[0]
assert native.count('pressureSolver->solve(')==1
assert 'method3Projection::solve' not in native and 'precondition(' not in native
assert 'correctKernel(' not in native and 'strongGrad(' not in native
assert native.index('pressureSolver->solve(')<native.index('solveNativePressureVelocities(time,pressure)')
assert 'if(log)' in native and 'nativePressureAction(pressureUpdate,work)' in native
main=header.split('inline void solvePressureAndCorrect(',1)[1]
assert 'solveNativePhasePressure(time,step,log,before,tau,incremental);' in main
assert 'solveConsistentPhasePressure(time,step,log,before,tau,incremental,havePreviousPressure);' in main
assert 'correctKernel(' not in header
assert udf.count('pressureSolver')==0 # no installation as default fluid pressure solver
print('PASS: consistent mode default and native reference remains one Poisson solve')

load=header.split('inline void addNativePhasePressureLoad(',1)[1].split('inline void solveNativePressureVelocities(',1)[0]
assert 'core-wGradientVolumeHex3D' in load and '1/p.rhoGas' in load and '1/p.rhoLiquid' in load
core=(root.parents[1]/'src/solver/fluid/fluidSolver.cpp').read_text()
velocity=core.split('void fluidSolver_t::solveVelocity(',1)[1].split('void fluidSolver_t::setupEllipticSolver',1)[0]
assert 'core-wGradientVolumeHex3D' in velocity and 'ellipticSolver.at(0)->solve' in velocity
print('PASS: native velocity and both phase RHSs use the identical weak gradient kernel before Helmholtz')
