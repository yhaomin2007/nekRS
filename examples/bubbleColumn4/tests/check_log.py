#!/usr/bin/env python3
"""Validate a native column run; continuity acceptance must be chosen explicitly."""
import argparse,re,math
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('log',type=Path)
p.add_argument('--steps',type=int,default=20)
p.add_argument('--max-weak-continuity',type=float,required=True)
p.add_argument('--max-liquid-l2',type=float,help='For the pure-liquid hydrostatic test')
a=p.parse_args()
lines=a.log.read_text(errors='replace').splitlines()
expected='pressure';step=1;continuity={}
for line in lines:
    if line.startswith('method4Pressure '):
        assert expected=='pressure',f'Missing phase momentum at step {step}'
        assert int(re.search(r'step=(\d+)',line)[1])==step
        expected='gas'
    elif line.startswith('method4Momentum '):
        assert int(re.search(r'step=(\d+)',line)[1])==step
        phase=re.search(r'phase=(\w+)',line)[1]
        assert phase==expected, f'Wrong solve order at step {step}: {line}'
        residual=float(re.search(r'scaledResidual=(\S+)',line)[1])
        target=float(re.search(r'target=(\S+)',line)[1])
        assert math.isfinite(residual) and residual<=target
        expected='liquid' if phase=='gas' else 'pressure'
        if phase=='liquid':step+=1
    elif line.startswith('method4Continuity '):
        k=int(re.search(r'step=(\d+)',line)[1])
        value=float(re.search(r'weakFreeL2=(\S+)',line)[1])
        assert math.isfinite(value) and value<=a.max_weak_continuity, line
        liquid=float(re.search(r'velocityL2=\([^,]+,([^\)]+)\)',line)[1])
        if a.max_liquid_l2 is not None:assert math.isfinite(liquid) and liquid<=a.max_liquid_l2,line
        continuity[k]=value
assert expected=='pressure' and step==a.steps+1, f'Only {step-1} full steps found'
assert a.steps in continuity, 'Use monitorInterval=1 or end at a monitored step'
assert not any(re.search(r'\b(nan|inf)\b',s,re.I) for s in lines if s.startswith('method4')), 'Nonfinite diagnostic'
print(f'PASS: {a.steps} ordered native pressure/gas/liquid solves; last weak continuity={continuity[a.steps]:.8e}')
