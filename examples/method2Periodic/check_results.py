#!/usr/bin/env python3
"""Assess recorded periodic diagnostics; case3 has no exact steady reference."""
import argparse,csv,math
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('case',choices=['1','2','3'])
p.add_argument('--velocity-tol',type=float,default=1e-6)
p.add_argument('--alpha-tol',type=float,default=1e-8)
a=p.parse_args();path=Path(__file__).resolve().parent/f'case{a.case}/periodic_diagnostics.csv'
with path.open() as f:rows=list(csv.DictReader(f))
if not rows:raise SystemExit('FAIL: no completed-step diagnostics')
if any(not math.isfinite(float(v)) for row in rows for v in row.values()):raise SystemExit('FAIL: nonfinite diagnostics')
worst={k:max(float(row[k]) for row in rows) for k in ('alpha_change_inf','ug_change_inf','um_change_inf','ug_cfl')}
print(f'case{a.case}: {len(rows)} samples; last step={rows[-1]["step"]}, t={rows[-1]["time"]}')
for k,v in worst.items():print(f'  max recorded {k} = {v:.8e}')
if a.case=='3':
 print('Finite diagnostics. Tube changes are physical transients, not errors against an exact solution; inspect alpha/UG/pressure fields and convergence under refinement.')
else:
 failed=worst['alpha_change_inf']>a.alpha_tol or max(worst['ug_change_inf'],worst['um_change_inf'])>a.velocity_tol
 print('FAIL: equilibrium drift exceeds tolerance' if failed else 'PASS: recorded uniform solution remains within tolerance')
 raise SystemExit(int(failed))
