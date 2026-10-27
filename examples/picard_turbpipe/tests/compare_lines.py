"""Known profile differences, mismatched sampling, and plotting CLI regression."""
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

case = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('profiles', case/'compare_lines.py')
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for mode in ('baseline', 'picard'):
        run = root/mode; run.mkdir()
        for step in (0, 100, 200):
            with (run/f'lineProfiles_step{step:06d}.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow('step,time,line,point,axis,coordinate,x,y,z,ux,uy,uz,p,s00'.split(','))
                for line in ('x_z50', 'axial'):
                    for i in range(11):
                        q=i/10
                        shift = .25 if mode=='picard' and step else 0
                        writer.writerow([step, step*.001, line, i, 'x' if line=='x_z50' else 'z',
                                         q, q if line=='x_z50' else 0, 0, q if line=='axial' else 10,
                                         *([1+shift]*5)])
    baseline=profiles.read_profiles(root/'baseline'); picard=profiles.read_profiles(root/'picard')
    metrics=profiles.compare(baseline,picard)
    assert len(metrics)==30
    for row in metrics:
        expected=0 if row['step']==0 else .25
        assert abs(row['rms']-expected)<1e-14 and abs(row['relative_l2']-expected)<1e-14
        assert row['max_abs']==expected
    original=picard[100]['axial']['time'].copy()
    picard[100]['axial']['time']+=.001
    try:profiles.compare(baseline,picard)
    except ValueError:pass
    else:raise AssertionError('Mismatched physical times were accepted')
    picard[100]['axial']['time']=original
    picard[100]['axial']['x'][0]+=.01
    try:profiles.compare(baseline,picard)
    except ValueError:pass
    else:raise AssertionError('Mismatched physical locations were accepted')
    output=root/'plots'
    subprocess.run([sys.executable,str(case/'compare_lines.py'),str(root/'baseline'),str(root/'picard'),
                    '--steps','100','--output',str(output)],check=True)
    assert len(list(output.glob('*.png')))==2
    assert len(list(csv.DictReader((output/'line_errors.csv').open())))==30
print('PASS: exact RMS/relative L2/max differences, time/location mismatch rejection, CSV summary and overlay PNGs.')
