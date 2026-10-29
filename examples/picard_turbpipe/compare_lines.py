#!/usr/bin/env python
"""Overlay native/Picard line CSVs and write line-integrated difference metrics."""
from __future__ import print_function
import argparse
import csv
import glob
import io
import os
import sys
import numpy as np


def open_csv(path, mode):
    # Python 2 csv needs binary streams; Python 3 needs newline handling.
    if sys.version_info[0] < 3:
        return open(str(path), mode + 'b')
    return io.open(str(path), mode, newline='')

FIELDS = ('ux', 'uy', 'uz', 'p', 's00')

def read_profiles(directory):
    profiles = {}
    for path in sorted(glob.glob(os.path.join(str(directory), 'lineProfiles_step*.csv'))):
        with open_csv(path, 'r') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError('Empty profile file: {0}'.format(path))
        step = int(rows[0]['step'])
        if step in profiles:
            raise ValueError('Duplicate step {0} in {1}'.format(step, directory))
        lines = {}
        for row in rows:
            if int(row['step']) != step:
                raise ValueError('Mixed steps in {0}'.format(path))
            lines.setdefault(row['line'], []).append(row)
        profiles[step] = {}
        for name, entries in lines.items():
            entries.sort(key=lambda row: int(row['point']))
            if [int(row['point']) for row in entries] != list(range(len(entries))):
                raise ValueError('Invalid point numbering in {0}: {1}'.format(path, name))
            data = {key: np.array([float(row[key]) for row in entries]) for key in ('time', 'coordinate', 'x', 'y', 'z') + FIELDS}
            if any((not np.isfinite(values).all() for values in data.values())):
                raise ValueError('Nonfinite data in {0}: {1}'.format(path, name))
            if not np.allclose(data['time'], data['time'][0], rtol=0, atol=1e-12):
                raise ValueError('Mixed times in {0}: {1}'.format(path, name))
            axes = {row['axis'] for row in entries}
            if len(axes) != 1 or not axes <= {'x', 'y', 'z'}:
                raise ValueError('Invalid axis in {0}: {1}'.format(path, name))
            data['axis'] = axes.pop()
            if len(entries) < 2 or np.any(np.diff(data['coordinate']) <= 0):
                raise ValueError('Non-increasing line coordinates in {0}: {1}'.format(path, name))
            profiles[step][name] = data
    if not profiles:
        raise ValueError('No lineProfiles_step*.csv files in {0}'.format(directory))
    return profiles

def compare(baseline, picard):
    if set(baseline) != set(picard):
        raise ValueError('Exported steps differ between runs; use matching intervals and completed steps.')
    metrics = []
    for step in sorted(baseline):
        if set(baseline[step]) != set(picard[step]):
            raise ValueError('Line names differ at step {0}'.format(step))
        for name, a in baseline[step].items():
            b = picard[step][name]
            if a['axis'] != b['axis']:
                raise ValueError('Line axes differ: step {0}, {1}'.format(step, name))
            for key in ('time', 'coordinate', 'x', 'y', 'z'):
                if a[key].shape != b[key].shape or not np.allclose(a[key], b[key], rtol=0, atol=1e-12):
                    raise ValueError('{0} mismatch: step {1}, {2}; compare identical times/locations.'.format(key, step, name))
            s = a['coordinate']
            integrate = getattr(np, 'trapezoid', None)
            if integrate is None:
                integrate = np.trapz
            for field in FIELDS:
                difference = b[field] - a[field]
                error2 = max(0.0, float(integrate(difference ** 2, s)))
                norm2 = max(0.0, float(integrate(a[field] ** 2, s)))
                metrics.append(dict(step=step, time=a['time'][0], line=name, field=field, rms=np.sqrt(error2 / (s[-1] - s[0])), max_abs=np.max(np.abs(difference)), relative_l2=np.sqrt(error2 / norm2) if norm2 > 0 else 0.0 if error2 == 0 else float('nan')))
    return metrics

def plot_profiles(baseline, picard, steps, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for step in steps:
        for name, a in baseline[step].items():
            b = picard[step][name]
            fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
            for field, axis in zip(FIELDS, axes.flat):
                axis.plot(a['coordinate'], a[field], label='Original turbPipe', lw=1.6)
                axis.plot(b['coordinate'], b[field], '--', label='Picard turbPipe', lw=1.5)
                axis.set(xlabel=a['axis'], ylabel=field)
                axis.grid(alpha=0.25)
            axis = axes.flat[-1]
            for field in FIELDS:
                axis.plot(a['coordinate'], b[field] - a[field], label=field)
            axis.set(xlabel=a['axis'], ylabel='Picard - original')
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            axes.flat[0].legend(fontsize=8)
            fig.suptitle('{0} | step {1} | t = {2:.8g}'.format(name, step, a['time'][0]))
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(os.path.join(str(output), 'comparison_step{0:06d}_{1}.png'.format(step, name)), dpi=180)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=str, help='original turbPipe run directory')
    parser.add_argument('picard', type=str, help='Picard run directory')
    parser.add_argument('--output', type=str, default='line_comparison')
    parser.add_argument('--steps', nargs='+', default=['last'], help='last (default), all, or step numbers')
    args = parser.parse_args()
    baseline, picard = (read_profiles(args.baseline), read_profiles(args.picard))
    metrics = compare(baseline, picard)
    if args.steps == ['last']:
        steps = [max(baseline)]
    elif args.steps == ['all']:
        steps = sorted(baseline)
    else:
        steps = [int(step) for step in args.steps]
        if any((step not in baseline for step in steps)):
            parser.error('Requested step is missing from the profiles')
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open_csv(os.path.join(args.output, 'line_errors.csv'), 'w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    plot_profiles(baseline, picard, steps, args.output)
    initial = [row['max_abs'] for row in metrics if row['step'] == 0]
    if initial:
        print('Initial-profile maximum absolute difference: {0:.8e}'.format(max(initial)))
        if max(initial) > 1e-12:
            print('WARNING: initial profiles differ. Check mesh, restart, MPI layout and initialization.')
    print('Compared {0} exported steps; plots and line_errors.csv: {1}'.format(len(baseline), args.output))
    print('Metrics measure differences from the native run, not error against an exact solution.')
if __name__ == '__main__':
    main()
