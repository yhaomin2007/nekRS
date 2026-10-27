#!/usr/bin/env python3
"""Overlay native/Picard line CSVs and write line-integrated difference metrics."""
import argparse
import csv
from pathlib import Path

import numpy as np

FIELDS = ('ux', 'uy', 'uz', 'p', 's00')


def read_profiles(directory):
    profiles = {}
    for path in sorted(Path(directory).glob('lineProfiles_step*.csv')):
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f'Empty profile file: {path}')
        step = int(rows[0]['step'])
        if step in profiles:
            raise ValueError(f'Duplicate step {step} in {directory}')
        lines = {}
        for row in rows:
            if int(row['step']) != step:
                raise ValueError(f'Mixed steps in {path}')
            lines.setdefault(row['line'], []).append(row)
        profiles[step] = {}
        for name, entries in lines.items():
            entries.sort(key=lambda row: int(row['point']))
            if [int(row['point']) for row in entries] != list(range(len(entries))):
                raise ValueError(f'Invalid point numbering in {path}: {name}')
            data = {key: np.array([float(row[key]) for row in entries])
                    for key in ('time', 'coordinate', 'x', 'y', 'z') + FIELDS}
            if any(not np.isfinite(values).all() for values in data.values()):
                raise ValueError(f'Nonfinite data in {path}: {name}')
            if not np.allclose(data['time'], data['time'][0], rtol=0, atol=1e-12):
                raise ValueError(f'Mixed times in {path}: {name}')
            axes = {row['axis'] for row in entries}
            if len(axes) != 1 or not axes <= {'x', 'y', 'z'}:
                raise ValueError(f'Invalid axis in {path}: {name}')
            data['axis'] = axes.pop()
            if len(entries) < 2 or np.any(np.diff(data['coordinate']) <= 0):
                raise ValueError(f'Non-increasing line coordinates in {path}: {name}')
            profiles[step][name] = data
    if not profiles:
        raise ValueError(f'No lineProfiles_step*.csv files in {directory}')
    return profiles


def compare(baseline, picard):
    if baseline.keys() != picard.keys():
        raise ValueError('Exported steps differ between runs; use matching intervals and completed steps.')
    metrics = []
    for step in sorted(baseline):
        if baseline[step].keys() != picard[step].keys():
            raise ValueError(f'Line names differ at step {step}')
        for name, a in baseline[step].items():
            b = picard[step][name]
            if a['axis'] != b['axis']:
                raise ValueError(f'Line axes differ: step {step}, {name}')
            for key in ('time', 'coordinate', 'x', 'y', 'z'):
                if a[key].shape != b[key].shape or not np.allclose(a[key], b[key], rtol=0, atol=1e-12):
                    raise ValueError(f'{key} mismatch: step {step}, {name}; compare identical times/locations.')
            s = a['coordinate']
            integrate = getattr(np, 'trapezoid', None)
            if integrate is None:
                integrate = np.trapz
            for field in FIELDS:
                difference = b[field] - a[field]
                error2 = max(0., float(integrate(difference**2, s)))
                norm2 = max(0., float(integrate(a[field]**2, s)))
                metrics.append(dict(step=step, time=a['time'][0], line=name, field=field,
                                    rms=np.sqrt(error2 / (s[-1]-s[0])),
                                    max_abs=np.max(np.abs(difference)),
                                    relative_l2=np.sqrt(error2/norm2) if norm2 > 0 else
                                                (0. if error2 == 0 else float('nan'))))
    return metrics


def plot_profiles(baseline, picard, steps, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for step in steps:
        for name, a in baseline[step].items():
            b = picard[step][name]
            fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
            for field, axis in zip(FIELDS, axes.flat):
                axis.plot(a['coordinate'], a[field], label='Original turbPipe', lw=1.6)
                axis.plot(b['coordinate'], b[field], '--', label='Picard turbPipe', lw=1.5)
                axis.set(xlabel=a['axis'], ylabel=field)
                axis.grid(alpha=.25)
            axis = axes.flat[-1]
            for field in FIELDS:
                axis.plot(a['coordinate'], b[field]-a[field], label=field)
            axis.set(xlabel=a['axis'], ylabel='Picard - original')
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f'{name} | step {step} | t = {a["time"][0]:.8g}')
            fig.savefig(output / f'comparison_step{step:06d}_{name}.png', dpi=180)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path, help='original turbPipe run directory')
    parser.add_argument('picard', type=Path, help='Picard run directory')
    parser.add_argument('--output', type=Path, default=Path('line_comparison'))
    parser.add_argument('--steps', nargs='+', default=['last'], help='last (default), all, or step numbers')
    args = parser.parse_args()
    baseline, picard = read_profiles(args.baseline), read_profiles(args.picard)
    metrics = compare(baseline, picard)
    if args.steps == ['last']:
        steps = [max(baseline)]
    elif args.steps == ['all']:
        steps = sorted(baseline)
    else:
        steps = [int(step) for step in args.steps]
        if any(step not in baseline for step in steps):
            parser.error('Requested step is missing from the profiles')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'line_errors.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    plot_profiles(baseline, picard, steps, args.output)
    initial = [row['max_abs'] for row in metrics if row['step'] == 0]
    if initial:
        print(f'Initial-profile maximum absolute difference: {max(initial):.8e}')
        if max(initial) > 1e-12:
            print('WARNING: initial profiles differ. Check mesh, restart, MPI layout and initialization.')
    print(f'Compared {len(baseline)} exported steps; plots and line_errors.csv: {args.output}')
    print('Metrics measure differences from the native run, not error against an exact solution.')


if __name__ == '__main__':
    main()
