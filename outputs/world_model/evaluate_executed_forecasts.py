"""Retrospective forecasts conditioned on actions actually executed in MuJoCo.

This is an evaluation, not online learning. Future executed actions are supplied
as conditioning to the world model. Calibration radii are read unchanged from
the original held-out calibration split. No threshold or weight is refitted.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from data import read_episodes, SequenceDataset
from model import load_model
from train import predictions, metrics


def main():
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    p.add_argument('--evaluations', type=Path, default=root/'outputs/simulation/evaluation_main')
    p.add_argument('--out', type=Path, default=root/'outputs/world_model/executed_forecasts')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = []
    for path in sorted(args.evaluations.glob('*/summary.json')):
        summary = json.loads(path.read_text())
        if summary['controller'] != 'policy':
            continue
        checkpoint = Path(summary['model'])
        model, _ = load_model(checkpoint, 'cuda')
        q = json.loads((checkpoint.parent/'calibration.json').read_text())
        episodes = read_episodes(path.parent)
        for ep in episodes:
            seed = int(ep['seed'])
            ep['split'] = 'id' if seed % 2 == 0 else 'ood'
        for domain in ['id', 'ood']:
            ds = SequenceDataset(episodes, domain, horizon=5, history=3, stride=5)
            # Only starts and targets wholly inside the learned controller window.
            ds.indices = [(j, t) for j, t in ds.indices if 44 <= t and t+5 <= 135]
            record = predictions(model, DataLoader(ds, batch_size=64, shuffle=False), 'cuda')
            result = metrics(record)
            ids = np.unique(record['episode_index'])
            peak_error = record['true_peak'] - record['pred_peak']
            height_error = np.abs(record['true_height'] - record['pred_height'])
            covered = [bool(np.all(peak_error[record['episode_index']==i] <= q['peak_radius_N'])
                       and np.all(height_error[record['episode_index']==i] <= q['height_radius_m'])) for i in ids]
            row = {'controller': path.parent.name, 'domain': domain,
                   'episodes': len(ids), 'windows': result['windows'],
                   'endpoint_force_mae_N': result['force_mae_N'],
                   'interval_peak_mae_N': result['substep_peak_mae_N'],
                   'height_mae_m': result['height_mae_m'],
                   'fixed_calibration_joint_coverage': float(np.mean(covered)),
                   'calibration_peak_radius_N': q['peak_radius_N']}
            rows.append(row)
            np.savez_compressed(args.out/f'{path.parent.name}_{domain}.npz', **record)
        print(path.parent.name, flush=True)
    with (args.out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.out/'provenance.json').write_text(json.dumps({
        'controller_runs': len(rows)//2, 'horizon': 5, 'stride': 5,
        'window': 'starts >=44, last transition <=135',
        'scope': 'retrospective action-conditioned forecasts of held-out executed trajectories',
        'calibration': 'unchanged original recording-split quantiles; no closed-loop recalibration',
        'not_claimed': 'online replanning accuracy without supplying future actions; closed-loop safety guarantee'
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
