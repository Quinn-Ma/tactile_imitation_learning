"""Execute the prespecified pilot matrix; each process writes an independent log."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
code = root / 'outputs/world_model'
p = argparse.ArgumentParser()
p.add_argument('--runs', type=Path, default=code / 'runs')
p.add_argument('--out', type=Path, default=root / 'work/reproduce_rl_matrix')
args = p.parse_args()
jobs = []
for seed in [0, 1, 2]:
    for variant in ['vision', 'visuotactile']:
        name = f'{variant}_seed{seed}'
        model_dir = args.runs / name
        jobs.append((name, model_dir, seed, 0.0))
    model_dir = args.runs / f'visuotactile_seed{seed}'
    calibration = json.loads((model_dir / 'calibration.json').read_text())
    margin = calibration['peak_radius_N']
    if not calibration['available'] or margin is None:
        raise RuntimeError('A finite, measured calibration margin is required')
    jobs.append((f'visuotactile_margin_seed{seed}', model_dir, seed, margin))

for name, model_dir, seed, margin in jobs:
    out = args.out / name
    if (out / 'rl_actor.pt').exists() or (out / 'bc_actor.pt').exists():
        raise FileExistsError(f'{out}: select a new --out directory to preserve completed experiments')
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(code / 'train_imagination_rl.py'),
               '--model', str(model_dir / 'best.pt'),
               '--data', str(root / 'outputs/simulation/episodes'),
               '--splits', str(args.runs / 'data_manifest.json'),
               '--out', str(out), '--seed', str(seed), '--start-step', '44',
               '--control-end', '135', '--force-cap', '8',
               '--force-margin', str(margin), '--device', 'cuda']
    print(json.dumps({'starting': name, 'margin_N': margin}), flush=True)
    with (out / 'console.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = json.loads((out / 'run_metadata.json').read_text())
    print(json.dumps({'completed': name, 'seconds': metadata['elapsed_seconds']}), flush=True)

(args.out / 'batch_complete.json').write_text(json.dumps({'jobs': [x[0] for x in jobs]}, indent=2))

