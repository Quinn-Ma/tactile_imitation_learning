"""Fixed second pilot: align reward with the 10cm evaluation height.

The first experiment is retained. No hyperparameters are selected on either
test set. This experiment uses new simulator seeds 20291000..20291019.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
code = root / 'outputs/world_model'
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, default=code/'rl_goal_aligned')
args = p.parse_args()
plan = []
for seed in [0, 1, 2]:
    model_dir = code/'runs'/f'visuotactile_seed{seed}'
    q = json.loads((model_dir/'calibration.json').read_text())['peak_radius_N']
    for tag, margin in [('visuotactile', 0.), ('visuotactile_margin', q)]:
        plan.append({'name': f'{tag}_seed{seed}', 'model': str(model_dir/'best.pt'),
                     'seed': seed, 'force_margin': margin})
args.out.mkdir(parents=True, exist_ok=True)
(args.out/'fixed_plan.json').write_text(json.dumps({'jobs': plan,
    'task_reward': 'clip(predicted_relative_height / 0.10, 0, 1)',
    'all_other_training_arguments': 'same as first pilot',
    'evaluation_seeds': list(range(20291000, 20291020)),
    'interpretation': 'exploratory goal-alignment correction; no test-based tuning'}, indent=2))
for job in plan:
    out = args.out/job['name']
    if (out/'rl_actor.pt').exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(code/'train_imagination_rl.py'), '--model', job['model'],
           '--data', str(root/'outputs/simulation/episodes'),
           '--splits', str(code/'runs/data_manifest.json'), '--out', str(out),
           '--seed', str(job['seed']), '--start-step', '44', '--control-end', '135',
           '--force-cap', '8', '--force-margin', str(job['force_margin']),
           '--task-reward', 'height', '--target-height', '.10', '--device', 'cuda']
    print(json.dumps({'starting': job['name']}), flush=True)
    with (out/'console.log').open('w', encoding='utf-8') as log:
        subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'completed': job['name']}), flush=True)
(args.out/'batch_complete.json').write_text(json.dumps({'jobs': plan}, indent=2))
