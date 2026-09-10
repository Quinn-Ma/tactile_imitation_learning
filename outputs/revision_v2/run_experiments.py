"""Run the fixed revision-v2 pipeline with user-supplied Python environments.

Use --dry-run to inspect commands without reading outcomes or running models.
Runtime files are separate from archived recorded/ evidence. No packages are
installed, files deleted, or outcome-dependent parameter choices made here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/revision_v2'
STAGES = ('protocol', 'collect', 'world-model', 'imagined-actors', 'reactive',
          'evaluate', 'counterfactual', 'forecast', 'analyze', 'sensor-stress', 'sensor-analyze')
NAMES = [(variant, seed, f'{variant}_seed{seed}') for variant in ('vision', 'visuotactile') for seed in (0, 1, 2)]


def method_names():
    names = [f'{kind}_{name}' for _, _, name in NAMES for kind in ('WM', 'WM_BC', 'BC', 'IQL')]
    names += ['script', 'force_feedback']
    names += [f'{kind}_seed{seed}' for kind in ('model_guard', 'model_no_guard') for seed in (0, 1, 2)]
    return names+['reactive_guard', 'reactive_no_guard']


def command_plan(sim, gpu, workers):
    rev = 'outputs/revision_v2/'
    data = rev+'dataset/episodes'
    jobs = {stage: [] for stage in STAGES}
    jobs['protocol'] = [[sim, rev+'build_protocol.py'], [sim, rev+'build_methods.py']]
    jobs['collect'] = [[sim, rev+'generate_expanded_data.py', '--shard', str(i), '--shards', str(workers)] for i in range(workers)]
    jobs['collect'].append([sim, rev+'audit_dataset.py'])
    jobs['world-model'] = [[gpu, 'outputs/world_model/train.py', '--data', data, '--output', rev+'world_models',
        '--variants', 'vision', 'visuotactile', '--seeds', '0', '1', '2', '--epochs', '25', '--batch-size', '64',
        '--horizon', '5', '--history', '3', '--lr', '0.0003', '--train-stride', '3', '--eval-stride', '5',
        '--eval-windows', '0', '--alpha', '0.1', '--device', 'cuda']]
    for variant, seed, name in NAMES:
        jobs['imagined-actors'].append([gpu, 'outputs/world_model/train_imagination_rl.py',
            '--model', rev+f'world_models/{name}/best.pt', '--data', data,
            '--splits', rev+'world_models/data_manifest.json', '--out', rev+f'rl/{name}', '--seed', str(seed),
            '--start-step', '44', '--control-end', '135', '--bc-steps', '1000', '--rl-steps', '1500',
            '--batch-size', '256', '--horizon', '5', '--force-cap', '8', '--force-margin', '0',
            '--force-weight', '1', '--task-reward', 'height', '--target-height', '0.10',
            '--behavior-weight', '2', '--device', 'cuda'])
        jobs['reactive'].append([gpu, rev+'train_reactive.py', '--data', data,
            '--splits', rev+'dataset/data_manifest.json', '--out', rev+f'reactive_runs/{name}',
            '--variant', variant, '--seed', str(seed), '--expected-train-episodes', '240', '--expected-val-episodes', '48',
            '--bc-steps', '2000', '--iql-steps', '10000', '--batch-size', '256', '--eval-every', '250',
            '--learning-rate', '0.0003', '--discount', '0.95', '--expectile', '0.7', '--beta', '3',
            '--weight-cap', '100', '--ema', '0.005', '--gradient-clip', '10', '--train-stride', '1',
            '--start-step', '44', '--control-end', '135', '--target-height', '0.10', '--force-cap', '8',
            '--force-weight', '1', '--action-weight', '0.005', '--device', 'cuda'])
    jobs['evaluate'] = [[sim, rev+'evaluate.py', '--method', name, '--gpu-python', gpu, '--resume'] for name in method_names()]
    jobs['counterfactual'] = [[sim, rev+'counterfactual_audit.py', '--gpu-python', gpu, '--device', 'cpu', '--resume']]
    jobs['forecast'] = [[gpu, rev+'forecast_eval.py', '--device', 'cuda', '--batch-size', '64', '--run-test-evaluation']]
    jobs['analyze'] = [[sim, rev+'analyze_control.py', '--skip-legacy']]
    stress_methods = [name for name in method_names() if name.startswith(('WM_visuotactile_', 'IQL_visuotactile_', 'model_guard_')) or name in ('force_feedback', 'reactive_guard')]
    jobs['sensor-stress'] = [[sim, rev+'sensor_stress.py', '--run', '--method', name, '--gpu-python', gpu, '--resume'] for name in stress_methods]
    jobs['sensor-analyze'] = [[sim, rev+'sensor_stress.py', '--analyze']]
    return jobs


def resolve_python(value):
    resolved = shutil.which(value)
    if resolved is None:
        candidate = Path(value).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError('Python executable not found: '+value)
        resolved = str(candidate.resolve())
    return str(Path(resolved).resolve())


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require_files(names):
    missing = [str(name) for name in names if not (ROOT/name).is_file()]
    if missing:
        raise FileNotFoundError('Required earlier-stage files missing: '+', '.join(missing))


def stage_preconditions(stage):
    rev = Path('outputs/revision_v2')
    if stage in ('world-model', 'reactive'):
        require_files([rev/'dataset/audit.json', rev/'dataset/data_manifest.json'])
        audit = read_json(BASE/'dataset/audit.json')
        if audit['episodes'] != 336 or audit['counts'] != {'train': 240, 'val': 48, 'calibration': 48}:
            raise ValueError('Require the full audited collection before training')
    if stage in ('imagined-actors', 'counterfactual', 'forecast'):
        require_files([rev/f'world_models/{name}/best.pt' for _, _, name in NAMES])
    if stage in ('evaluate', 'sensor-stress'):
        require_files([rev/f'rl/{name}/{kind}_actor.pt' for _, _, name in NAMES for kind in ('bc', 'rl')])
        require_files([rev/f'reactive_runs/{name}/{kind}_policy.pt' for _, _, name in NAMES for kind in ('bc', 'iql')])
        require_files([rev/f'world_models/{name}/best.pt' for _, _, name in NAMES])
    if stage == 'forecast':
        require_files([rev/f'evaluations/script/episode_{row["seed"]}.npz' for row in read_json(BASE/'protocol.json')['control_test']])
    if stage == 'analyze':
        require_files([rev/f'evaluations/{name}/episodes.jsonl' for name in method_names()])
    if stage == 'sensor-stress':
        require_files([rev/'analysis/control_analysis.json'])
    if stage == 'sensor-analyze':
        require_files([rev/f'sensor_stress_evaluations/{row["name"]}/summary.json' for row in read_json(BASE/'sensor_stress_protocol.json')['methods']])


def training_state(stage, command):
    """Skip complete training only; never overwrite a partial fitted run."""
    if stage == 'world-model':
        directory = BASE/'world_models'
        if not directory.exists() or not any(path.name != 'runner_training_inputs.json' for path in directory.iterdir()):
            return 'new'
        complete = all((directory/name/'best.pt').is_file() and (directory/name/'summary.json').is_file() for _, _, name in NAMES)
        if complete and (directory/'all_runs.json').is_file():
            return 'complete'
        raise FileExistsError('Partial world-model training exists. Preserve it and use a fresh checkout for a full rerun.')
    if stage not in ('imagined-actors', 'reactive'):
        return 'new'
    directory = ROOT/command[command.index('--out')+1]
    if not directory.exists() or not any(path.name != 'runner_training_inputs.json' for path in directory.iterdir()):
        return 'new'
    metadata_path = directory/'run_metadata.json'
    artifacts = ('bc_actor.pt', 'rl_actor.pt') if stage == 'imagined-actors' else ('bc_policy.pt', 'iql_policy.pt')
    if metadata_path.is_file() and all((directory/name).is_file() for name in artifacts):
        metadata = read_json(metadata_path)
        if stage == 'reactive' and (metadata.get('completed_bc_updates'), metadata.get('completed_iql_updates')) != (2000, 10000):
            raise ValueError('Existing reactive run used another update budget')
        if stage == 'imagined-actors' and (metadata['arguments']['bc_steps'], metadata['arguments']['rl_steps'], metadata['arguments']['task_reward']) != (1000, 1500, 'height'):
            raise ValueError('Existing imagined run used another budget/reward')
        return 'complete'
    raise FileExistsError('Partial training exists: '+str(directory)+'. Preserve it and use a fresh checkout; this runner never deletes fitted work.')


def training_receipt(stage, command):
    if stage not in ('world-model', 'imagined-actors', 'reactive'):
        return None, None
    directory = BASE/'world_models' if stage == 'world-model' else ROOT/command[command.index('--out')+1]
    files = list(BASE.glob('*.py'))+list((ROOT/'outputs/world_model').glob('*.py'))
    files += [BASE/'protocol.json', BASE/'dataset/data_manifest.json']
    if stage == 'imagined-actors':
        files += [ROOT/command[command.index('--model')+1], BASE/'world_models/data_manifest.json']
    inputs = {'command_arguments': command[1:],
              'input_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}}
    receipt = directory/'runner_training_inputs.json'
    if receipt.exists():
        if read_json(receipt) != inputs:
            raise ValueError('Training inputs changed; refusing mixed rerun: '+str(directory))
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError('Existing fitted work lacks this runner input receipt; use a fresh checkout: '+str(directory))
    return receipt, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-python', required=True, help='Python 3.12 with simulation requirements installed')
    parser.add_argument('--gpu-python', required=True, help='Python with compatible CUDA PyTorch installed')
    parser.add_argument('--workers', type=int, default=2, help='Simulation parallelism; reactive GPU jobs are capped at two')
    parser.add_argument('--from-stage', choices=STAGES, default=STAGES[0])
    parser.add_argument('--through-stage', choices=STAGES, default=STAGES[-1])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 34 or STAGES.index(args.from_stage) > STAGES.index(args.through_stage):
        parser.error('Require 1 <= workers <= 34 and ordered stage bounds')
    sim = args.sim_python if args.dry_run else resolve_python(args.sim_python)
    gpu = args.gpu_python if args.dry_run else resolve_python(args.gpu_python)
    jobs = command_plan(sim, gpu, args.workers)
    stages = STAGES[STAGES.index(args.from_stage):STAGES.index(args.through_stage)+1]
    if args.dry_run:
        print(json.dumps({'fixed_collection': 336, 'fixed_test_environments': 120, 'fixed_policies': 34,
            'commands_by_stage': {stage: jobs[stage] for stage in stages}}, indent=2))
        return
    logs = ROOT/'work/revision_v2_runner_logs'
    logs.mkdir(parents=True, exist_ok=True)
    def run(stage, numbered):
        number, command = numbered
        receipt, inputs = training_receipt(stage, command)
        if training_state(stage, command) == 'complete':
            print(json.dumps({'stage': stage, 'job': number, 'status': 'existing complete training retained'}), flush=True)
            return
        if receipt is not None and not receipt.exists():
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps(inputs, indent=2), encoding='utf-8')
        path = logs/f'{stage}_{number:02d}.log'
        print(json.dumps({'stage': stage, 'job': number, 'status': 'started'}), flush=True)
        with path.open('w', encoding='utf-8') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        print(json.dumps({'stage': stage, 'job': number, 'status': 'complete'}), flush=True)
    for stage in stages:
        stage_preconditions(stage)
        commands = jobs[stage]
        if stage == 'collect':
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda item: run(stage, item), enumerate(commands[:-1])))
            run(stage, (len(commands)-1, commands[-1]))
        elif stage in ('evaluate', 'sensor-stress', 'reactive'):
            width = min(args.workers, 2) if stage == 'reactive' else args.workers
            with ThreadPoolExecutor(max_workers=width) as pool:
                list(pool.map(lambda item: run(stage, item), enumerate(commands)))
        else:
            for numbered in enumerate(commands):
                run(stage, numbered)
    print(json.dumps({'completed_stages': list(stages), 'note': 'Fresh runtime results; archived recorded/ evidence was untouched.'}, indent=2))


if __name__ == '__main__':
    main()
