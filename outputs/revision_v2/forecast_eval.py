"""Fixed held-out forecasts on common SCRIPT trajectories, never policy traces.

Main execution requires --run-test-evaluation. Without that flag this module
opens no test trajectory. Only the six protocol-specified new checkpoints and
their already-computed 48-episode calibration files are accepted. No predictor,
normalization, calibration radius, action, or threshold is fitted here.

The original SequenceDataset, predictions, and metrics functions define the
H=5/history=3/stride=5 semantics, including per-interval physics-substep peaks.
Uncertainty uses whole environment resampling, shared across fixed model seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
WM_SOURCE = ROOT/'outputs/world_model'
sys.path.insert(0, str(WM_SOURCE))
from data import SequenceDataset
from model import load_model
from train import metrics as original_metrics, predictions

DOMAINS = ('test_id', 'test_geometry_ood', 'test_physics_ood', 'test_combined_ood')
DOMAIN_MAP = {x.removeprefix('test_'): x for x in DOMAINS}
VARIANTS = ('vision', 'visuotactile')
TRAINING_SEEDS = (0, 1, 2)
HORIZON, HISTORY, STRIDE, EPISODE_ACTIONS = 5, 3, 5, 150
ERROR_FIELDS = {'endpoint_force_mae_N': 'tactile', 'interval_peak_mae_N': 'peak', 'height_mae_m': 'height'}
COVERAGE_FIELDS = ('peak_episode_coverage', 'height_episode_coverage', 'joint_episode_coverage')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_native(value):
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_native(value), indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    rows = [json_native(row) for row in rows]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def validate_protocol(protocol):
    """Validate design metadata only; no trajectory or outcome access."""
    if protocol.get('prospective') is not True or protocol.get('training_seeds') != list(TRAINING_SEEDS):
        raise ValueError('Require prospective fixed three-seed protocol')
    if protocol.get('horizon') != EPISODE_ACTIONS or protocol.get('takeover') != 44 or protocol.get('control_end') != 135:
        raise ValueError('Unexpected intervention/horizon protocol')
    if protocol.get('main_control_dimensions') != [2, 6] or protocol.get('zero_intervention_dimensions') != [0, 1, 3, 4, 5]:
        raise ValueError('Unexpected action subspace')
    collection, tests = protocol['collection'], protocol['control_test']
    if Counter(row['split'] for row in collection) != {'train': 240, 'val': 48, 'calibration': 48}:
        raise ValueError('Require explicit expanded 240/48/48 collection')
    if Counter(row['domain'] for row in tests) != {domain: 30 for domain in DOMAIN_MAP}:
        raise ValueError('Require 30 held-out environments in each of four domains')
    for split in ('train', 'val', 'calibration'):
        counts = Counter(row['controller'] for row in collection if row['split'] == split)
        if counts != {'script': 120 if split == 'train' else 24,
                       'force_feedback': 120 if split == 'train' else 24}:
            raise ValueError('Unexpected collection-controller mixture')
    all_rows = collection+tests+protocol.get('counterfactual', [])
    if len({int(r['seed']) for r in all_rows}) != len(all_rows):
        raise ValueError('Collection, main test, and diagnostic seeds must be disjoint')
    if len({r['episode_id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Episode identities must be disjoint')
    if protocol.get('bootstrap_resamples') != 2000:
        raise ValueError('Expected predeclared 2000 environment resamples')
    return sorted(tests, key=lambda row: (DOMAINS.index(DOMAIN_MAP[row['domain']]), int(row['seed'])))


def validate_calibration(calibration):
    """Read-only sanity check of existing quantiles, never recompute them."""
    if calibration.get('available') is not True or calibration.get('calibration_episodes') != 48:
        raise ValueError('Require existing calibration from 48 independent episodes')
    if not math.isclose(float(calibration.get('alpha_joint', -1)), .10, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Unexpected predeclared joint alpha')
    rank = math.ceil(49*.95)
    if calibration.get('order_statistic_rank') != rank or calibration.get('infinite_radius', False):
        raise ValueError('Unexpected finite-sample calibration rank/radius status')
    for name in ('peak_radius_N', 'height_radius_m'):
        radius = calibration.get(name)
        if radius is None or not math.isfinite(float(radius)) or radius < 0:
            raise ValueError(f'Invalid precomputed {name}')
    return calibration


def validate_checkpoint(payload, model, variant, seed, protocol):
    if model.config.variant != variant or payload.get('seed') != seed:
        raise ValueError('Checkpoint identity does not match fixed model matrix')
    if model.config.history != HISTORY or model.config.action_dim != 7 or payload.get('horizon') != HORIZON:
        raise ValueError('Checkpoint history/action/horizon mismatch')
    expected = {(r['episode_id'], r['split']) for r in protocol['collection']}
    manifest = payload.get('data_manifest')
    if not isinstance(manifest, list) or len(manifest) != len(expected):
        raise ValueError('Checkpoint lacks the complete expanded collection manifest')
    actual = [(r.get('episode_id'), r.get('split')) for r in manifest]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Checkpoint manifest differs from fixed train/val/calibration collection')
    arguments = payload.get('arguments', {})
    for key, value in [('horizon', 5), ('history', 3), ('train_stride', 3), ('eval_stride', 5), ('eval_windows', 0)]:
        if arguments.get(key) != value:
            raise ValueError(f'Checkpoint {key} does not match fixed forecast/training protocol')


def _scalar(value):
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else value


def _matches(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _matches(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(_matches(a, b) for a, b in zip(expected, actual))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-9)
    return expected == actual


def parameters_match(expected, actual):
    """Check prescribed quantities, allowing the audited compiled-size bound.

    The legacy cube's pre-compilation aperture screen uses .022 m half-size
    bounds. Reset recomputes only derived widths from the actual seeded cube.
    It is incorrect to reject this deterministic refinement as changed physics.
    """
    if not _matches({k:v for k,v in expected.items() if k!='feasibility'}, actual):
        return False
    planned,realized=expected.get('feasibility'),actual.get('feasibility')
    if planned is None:return True
    if not isinstance(realized,dict):return False
    for key,value in planned.items():
        if key in ('horizontal_width_bound_m','required_aperture_m'):
            if key not in realized or not 0<float(realized[key])<=float(value)+1e-9:return False
        elif key not in realized or not _matches(value,realized[key]):return False
    return True


def load_script_episode(path, row):
    """Load a single named raw trace; do not call read_episodes or glob files."""
    with np.load(path, allow_pickle=False) as source:
        required = ('rgb', 'tactile', 'proprio', 'action', 'height', 'support', 'contact',
                    'substep_normal_peak', 'reward', 'seed', 'params_json')
        if any(name not in source for name in required):
            raise ValueError(f'{path}: missing required raw-script trajectory fields')
        ep = {key: source[key].copy() for key in required}
        for key in ('controller', 'collection_controller'):
            if key in source and _scalar(source[key]) != 'script':
                raise ValueError(f'{path}: cannot use another controller distribution')
        if 'dt' in source and not math.isclose(float(source['dt']), .05, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f'{path}: wrong observation frequency')
    if int(ep['seed']) != int(row['seed']):
        raise ValueError(f'{path}: seed differs from protocol')
    params = json.loads(str(_scalar(ep['params_json'])))
    if not parameters_match(row['params'], params):
        raise ValueError(f'{path}: physical parameters differ from frozen protocol')
    if ep['rgb'].shape != (151, 64, 64, 3) or ep['rgb'].dtype != np.uint8:
        raise ValueError(f'{path}: require complete 151-frame uint8 RGB trace')
    for name, shape in [('action', (150, 7)), ('tactile', (151, 6)), ('proprio', (151, 9)),
                        ('height', (151, 1)), ('support', (151, 1)), ('contact', (151, 2)),
                        ('substep_normal_peak', (151, 2))]:
        ep[name] = np.asarray(ep[name], dtype=np.float32)
        if ep[name].shape != shape or not np.isfinite(ep[name]).all():
            raise ValueError(f'{path}: invalid {name} shape/data')
    if np.abs(ep['action'][44:135][:, [0, 1, 3, 4, 5]]).max() > 1e-6:
        raise ValueError(f'{path}: script trace violates common two-action intervention')
    reward = np.asarray(ep['reward'], dtype=np.float32).reshape(-1, 1)
    if len(reward) == 150:
        reward = np.concatenate((np.zeros((1, 1), np.float32), reward))
    if reward.shape != (151, 1) or not np.isfinite(reward).all():
        raise ValueError(f'{path}: invalid reward alignment')
    ep.update(reward=reward, has_reward=True, peak=ep['substep_normal_peak'],
              episode_id=row['episode_id'], environment_seed=int(row['seed']),
              split=DOMAIN_MAP[row['domain']], path=str(Path(path).resolve()))
    return ep


def forecast_records(model, episodes, split, device, batch_size, mode='world_model'):
    if mode not in ('world_model', 'persistence', 'shuffled_touch'):
        raise ValueError(mode)
    if mode != 'world_model' and model.config.variant != 'visuotactile':
        raise ValueError('Observed-touch persistence/shuffling are VT diagnostics')
    dataset = SequenceDataset(episodes, split, horizon=HORIZON, history=HISTORY, stride=STRIDE,
                              shuffle_tactile=mode == 'shuffled_touch')
    expected = [(i, start) for i in range(len(dataset.episodes)) for start in range(0, 146, 5)]
    if dataset.indices != expected or not dataset.episodes:
        raise ValueError('Forecasts must use all fixed 30 windows per complete episode')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    record = predictions(model, loader, device, persistence=mode == 'persistence')
    return record, dataset.episodes


def sufficient_statistics(record, episodes, calibration=None):
    """Per-environment sums/counts preserve original pooled-coordinate MAE.

Each dict maps metric to (numerator, denominator). Using these pairs in a
whole-environment bootstrap also keeps contact-conditioned errors correctly
weighted when contact counts differ among episodes.
"""
    indices = np.asarray(record['episode_index'])
    if set(np.unique(indices)) != set(range(len(episodes))):
        raise ValueError('Missing or unidentified environment in forecast records')
    result = {}
    for i, episode in enumerate(episodes):
        mask = indices == i
        if int(mask.sum()) != 30 or not np.array_equal(record['start'][mask], np.arange(0, 146, 5)):
            raise ValueError('Missing/duplicate/reordered forecast windows')
        row = {}
        for name, target in ERROR_FIELDS.items():
            error = np.abs(record['pred_'+target][mask]-record['true_'+target][mask]).astype(np.float64)
            if error.shape[1] != HORIZON or not np.isfinite(error).all():
                raise ValueError('Invalid forecast error data')
            row[name] = float(error.sum()), int(error.size)
            for h in range(HORIZON):
                row[f'{name}_h{h+1}'] = float(error[:, h].sum()), int(error[:, h].size)
        active = record['true_contact'][mask].max(-1) > .5
        contact_error = np.abs(record['pred_tactile'][mask]-record['true_tactile'][mask])[active]
        row['contact_phase_endpoint_force_mae_N'] = float(contact_error.astype(np.float64).sum()), int(contact_error.size)
        row['contact_forecast_fraction'] = int(active.sum()), int(active.size)
        if calibration is not None:
            good_peak = bool(np.all(record['true_peak'][mask]-record['pred_peak'][mask] <= calibration['peak_radius_N']))
            good_height = bool(np.all(np.abs(record['pred_height'][mask]-record['true_height'][mask]) <= calibration['height_radius_m']))
            for name, covered in zip(COVERAGE_FIELDS, (good_peak, good_height, good_peak and good_height)):
                row[name] = int(covered), 1
        result[episode['environment_seed']] = row
    # Check the main MAEs against the original code, rather than a new convention.
    original = original_metrics(record)
    aliases = {'endpoint_force_mae_N': 'force_mae_N', 'interval_peak_mae_N': 'substep_peak_mae_N', 'height_mae_m': 'height_mae_m'}
    for name, alias in aliases.items():
        aggregate = sum(row[name][0] for row in result.values())/sum(row[name][1] for row in result.values())
        if not np.isclose(aggregate, original[alias], rtol=2e-6, atol=1e-8):
            raise ValueError(f'New summary differs from original {alias}')
    return result, original


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(np.shape(numerator), np.nan, dtype=float), where=denominator > 0)


def grouped_environment_statistics(records, metric, draws):
    """Average fixed-model pooled errors; resample entire aligned environments.

records is an ordered list of per-model {environment_seed: sufficient_stats}.
One shared draw array is reused for every model, method, and metric in a domain.
No training seed, horizon, frame, or component is sampled independently.
"""
    environment_ids = sorted(records[0])
    if any(sorted(record) != environment_ids for record in records):
        raise ValueError('Model seeds do not cover exactly the same environments')
    if draws.ndim != 2 or draws.shape[1] != len(environment_ids) or draws.min() < 0 or draws.max() >= len(environment_ids):
        raise ValueError('Bootstrap draws must index whole shared environments')
    numerator = np.asarray([[record[e][metric][0] for e in environment_ids] for record in records], dtype=float)
    denominator = np.asarray([[record[e][metric][1] for e in environment_ids] for record in records], dtype=float)
    per_model = _ratio(numerator.sum(1), denominator.sum(1))
    point = per_model.mean()
    sampled = _ratio(numerator[:, draws].sum(2), denominator[:, draws].sum(2)).mean(0)
    finite = sampled[np.isfinite(sampled)]
    low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
    return {'estimate': point, 'ci_low': low, 'ci_high': high,
            'environment_episodes': len(environment_ids), 'fixed_model_seeds': len(records),
            'training_seed_sd': per_model.std(ddof=1) if len(records) > 1 else None,
            'valid_bootstrap_resamples': len(finite), 'per_model_estimates': per_model.tolist()}


def paired_difference(left, right, metric):
    if sorted(left) != sorted(right):
        raise ValueError('Paired comparison requires identical environments')
    result = {}
    for environment in sorted(left):
        ln, ld = left[environment][metric]
        rn, rd = right[environment][metric]
        if ld != rd:
            raise ValueError('Paired errors must have identical target/component counts')
        result[environment] = {metric: (ln-rn, ld)}
    return result


def method_name(variant, mode):
    return variant+'_'+mode


def build_summaries(all_statistics, protocol):
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    draws = {domain: rng.integers(0, 30, (protocol['bootstrap_resamples'], 30)) for domain in DOMAINS}
    aggregate, per_run, paired = [], [], []
    for (method, domain), seeds in sorted(all_statistics.items()):
        if sorted(seeds) != list(TRAINING_SEEDS):
            raise ValueError('Every method requires all three fixed model seeds')
        records = [seeds[seed] for seed in TRAINING_SEEDS]
        metric_names = sorted(next(iter(records[0].values())))
        for metric in metric_names:
            aggregate.append({'method': method, 'domain': domain, 'metric': metric,
                **grouped_environment_statistics(records, metric, draws[domain])})
            for seed in TRAINING_SEEDS:
                per_run.append({'method': method, 'domain': domain, 'training_seed': seed, 'metric': metric,
                    **grouped_environment_statistics([seeds[seed]], metric, draws[domain])})
    comparisons = [('visuotactile_world_model', 'vision_world_model'),
                   ('visuotactile_world_model', 'visuotactile_persistence'),
                   ('visuotactile_shuffled_touch', 'visuotactile_world_model')]
    for left, right in comparisons:
        for domain in DOMAINS:
            ls, rs = all_statistics[(left, domain)], all_statistics[(right, domain)]
            metric_names = sorted(set(next(iter(ls[0].values()))) & set(next(iter(rs[0].values()))))
            for metric in metric_names:
                differences = [paired_difference(ls[seed], rs[seed], metric) for seed in TRAINING_SEEDS]
                paired.append({'comparison': left+' minus '+right, 'domain': domain, 'metric': metric,
                    **grouped_environment_statistics(differences, metric, draws[domain])})
    return aggregate, per_run, paired


def main():
    revision = ROOT/'outputs/revision_v2'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=revision/'protocol.json')
    parser.add_argument('--models', type=Path, default=revision/'world_models')
    parser.add_argument('--trajectories', type=Path, default=revision/'evaluations/script')
    parser.add_argument('--out', type=Path, default=revision/'forecast_analysis')
    parser.add_argument('--calibration-filename', default='calibration.json')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--run-test-evaluation', action='store_true',
                        help='Explicitly enable the prospective test read after the root runner authorizes it')
    args = parser.parse_args()
    if not args.run_test_evaluation:
        parser.error('No test files opened. Run only when authorized, with --run-test-evaluation.')
    if args.trajectories.name != 'script' or args.trajectories.parent.name != 'evaluations':
        parser.error('Only the common evaluations/script directory is accepted')
    if Path(args.calibration_filename).name != args.calibration_filename or args.batch_size < 1:
        parser.error('Invalid calibration filename or batch size')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no unreported fallback')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use a fresh output directory; completed analyses cannot be silently replaced')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    test_rows = validate_protocol(protocol)
    torch.set_num_threads(4)
    locked = {'protocol': {'path': str(args.protocol.resolve()), 'sha256': sha256(args.protocol)},
              'source_files': {str(path.resolve()): sha256(path) for path in
                               [Path(__file__), WM_SOURCE/'model.py', WM_SOURCE/'data.py', WM_SOURCE/'train.py']},
              'models': [], 'trajectories': []}
    # Validate every weight/calibration artifact before opening any test trace.
    for variant in VARIANTS:
        for seed in TRAINING_SEEDS:
            checkpoint = args.models/f'{variant}_seed{seed}'/'best.pt'
            calibration_path = checkpoint.parent/args.calibration_filename
            model, payload = load_model(checkpoint, 'cpu')
            validate_checkpoint(payload, model, variant, seed, protocol)
            calibration = validate_calibration(json.loads(calibration_path.read_text(encoding='utf-8')))
            locked['models'].append({'variant': variant, 'training_seed': seed,
                'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256(checkpoint),
                'calibration_file': str(calibration_path.resolve()), 'calibration_sha256': sha256(calibration_path),
                'calibration': calibration, 'selected_epoch': payload.get('epoch'),
                'validation_loss': payload.get('validation_loss'), 'parameters': model.num_parameters})
            del model, payload
    episodes = []
    for row in test_rows:
        path = args.trajectories/f'episode_{row["seed"]}.npz'
        episodes.append(load_script_episode(path, row))
        locked['trajectories'].append({'path': str(path.resolve()), 'sha256': sha256(path),
            'environment_seed': row['seed'], 'episode_id': row['episode_id'], 'domain': DOMAIN_MAP[row['domain']]})
    args.out.mkdir(parents=True)
    write_json(args.out/'locked_inputs.json', locked)
    all_statistics, episode_rows, original_rows, shuffled_pairs = {}, [], [], []
    for domain in DOMAINS:
        group = [ep for ep in episodes if ep['split'] == domain]
        for i, ep in enumerate(group):
            shuffled_pairs.append({'domain': domain, 'recipient_seed': ep['environment_seed'],
                'donor_seed': group[(i+1) % len(group)]['environment_seed'],
                'rule': 'cyclic next episode within domain, same observation indices; only encoder touch replaced'})
    for info in locked['models']:
        variant, seed = info['variant'], info['training_seed']
        model, payload = load_model(info['checkpoint'], args.device)
        validate_checkpoint(payload, model, variant, seed, protocol)
        model.eval().requires_grad_(False)
        for domain in DOMAINS:
            modes = ('world_model',) if variant == 'vision' else ('world_model', 'persistence', 'shuffled_touch')
            for mode in modes:
                record, domain_episodes = forecast_records(model, episodes, domain, args.device, args.batch_size, mode)
                # Original WM radii are not calibrated for a changed predictor.
                calibration = info['calibration'] if mode == 'world_model' else None
                stats, original = sufficient_statistics(record, domain_episodes, calibration)
                method = method_name(variant, mode)
                all_statistics.setdefault((method, domain), {})[seed] = stats
                original_rows.append({'method': method, 'domain': domain, 'training_seed': seed, **original})
                for episode in domain_episodes:
                    environment = episode['environment_seed']
                    row = {'method': method, 'domain': domain, 'training_seed': seed,
                           'environment_seed': environment, 'episode_id': episode['episode_id'], 'windows': 30}
                    for metric, (numerator, denominator) in stats[environment].items():
                        row[metric] = numerator/denominator if denominator else None
                    row['contact_force_error_sum'] = stats[environment]['contact_phase_endpoint_force_mae_N'][0]
                    row['contact_force_component_count'] = stats[environment]['contact_phase_endpoint_force_mae_N'][1]
                    episode_rows.append(row)
                del record
        del model, payload
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        print(json.dumps({'forecasted_variant': variant, 'training_seed': seed, 'environments': 120}), flush=True)
    aggregate, per_run, paired = build_summaries(all_statistics, protocol)
    # Prevent metadata claiming frozen inputs if another process changed them.
    checks = [(args.protocol, locked['protocol']['sha256'])]
    checks += [(i['checkpoint'], i['checkpoint_sha256']) for i in locked['models']]
    checks += [(i['calibration_file'], i['calibration_sha256']) for i in locked['models']]
    checks += [(i['path'], i['sha256']) for i in locked['trajectories']]
    checks += list(locked['source_files'].items())
    if any(sha256(path) != digest for path, digest in checks):
        raise RuntimeError('A frozen input changed during evaluation; results not finalized')
    summary = {'aggregate': aggregate, 'individual_models': per_run, 'paired_differences': paired}
    write_json(args.out/'forecast_analysis.json', summary)
    write_csv(args.out/'aggregate_metrics.csv', [{**r, 'per_model_estimates': json.dumps(json_native(r['per_model_estimates']))} for r in aggregate])
    write_csv(args.out/'model_metrics.csv', [{**r, 'per_model_estimates': json.dumps(json_native(r['per_model_estimates']))} for r in per_run])
    write_csv(args.out/'paired_differences.csv', [{**r, 'per_model_estimates': json.dumps(json_native(r['per_model_estimates']))} for r in paired])
    write_csv(args.out/'episode_metrics.csv', episode_rows)
    write_json(args.out/'original_metric_summaries.json', original_rows)
    write_json(args.out/'shuffled_touch_pairs.json', shuffled_pairs)
    write_json(args.out/'analysis_metadata.json', {
        'complete': True, 'environment_episodes': 120, 'environments_per_domain': 30,
        'fixed_training_seeds': list(TRAINING_SEEDS), 'history': HISTORY, 'forecast_horizon': HORIZON,
        'stride': STRIDE, 'windows_per_episode': 30, 'window_starts': list(range(0, 146, 5)),
        'trajectory_distribution': 'the same independently executed script trajectories for every predictor',
        'prediction_conditioning': 'actual subsequent recorded actions; not an online policy-return evaluation',
        'endpoint_metric': 'MAE over all windows, five horizons, six signed/normal tactile components',
        'peak_metric': 'MAE over all windows, five horizons, two per-finger physics-substep interval maxima',
        'height_metric': 'relative object height MAE in meters; no threshold-success substitution',
        'persistence': 'VT: last observed tactile endpoint held constant; current decoded peak and height held constant; no transition',
        'shuffled_touch': 'cyclic next recipient-domain episode, observed tactile history only; targets/actions/RGB/proprio unchanged',
        'calibration': 'unchanged per-WM 48-episode radii; all-window/five-horizon/two-finger peak and absolute-height joint episode coverage',
        'baseline_calibration': 'not reported: persistence/shuffling change the predictor and have no separately calibrated radii',
        'coverage_scope': 'empirical only, including ID: calibration mixes script/feedback while these traces use script alone; OOD adds further shift',
        'bootstrap_resamples': protocol['bootstrap_resamples'], 'bootstrap_seed': protocol['bootstrap_seed'],
        'confidence_interval': '95% percentile whole-environment bootstrap within each domain; identical draws for all fixed seeds/methods/metrics',
        'training_seed_sd': 'separate sample SD over three fitted models; not additional independent environments',
        'degenerate_intervals': 'identical observed outcomes may produce a point interval; not proof of zero population uncertainty',
        'test_used_for_training_selection_calibration': False, 'no_safety_guarantee': True})
    print(json.dumps({'complete': True, 'episode_metric_rows': len(episode_rows),
                      'aggregate_metric_rows': len(aggregate), 'output': str(args.out.resolve())}), flush=True)


if __name__ == '__main__':
    main()
