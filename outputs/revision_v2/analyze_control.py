"""Prespecified paired control analysis; uses only Python's standard library.

Do not run on test results until the fixed matrix is complete and analysis is
authorized. --allow-partial writes completeness information only, never an
estimate based on an opportunistic intersection of completed environments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ('id', 'geometry_ood', 'physics_ood', 'combined_ood')
LEARNED_GROUPS = ('WM_vision', 'WM_visuotactile', 'BC_vision', 'BC_visuotactile',
                  'IQL_vision', 'IQL_visuotactile')
SEEDED_GROUPS = LEARNED_GROUPS + ('model_guard', 'model_no_guard')
SINGLE_GROUPS = ('script', 'force_feedback', 'reactive_guard', 'reactive_no_guard')
GROUPS = SEEDED_GROUPS + SINGLE_GROUPS
SUPPLEMENTARY_GROUPS = ('WM_BC_vision', 'WM_BC_visuotactile')
CONTRASTS = (
    ('WM_visuotactile', 'WM_vision'),
    ('IQL_visuotactile', 'IQL_vision'),
    ('WM_visuotactile', 'IQL_visuotactile'),
    ('model_guard', 'reactive_guard'),
    ('model_guard', 'model_no_guard'),
    ('reactive_guard', 'reactive_no_guard'),
    ('model_guard', 'force_feedback'),
)
SUPPLEMENTARY_CONTRASTS = (
    ('WM_vision', 'WM_BC_vision'),
    ('WM_visuotactile', 'WM_BC_visuotactile'),
    ('WM_BC_visuotactile', 'BC_visuotactile'),
)
RATE_METRICS = ('strict_lift', 'joint_success', 'force_violation')
METRICS = RATE_METRICS + ('peak_N', 'max_height_m', 'setup_peak_N', 'strict_longest')
LEGACY_METRICS = tuple(k for k in METRICS if k != 'strict_longest')
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20401010
LEGACY_MAPPING = {
    'strict_lift': 'sustained_10cm_lift_success',
    'joint_success': 'sustained_10cm_lift_within_force_budget',
    'force_violation': 'force_budget_exceeded',
    'peak_N': 'full_episode_peak_normal_N',
    'max_height_m': 'max_relative_height_m',
    'setup_peak_N': 'setup_peak_normal_N',
}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path, allow_partial=False):
    if not path.exists():
        return [], False
    text = path.read_text(encoding='utf-8-sig')
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # An append may be in flight during a progress check. Never ignore
            # malformed completed lines or silently include a partial record.
            if allow_partial and index == len(lines) - 1 and not text.endswith('\n'):
                return rows, True
            raise ValidationError(f'{path}:{index+1}: invalid JSON') from exc
    return rows, False


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_metric_row(row, metrics, label):
    for metric in metrics:
        require(metric in row, f'{label}: missing metric {metric}')
        value = row[metric]
        require(isinstance(value, (int, float)) and math.isfinite(value),
                f'{label}: {metric} must be a finite numeric value or boolean')
        if metric in RATE_METRICS:
            require(value in (0, 1), f'{label}: {metric} must be boolean/0/1')
    require(row['peak_N'] >= 0 and row['setup_peak_N'] >= 0,
            f'{label}: negative normal-force peak')
    require(row['setup_peak_N'] <= row['peak_N'] + 1e-9,
            f'{label}: setup peak exceeds full-episode peak')
    require(bool(row['force_violation']) == (row['peak_N'] > 8.),
            f'{label}: force violation disagrees with full-episode 8 N threshold')
    require(bool(row['joint_success']) == (bool(row['strict_lift']) and not row['force_violation']),
            f'{label}: joint_success disagrees with lift and force outcomes')
    if row['strict_lift']:
        require(row['max_height_m'] >= .10 - 1e-7,
                f'{label}: successful strict lift never reached 10 cm')
    if 'strict_longest' in metrics:
        require(isinstance(row['strict_longest'], int) and not isinstance(row['strict_longest'], bool)
                and 0 <= row['strict_longest'] <= 135,
                f'{label}: strict_longest must be an integer in [0,135]')
        require(bool(row['strict_lift']) == (row['strict_longest'] >= 10),
                f'{label}: strict lift disagrees with consecutive-observation count')
    require(isinstance(row.get('params'), dict), f'{label}: missing parameter dictionary')
    require(row['params'].get('force_budget') == 8., f'{label}: force budget differs from protocol')


def validate_prescribed_params(actual, expected, label):
    # reset() adds half_size and replaces feasibility with a derived calculation
    # using the actual compiled legacy dimensions. Those are still compared
    # exactly ACROSS methods below; every prescribed input remains mandatory.
    for key, value in expected.items():
        if key == 'feasibility':
            continue
        require(key in actual and actual[key] == value,
                f'{label}: parameter {key!r} differs from frozen manifest')


def planned_contrasts(methods):
    groups = {job['group'] for job in methods}
    supplemental = groups.intersection(SUPPLEMENTARY_GROUPS)
    require(not supplemental or supplemental == set(SUPPLEMENTARY_GROUPS),
            'Supplementary WM-BC amendment requires both modality groups')
    return CONTRASTS + (SUPPLEMENTARY_CONTRASTS if supplemental else ())


def collect_matrix(methods, expected, path_for, metrics, domains, allow_partial=False,
                   legacy=False):
    """Validate every available record; return matrix, progress, missing list."""
    expected_keys = set(expected)
    matrix, progress, missing = {}, [], []
    reference_params = {}
    for job in methods:
        path = path_for(job)
        rows, in_flight = read_jsonl(path, allow_partial)
        indexed = {}
        for raw in rows:
            row = dict(raw)
            label = f'{job["name"]} seed={row.get("seed")}'
            require(isinstance(row.get('seed'), int) and not isinstance(row.get('seed'), bool),
                    f'{label}: seed must be an integer')
            require(row.get('domain') in domains, f'{label}: unknown domain {row.get("domain")}')
            key = (row['domain'], row['seed'])
            require(key in expected_keys, f'{label}: unexpected environment {key}')
            require(key not in indexed, f'{label}: duplicate environment')
            spec = expected[key]
            if legacy:
                for new, old in LEGACY_MAPPING.items():
                    require(old in row, f'{label}: missing legacy field {old}')
                    row[new] = row[old]
            else:
                require(row.get('geometry') == spec['params']['geometry'],
                        f'{label}: geometry differs from frozen manifest')
                validate_prescribed_params(row.get('params', {}), spec['params'], label)
                require(row['params'].get('geometry', row['params'].get('geometry_key')) == row['geometry'],
                        f'{label}: top-level and parameter geometry disagree')
            validate_metric_row(row, metrics, label)
            # Full exact comparison detects different object sizes, nuisance
            # parameters or force settings even if all methods share a seed ID.
            fingerprint = canonical(row['params'])
            if key in reference_params:
                require(reference_params[key] == fingerprint,
                        f'{label}: full params differ across paired methods')
            else:
                reference_params[key] = fingerprint
            indexed[key] = row
        absent = sorted(expected_keys - set(indexed))
        if absent:
            missing.append(f'{job["name"]}: missing {len(absent)}/{len(expected_keys)} environments')
        if in_flight:
            missing.append(f'{job["name"]}: incomplete final JSONL record')
        matrix[job['name']] = indexed
        progress.append(dict(name=job['name'], group=job['group'],
            training_seed=job.get('seed'), path=str(path), completed=len(indexed),
            expected=len(expected_keys), complete=not absent and not in_flight,
            in_flight_record=in_flight, missing_environments=[list(key) for key in absent]))
    return matrix, progress, missing


def percentile(values, probability):
    """Linear interpolation at (N-1)*p, matching NumPy's default quantile."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def interval(draws):
    return percentile(draws, .025), percentile(draws, .975)


def bootstrap_plan(keys, domains, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed)
    environment_keys, indices = {}, {}
    for domain in domains:
        environment_keys[domain] = sorted(key for key in keys if key[0] == domain)
        n = len(environment_keys[domain])
        require(n > 0, f'No environments in domain {domain}')
        indices[domain] = [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]
    return environment_keys, indices


def analyze_matrix(methods, matrix, expected, domains, metrics, contrasts,
                   resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Average fixed policies within environment; bootstrap environments only."""
    groups = {}
    for job in methods:
        groups.setdefault(job['group'], []).append(job)
    keys, draws = bootstrap_plan(expected, domains, resamples, seed)
    stats, seed_variation, policy_stats = [], [], []
    estimates, replicates = {}, {}
    for group, jobs in groups.items():
        for metric in metrics:
            for domain in domains:
                ordered_keys = keys[domain]
                values = [statistics.fmean(float(matrix[job['name']][key][metric]) for job in jobs)
                          for key in ordered_keys]
                estimates[(group, domain, metric)] = statistics.fmean(values)
                replicates[(group, domain, metric)] = [statistics.fmean(values[i] for i in draw)
                                                       for draw in draws[domain]]
            # Pooled rows preserve each domain's environment count on every
            # replicate; this is a stratified bootstrap, not an IID domain mix.
            count = sum(len(keys[domain]) for domain in domains)
            estimates[(group, 'all', metric)] = sum(
                estimates[(group, domain, metric)] * len(keys[domain]) for domain in domains) / count
            replicates[(group, 'all', metric)] = [sum(
                replicates[(group, domain, metric)][b] * len(keys[domain]) for domain in domains) / count
                for b in range(resamples)]
            for domain in tuple(domains) + ('all',):
                selected_keys = sorted(expected) if domain == 'all' else keys[domain]
                policy_means = []
                for job in jobs:
                    mean = statistics.fmean(float(matrix[job['name']][key][metric]) for key in selected_keys)
                    policy_means.append(mean)
                    policy_stats.append(dict(group=group, method=job['name'], training_seed=job.get('seed'),
                        domain=domain, metric=metric, estimate=mean, n_environments=len(selected_keys)))
                low, high = interval(replicates[(group, domain, metric)])
                stats.append(dict(group=group, domain=domain, metric=metric,
                    metric_type='rate' if metric in RATE_METRICS else 'mean',
                    estimate=estimates[(group, domain, metric)], ci_low=low, ci_high=high,
                    n_environments=len(selected_keys), n_policies=len(jobs),
                    ci_method='paired_environment_percentile_bootstrap_95pct'))
                seed_variation.append(dict(group=group, domain=domain, metric=metric,
                    n_policies=len(jobs), mean=statistics.fmean(policy_means),
                    sample_sd=statistics.stdev(policy_means) if len(jobs) > 1 else None,
                    minimum=min(policy_means), maximum=max(policy_means),
                    training_seeds=[job.get('seed') for job in jobs], policy_estimates=policy_means))
    effects = []
    for first, second in contrasts:
        require(first in groups and second in groups, f'Missing planned contrast group: {first} - {second}')
        for domain in tuple(domains) + ('all',):
            for metric in metrics:
                delta = [a - b for a, b in zip(replicates[(first, domain, metric)],
                                                replicates[(second, domain, metric)])]
                low, high = interval(delta)
                estimate = estimates[(first, domain, metric)] - estimates[(second, domain, metric)]
                effects.append(dict(contrast=f'{first} - {second}', first=first, second=second,
                    domain=domain, metric=metric, estimate_difference=estimate,
                    ci_low=low, ci_high=high,
                    unit='proportion_difference' if metric in RATE_METRICS else 'native_metric_difference',
                    percentage_point_difference=100 * estimate if metric in RATE_METRICS else None,
                    percentage_point_ci_low=100 * low if metric in RATE_METRICS else None,
                    percentage_point_ci_high=100 * high if metric in RATE_METRICS else None,
                    n_environments=sum(len(keys[d]) for d in domains) if domain == 'all' else len(keys[domain]),
                    ci_method='paired_environment_percentile_bootstrap_95pct'))
    return dict(group_stats=stats, contrasts=effects, seed_variation=seed_variation,
                policy_stats=policy_stats,
                bootstrap=dict(resamples=resamples, seed=seed, rng='Python random.Random MT19937',
                    same_resamples_across_groups_metrics=True, sampling_unit='environment',
                    policy_aggregation='equal-weight average within each environment before resampling',
                    pooled_domain_aggregation='environment-count-weighted, stratified resampling',
                    interval='pointwise 95% percentile; no multiplicity correction',
                    training_seed_resampled=False,
                    environment_order={d: [list(k) for k in keys[d]] for d in domains},
                    resample_indices_sha256=hashlib.sha256(canonical(draws).encode()).hexdigest()))


def prepare_main(base, allow_partial=False):
    methods_path, protocol_path = base / 'methods.json', base / 'protocol.json'
    methods, protocol = read_json(methods_path), read_json(protocol_path)
    require(isinstance(methods, list) and methods, 'methods.json must be a nonempty list')
    require(len({job['name'] for job in methods}) == len(methods), 'Duplicate method names')
    actual_groups = {job['group'] for job in methods}
    require(actual_groups in (set(GROUPS), set(GROUPS + SUPPLEMENTARY_GROUPS)),
            'Method groups differ from the original or amended planned matrix')
    contrasts = planned_contrasts(methods)
    seeds = protocol.get('training_seeds', [0, 1, 2])
    require(seeds == [0, 1, 2], 'Training seeds differ from fixed [0,1,2] plan')
    for group in GROUPS + tuple(g for g in SUPPLEMENTARY_GROUPS if g in actual_groups):
        jobs = [job for job in methods if job['group'] == group]
        if group in SEEDED_GROUPS + SUPPLEMENTARY_GROUPS:
            require(len(jobs) == 3 and sorted(job.get('seed', -1) for job in jobs) == seeds,
                    f'{group}: expected exactly one policy for each training seed {seeds}')
        else:
            require(len(jobs) == 1, f'{group}: expected one fixed controller')
    require(protocol.get('bootstrap_resamples') == BOOTSTRAP_RESAMPLES and
            protocol.get('bootstrap_seed') == BOOTSTRAP_SEED, 'Bootstrap plan differs from frozen protocol')
    expected = {}
    for spec in protocol['control_test']:
        key = (spec['domain'], spec['seed'])
        require(key not in expected and spec['domain'] in DOMAINS, 'Invalid/duplicate protocol environment')
        expected[key] = spec
    require(all(sum(key[0] == d for key in expected) == 30 for d in DOMAINS),
            'Expected exactly 30 environments per main domain')
    matrix, progress, missing = collect_matrix(methods, expected,
        lambda job: base / 'evaluations' / job['name'] / 'episodes.jsonl', METRICS, DOMAINS, allow_partial)
    input_hashes = {'methods.json': sha256(methods_path), 'protocol.json': sha256(protocol_path)}
    amendment_path = base / 'protocol_amendment.json'
    if amendment_path.exists():
        input_hashes['protocol_amendment.json'] = sha256(amendment_path)
    return dict(methods=methods, expected=expected, matrix=matrix, progress=progress, missing=missing,
        input_hashes=input_hashes, contrasts=contrasts)


def prepare_legacy(base, original_base, allow_partial=False):
    methods = []
    for group, prefix in [('WM_vision', 'vision'), ('WM_visuotactile', 'visuotactile')]:
        for seed in (0, 1, 2):
            methods.append(dict(name=f'{prefix}_height_seed{seed}_rl', group=group, seed=seed))
    expected = {('ood' if i % 2 else 'id', 20291000+i): {} for i in range(20)}

    def path_for(job):
        source = base / 'legacy_visual_evaluation' if job['group'] == 'WM_vision' else original_base
        return source / job['name'] / 'episodes.jsonl'

    matrix, progress, missing = collect_matrix(methods, expected, path_for,
        LEGACY_METRICS, ('id', 'ood'), allow_partial, legacy=True)
    return dict(methods=methods, expected=expected, matrix=matrix, progress=progress, missing=missing)


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs' / 'revision_v2')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--legacy-original', type=Path,
                        default=ROOT / 'outputs' / 'simulation' / 'evaluation_goal_aligned')
    parser.add_argument('--allow-partial', action='store_true', help='Completeness report only if any data are missing')
    parser.add_argument('--skip-legacy', action='store_true')
    args = parser.parse_args(argv)
    base, out = args.root.resolve(), (args.out or args.root / 'analysis').resolve()
    prepared = {'main': prepare_main(base, args.allow_partial)}
    if not args.skip_legacy:
        prepared['legacy_visual'] = prepare_legacy(base, args.legacy_original.resolve(), args.allow_partial)
    missing = [f'{name}: {issue}' for name, data in prepared.items() for issue in data['missing']]
    if missing and not args.allow_partial:
        raise ValidationError('Incomplete fixed matrix; no estimates written:\n' + '\n'.join(missing))
    out.mkdir(parents=True, exist_ok=True)
    if missing:
        progress = dict(status='INCOMPLETE_PROGRESS_ONLY', analysis_performed=False, missing=missing,
                        datasets={name: data['progress'] for name, data in prepared.items()})
        (out / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
        for name, data in prepared.items():
            write_csv(out / f'{name}_progress.csv', data['progress'])
        # Do not overwrite a completed analysis README or falsely label a
        # truncated subset as the completed fixed matrix.
        (out / 'PROGRESS.md').write_text('INCOMPLETE — progress only. No outcome estimates or confidence intervals '
            'were computed. Missing environments must be completed before analysis.\n', encoding='utf-8')
        print(json.dumps({'status': progress['status'], 'missing_items': len(missing), 'out': str(out)}))
        return progress
    report = dict(status='COMPLETE', primary_metric='joint_success',
        planned_contrasts=[list(pair) for pair in prepared['main']['contrasts']],
        main_input_hashes=prepared['main']['input_hashes'],
        analysis_source_sha256=sha256(Path(__file__)), datasets={})
    for name, data in prepared.items():
        is_main = name == 'main'
        analyzed = analyze_matrix(data['methods'], data['matrix'], data['expected'],
            DOMAINS if is_main else ('id', 'ood'), METRICS if is_main else LEGACY_METRICS,
            data['contrasts'] if is_main else (('WM_visuotactile', 'WM_vision'),))
        analyzed['methods'] = data['methods']
        analyzed['input_files'] = [dict(path=row['path'], sha256=sha256(row['path'])) for row in data['progress']]
        report['datasets'][name] = analyzed
        prefix = 'control' if is_main else 'legacy_visual'
        for key in ('group_stats', 'contrasts', 'seed_variation', 'policy_stats'):
            write_csv(out / f'{prefix}_{key}.csv', analyzed[key])
    (out / 'control_analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    readme = '''Fixed control-matrix analysis is complete. All expected environment seeds and parameter dictionaries were checked before estimating outcomes.

The primary outcome is joint_success: a lift of at least 10 cm for at least 10 consecutive observations before lowering, with the complete-episode per-finger peak at most 8 N. Strict lift and force violation are reported separately. Rates are proportions from 0 to 1; contrast CSVs also give percentage-point differences. Continuous outcomes retain their stated units.

Each group's estimate first averages its fixed policies within the same environment, then averages environments. The 95% percentile intervals use 2,000 environment-bootstrap replicates with seed 20401010. Every group and metric uses the same resampled indices within a domain, so contrasts retain environment pairing. Pooled rows use a stratified bootstrap preserving domain counts. These intervals are conditional on the trained policies; they do not estimate retraining uncertainty. Training-seed sample SD, range and individual policy means are reported separately; a singleton has no estimated seed SD.

The seven original prespecified main contrasts are retained. When the pretest WM-BC amendment is present, three supplementary contrasts compare WM_vision with WM_BC_vision, WM_visuotactile with WM_BC_visuotactile, and WM_BC_visuotactile with BC_visuotactile. The first two compare policy learning on the same world-model representation; they measure the combined effect of the subsequent policy-improvement procedure and its extra optimization budget. The last is a representation/pipeline comparison with distinct training budgets, not an isolated representation effect. The exact included contrasts are listed in control_analysis.json. All intervals are pointwise, without multiplicity correction; no significance screening, selected contrasts, or p-value search is performed. Constant outcomes may yield degenerate bootstrap intervals, which do not prove zero population uncertainty. Read the primary outcome first and treat the other metrics as supporting descriptions.

Legacy visual ablation, when included, is analyzed separately using the six fixed height-policy checkpoints and shared environment seeds 20291000–20291019. Its original ID/OOD distributions are not pooled with the revised feasible-domain experiment. Legacy strict_lift, joint_success, and force_violation map to sustained_10cm_lift_success, sustained_10cm_lift_within_force_budget, and force_budget_exceeded respectively.

CSV tables provide group intervals, paired contrasts, seed variation, and each policy's mean. JSON includes the complete analysis specification, input hashes, ordered environment identifiers and resample-index hashes. Analytic capacity screening and simulation results do not establish successful grasping or physical safety outside this rigid-box benchmark.
'''
    (out / 'README.md').write_text(readme, encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'datasets': list(report['datasets']), 'out': str(out)}))
    return report


if __name__ == '__main__':
    main()
