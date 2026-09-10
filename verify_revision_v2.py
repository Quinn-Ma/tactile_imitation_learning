"""Verify archived revision-v2 records with the Python standard library.

This checks text records and provenance; it does not recreate trajectories,
checkpoints, or simulator outcomes. Run without Python's -O flag.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct


ROOT = Path(__file__).resolve().parent
RAW_SUFFIXES = {'.pt', '.pth', '.npz', '.npy', '.pkl', '.pickle', '.zip', '.pdf', '.mp4'}
HOST_PATH = re.compile(r'\b[A-Za-z]:[/\\]|/(?:Users|home)/[^/\s]+/')
SECRET = re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})\b')
PRIVATE_MARKERS = ('https-'+'chatgpt-com-share-', '.cache/'+'codex-runtimes', '.cache\\'+'codex-runtimes')
DOMAINS = ('id', 'geometry_ood', 'physics_ood', 'combined_ood')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def json_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def validate_protocol(base):
    protocol = read_json(base/'protocol.json')
    methods = read_json(base/'methods.json')
    collection, cases, diagnostic = (protocol[k] for k in ('collection', 'control_test', 'counterfactual'))
    require(Counter(r['split'] for r in collection) == {'train': 240, 'val': 48, 'calibration': 48}, 'Collection split counts')
    for split, count in [('train', 120), ('val', 24), ('calibration', 24)]:
        require(Counter(r['controller'] for r in collection if r['split'] == split) == {'script': count, 'force_feedback': count}, 'Collection controller balance')
    require(Counter(r['domain'] for r in cases) == {d: 30 for d in DOMAINS}, '120 test environments / four domains')
    require([r['seed'] for r in diagnostic] == list(range(20364000, 20364012)), '12 diagnostic seeds')
    require(len({r['seed'] for r in collection+cases+diagnostic}) == 468, 'Splits overlap or duplicate seeds')
    for domain, start in zip(DOMAINS, (20350000, 20351000, 20352000, 20353000)):
        require({r['seed'] for r in cases if r['domain'] == domain} == set(range(start, start+30)), 'Fixed test seeds')
    require(protocol['training_seeds'] == [0, 1, 2], 'Training seeds')
    require((protocol['horizon'], protocol['takeover'], protocol['control_end']) == (150, 44, 135), 'Action intervals')
    require(protocol['main_control_dimensions'] == [2, 6] and protocol['zero_intervention_dimensions'] == [0, 1, 3, 4, 5], '2D controls')
    require((protocol['bootstrap_resamples'], protocol['bootstrap_seed']) == (2000, 20401010), 'Bootstrap protocol')
    expected_groups = {f'{kind}_{variant}': 3 for kind in ('WM', 'WM_BC', 'BC', 'IQL') for variant in ('vision', 'visuotactile')}
    expected_groups.update(script=1, force_feedback=1, model_guard=3, model_no_guard=3, reactive_guard=1, reactive_no_guard=1)
    require(len(methods) == len({m['name'] for m in methods}) == 34, '34 distinct policies')
    require(Counter(m['group'] for m in methods) == expected_groups, 'Frozen 34-policy groups')
    for group, count in expected_groups.items():
        if count == 3:
            require(sorted(m['seed'] for m in methods if m['group'] == group) == [0, 1, 2], 'Three fixed training seeds per learned group')
    return protocol, methods


def validate_sensor_protocol(base):
    plan = read_json(base/'sensor_stress_protocol.json')
    main, methods = validate_protocol(base)
    require(plan['main_protocol_sha256'] == digest(base/'protocol.json') and plan['main_methods_sha256'] == digest(base/'methods.json'), 'Sensor protocol main-input hashes')
    require(len(plan['cases']) == 30 and len(plan['methods']) == 11, 'Sensor stress fixed 11 x 30 matrix')
    require(Counter(m['group'] for m in plan['methods']) == {'WM_visuotactile': 3, 'IQL_visuotactile': 3, 'model_guard': 3, 'force_feedback': 1, 'reactive_guard': 1}, 'Sensor stress method groups')
    require(all(m in methods for m in plan['methods']), 'Sensor stress policies must match main policies')
    require({r['seed'] for r in plan['cases']} == {r['seed'] for r in main['control_test'] if r['domain'] == 'id'}, 'Sensor stress paired ID environments')
    for case in plan['cases']:
        require(case['domain'] == 'id' and len(case['normal_residual_N']) == 93 and all(len(row) == 2 for row in case['normal_residual_N']), 'Embedded residual shape')
        values = [v for row in case['normal_residual_N'] for v in row]
        require(all(math.isfinite(v) for v in values), 'Finite embedded residuals')
        packed = b''.join(struct.pack('<f', v) for v in values)
        require(hashlib.sha256(packed).hexdigest() == case['residual_float32_sha256'], 'Embedded float32 residual digest')
    return plan


def validate_sensor_records(base):
    plan = validate_sensor_protocol(base)
    cases = {r['seed']: r for r in plan['cases']}
    protocol_sha = digest(base/'sensor_stress_protocol.json')
    total = 0
    for method in plan['methods']:
        directory = base/'sensor_stress_evaluations'/method['name']
        rows = json_rows(directory/'episodes.jsonl')
        require(len(rows) == 30 and {r['seed'] for r in rows} == set(cases), 'Sensor stress complete paired environments')
        inputs = read_json(directory/'inputs.json')
        require(inputs['method'] == method and inputs['smoke'] is False and inputs['sensor_protocol_sha256'] == protocol_sha, 'Sensor stress inputs')
        require(inputs['protocol_sha256'] == plan['main_protocol_sha256'] and inputs['residual_source_sha256'] == plan['source']['sha256'], 'Sensor stress inherited provenance')
        marker = read_json(directory/'summary.json')
        require(marker['completed'] == 30 and marker['inputs'] == inputs, 'Sensor stress completion marker')
        clean = read_json(base/'evaluations'/method['name']/'inputs.json')
        require(all(clean[key] == inputs[key] for key in ('model_inputs', 'adapter_sha256', 'original_adapter_sha256')), 'Sensor/clean model and simulator equality')
        for row in rows:
            case = cases[row['seed']]
            require(row['method'] == method['name'] and row['group'] == method['group'] and row['domain'] == 'id' and row['smoke'] is False, 'Sensor stress row identity')
            require(row['sensor_protocol_sha256'] == protocol_sha and row['sensor_residual_float32_sha256'] == case['residual_float32_sha256'], 'Sensor stress residual protocol')
            require(row['sensor_draws'] == case['sensor_draws'], 'Sensor stress frozen block assignment')
            require(math.isfinite(row['peak_N']) and row['peak_N'] >= 0, 'Sensor stress physics peak')
            require(row['strict_lift'] is (row['strict_longest'] >= 10), 'Sensor stress strict lift')
            require(row['force_violation'] is (row['peak_N'] > 8.0), 'Sensor stress force violation')
            require(row['joint_success'] is (row['strict_longest'] >= 10 and row['peak_N'] <= 8.0), 'Sensor stress primary definition')
            require(re.fullmatch('[0-9a-f]{64}', row['trajectory_sha256']) is not None, 'Sensor trajectory hash')
        total += len(rows)
    report = read_json(base/'sensor_stress_analysis/analysis.json')
    require(report['status'] == 'COMPLETE' and report['primary_metric'] == 'joint_success' and report['protocol_sha256'] == protocol_sha, 'Sensor stress complete analysis')
    for name in ('group_stats', 'contrasts', 'seed_variation', 'policy_stats', 'paired_policy_changes', 'paired_change_seed_variation'):
        require((base/'sensor_stress_analysis'/f'{name}.csv').stat().st_size > 0, 'Sensor stress analysis table missing')
    return total


def validate_records(base, sanitized_archive=False):
    """Validate complete text records at a runtime or archived revision root."""
    protocol, methods = validate_protocol(base)
    audit = read_json(base/'dataset/audit.json')
    require(audit['episodes'] == 336 and audit['counts'] == {'train': 240, 'val': 48, 'calibration': 48}, 'Dataset audit counts')
    for key in ('all_prescribed_parameters_match', 'all_intervention_actions_are_2D', 'all_outcomes_retained', 'test_and_diagnostic_seeds_disjoint'):
        require(audit[key] is True, 'Dataset audit: '+key)
    manifest = read_json(base/'dataset/data_manifest.json')
    expected_collection = {r['episode_id']: r for r in protocol['collection']}
    require(len(manifest) == 336 and len({r['episode_id'] for r in manifest}) == 336, 'Dataset manifest completeness')
    for row in manifest:
        case = expected_collection[row['episode_id']]
        require(row['seed'] == case['seed'] and row['split'] == case['split'] and row['steps'] == 150, 'Dataset manifest mismatch')
    for variant in ('vision', 'visuotactile'):
        for seed in (0, 1, 2):
            name = f'{variant}_seed{seed}'
            summary = read_json(base/'world_models'/name/'summary.json')
            require(summary['variant'] == variant and summary['seed'] == seed, 'WM run identity')
            calibration = read_json(base/'world_models'/name/'calibration.json')
            require(calibration['available'] is True and calibration['calibration_episodes'] == 48, '48-episode calibration')
            require(calibration['alpha_joint'] == .1 and calibration['order_statistic_rank'] == 47 and not calibration.get('infinite_radius', False), 'Fixed finite-sample calibration')
            require(all(math.isfinite(calibration[key]) and calibration[key] >= 0 for key in ('peak_radius_N', 'height_radius_m')), 'Finite calibration radii')
            imagined = read_json(base/'rl'/name/'run_metadata.json')['arguments']
            require(imagined['seed'] == seed and imagined['bc_steps'] == 1000 and imagined['rl_steps'] == 1500 and imagined['task_reward'] == 'height', 'Imagined actor budget/reward')
            reactive = read_json(base/'reactive_runs'/name/'run_metadata.json')
            require(reactive['arguments']['seed'] == seed and reactive['arguments']['variant'] == variant, 'Reactive identity')
            require(reactive['completed_bc_updates'] == 2000 and reactive['completed_iql_updates'] == 10000, 'Reactive update budget')
            require(reactive['no_world_model'] is True and reactive['random_initialization'] is True, 'Independent reactive training')
    total = 0
    cases = {r['seed']: r for r in protocol['control_test']}
    protocol_sha = digest(base/'protocol.json')
    for method in methods:
        directory = base/'evaluations'/method['name']
        rows = json_rows(directory/'episodes.jsonl')
        require(len(rows) == len({r['seed'] for r in rows}) == 120, method['name']+': incomplete or duplicated environments')
        require({r['seed'] for r in rows} == set(cases), method['name']+': wrong environments')
        inputs = read_json(directory/'inputs.json')
        require(inputs['method'] == method and inputs['smoke'] is False and set(inputs['cases']) == set(cases), 'Evaluation input identity')
        require(inputs['protocol_sha256'] == protocol_sha, 'Evaluation protocol provenance')
        complete = read_json(directory/'complete.json')
        require(complete['episodes'] == 120 and complete['inputs'] == inputs, 'Evaluation completion marker')
        for row in rows:
            case = cases[row['seed']]
            require(row['method'] == method['name'] and row['group'] == method['group'] and row['domain'] == case['domain'], 'Evaluation row identity')
            require(row['geometry'] == case['params']['geometry'], 'Evaluation geometry')
            require(math.isfinite(row['peak_N']) and row['peak_N'] >= 0 and math.isfinite(row['max_height_m']), 'Finite control metrics')
            require(row['strict_lift'] is (row['strict_longest'] >= 10), 'Strict-lift definition')
            require(row['force_violation'] is (row['peak_N'] > 8.0), 'Force violation definition')
            require(row['joint_success'] is (row['strict_longest'] >= 10 and row['peak_N'] <= 8.0), 'Joint-success definition')
            require(abs(row['peak_N']-max(row[k] for k in ('setup_peak_N', 'intervention_peak_N', 'lower_peak_N'))) <= 1e-9, 'Full-episode peak partition')
            require(re.fullmatch('[0-9a-f]{64}', row['trajectory_sha256']) is not None, 'Trajectory hash missing')
        total += len(rows)
    analysis = read_json(base/'analysis/control_analysis.json')
    require(analysis['status'] == 'COMPLETE' and analysis['primary_metric'] == 'joint_success' and 'main' in analysis['datasets'], 'Complete control analysis')
    for suffix in ('group_stats', 'contrasts', 'seed_variation', 'policy_stats'):
        require((base/'analysis'/f'control_{suffix}.csv').stat().st_size > 0, 'Control analysis table missing')
    forecast = read_json(base/'forecast_analysis/analysis_metadata.json')
    require(forecast['complete'] is True and forecast['environment_episodes'] == 120 and forecast['environments_per_domain'] == 30, 'Forecast completeness')
    require((forecast['history'], forecast['forecast_horizon'], forecast['stride'], forecast['windows_per_episode']) == (3, 5, 5, 30), 'Forecast window protocol')
    require(forecast['test_used_for_training_selection_calibration'] is False, 'Forecast split separation')
    with (base/'forecast_analysis/episode_metrics.csv').open(encoding='utf-8', newline='') as stream:
        forecast_rows = list(csv.DictReader(stream))
    require(len(forecast_rows) == 1440, 'Four forecast families x three seeds x 120 environments')
    tuples = {(r['method'], int(r['training_seed']), int(r['environment_seed'])) for r in forecast_rows}
    expected_forecasters = {'vision_world_model', 'visuotactile_world_model', 'visuotactile_persistence', 'visuotactile_shuffled_touch'}
    require(len(tuples) == 1440 and {r['method'] for r in forecast_rows} == expected_forecasters, 'Forecast duplicate/missing cells')
    for row in forecast_rows:
        require(row['domain'] == 'test_'+cases[int(row['environment_seed'])]['domain'] and int(row['windows']) == 30, 'Forecast domain/window mismatch')
    for group in {r['method'] for r in forecast_rows}:
        for seed in (0, 1, 2):
            require({e for m, s, e in tuples if m == group and s == seed} == set(cases), 'Forecast paired environments')
    for name in ('forecast_analysis.json', 'aggregate_metrics.csv', 'model_metrics.csv', 'paired_differences.csv', 'locked_inputs.json'):
        require((base/'forecast_analysis'/name).stat().st_size > 0, 'Forecast analysis output missing: '+name)
    counter = read_json(base/'counterfactual/summary.json')
    require(counter['kind'] == 'held-out counterfactual action diagnostic', 'Counterfactual smoke is not full evidence')
    require((counter['environment_count'], counter['state_count'], counter['branch_count']) == (12, 36, 324), 'Counterfactual matrix')
    require(counter['environment_seeds'] == [r['seed'] for r in protocol['counterfactual']], 'Counterfactual environments')
    counter_rows = json_rows(base/'counterfactual/per_environment_metrics.jsonl')
    require(len(counter_rows) == 12 and {r['seed'] for r in counter_rows} == set(counter['environment_seeds']), 'Counterfactual per-environment records')
    require(all(r['prefix_steps'] == [44, 78, 112] for r in counter_rows), 'Counterfactual prefixes')
    run = read_json(base/'counterfactual/run_manifest.json')
    require(run['smoke'] is False and run['protocol_sha256'] == protocol_sha, 'Counterfactual provenance')
    require(run['model_variants'] == ['vision']*3+['visuotactile']*3 and run['model_seeds'] == [0, 1, 2]*2, 'All six counterfactual models')
    require(run['device'] == counter['device'] == 'cpu', 'Recorded counterfactual inference must be CPU')
    if not sanitized_archive:
        require(counter['run_manifest_sha256'] == digest(base/'counterfactual/run_manifest.json'), 'Executed counterfactual manifest hash')
    runtime = read_json(base/'counterfactual/runtime_provenance.json')
    require(runtime['device'] == 'cpu' and runtime['manifest_sha256'] == counter['run_manifest_sha256'], 'Counterfactual runtime record')
    require(runtime['runtime_provenance']['model_inference']['resolved_device'] == 'cpu', 'Counterfactual resolved model device')
    stress_total = validate_sensor_records(base)
    return {'collection_episodes': 336, 'policies': 34, 'control_executions': total,
            'distinct_control_environments': 120, 'forecast_metric_rows': 1440,
            'counterfactual_environments': 12, 'counterfactual_branches': 324,
            'sensor_stress_executions': stress_total, 'sensor_stress_distinct_environments': 30}


def verify(root=ROOT):
    index = read_json(root/'REVISION_V2_FILES_SHA256.json')
    require(index['format_version'] == 1, 'Unsupported index format')
    for name, expected in index['files'].items():
        path = root/name
        require(path.resolve().is_relative_to(root.resolve()), 'Unsafe indexed path')
        require(path.is_file() and digest(path) == expected, 'Hash mismatch: '+name)
        require(path.suffix.lower() not in RAW_SUFFIXES, 'Raw/binary artifact in package: '+name)
        source = path.read_text(encoding='utf-8-sig')
        require(HOST_PATH.search(source) is None, 'Host-specific path: '+name)
        require(SECRET.search(source) is None, 'Possible credential: '+name)
        require(not any(marker in source for marker in PRIVATE_MARKERS), 'Private host-relative marker: '+name)
        if path.suffix == '.py':
            ast.parse(source, filename=name)
    # Verify every predecessor indexed byte remains unchanged as well.
    original = read_json(root/'FILES_SHA256.json')
    require(digest(root/'FILES_SHA256.json') == index['predecessor_index_sha256'], 'Predecessor file index changed')
    for name, expected in original['files'].items():
        require(digest(root/name) == expected, 'Predecessor package changed: '+name)
    validate_protocol(root/'outputs/revision_v2')
    sensor = validate_sensor_protocol(root/'outputs/revision_v2')
    for name, expected in sensor['source_sha256'].items():
        require(digest(root/name) == expected, 'Portable sensor inherited source hash: '+name)
    expected_source = digest(root/'outputs/revision_v2/sim_adapter_v2.py')
    require(read_json(root/'outputs/revision_v2/protocol.json')['source_sha256'] == expected_source, 'Portable protocol adapter hash')
    record_files = {p.relative_to(root).as_posix() for p in (root/'recorded/outputs/revision_v2').rglob('*') if p.is_file()}
    require(record_files <= set(index['files']), 'Unindexed archived files')
    report = {'integrity': 'PASS', 'indexed_revision_files': len(index['files']), 'code_only': index['code_only']}
    if index['code_only']:
        require(not record_files, 'Code-only package unexpectedly includes archived records')
        report['note'] = 'Code review package only. Completed result records have not been packaged or verified.'
    else:
        # Recorded internal hashes describe original execution bytes. Path
        # sanitization changes exported manifest bytes, whose integrity is
        # independently checked by this package's files index above.
        counter = read_json(root/'recorded/outputs/revision_v2/counterfactual/summary.json')
        origin = index['original_to_export_provenance']['recorded/outputs/revision_v2/counterfactual/run_manifest.json']['original_record_sha256']
        require(counter['run_manifest_sha256'] == origin, 'Original counterfactual manifest provenance')
        report.update(validate_records(root/'recorded/outputs/revision_v2', sanitized_archive=True))
        report['note'] = 'Archived text verification only; raw arrays and checkpoints are regenerated by the runner.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))


if __name__ == '__main__':
    main()
