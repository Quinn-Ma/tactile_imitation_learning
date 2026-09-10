"""Frozen appendix sensor-error diagnostic; never modifies main evaluate.py.

--check-ready inspects final-model completion markers and hashes only.
--smoke --method force_feedback --out work/... runs one VALIDATION environment.
--run --method NAME executes the30 frozen ID stress cases after explicit go.
--analyze compares completed stress records with matched main clean ID records.

Errors are contact-data CNN residuals applied additively at an imposed20Hz
index rate, including zero-contact states. This is synthetic error transport,
not a calibrated GelSight sensor model, physical safety or sim-to-real evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PROTOCOL = OUT / 'sensor_stress_protocol.json'
FROZEN_PROTOCOL_SHA256 = '1f1751ca532120e20cb222653a7db5369ff68622c882c27525b1bc56796797a5'
GROUPS = ('WM_visuotactile', 'IQL_visuotactile', 'model_guard', 'force_feedback', 'reactive_guard')
SEEDED_GROUPS = GROUPS[:3]
# Portable packaging: execution requires an explicit --gpu-python.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def residual_hash(values):
    packed = b''.join(struct.pack('<f', float(value)) for row in values for value in row)
    return hashlib.sha256(packed).hexdigest()


def load_protocol(verify_sources=True):
    require(sha(PROTOCOL) == FROZEN_PROTOCOL_SHA256, 'Frozen sensor protocol changed; refuse execution')
    plan = read_json(PROTOCOL)
    require(len(plan['cases']) == 30 and len(plan['methods']) == 11, 'Incorrect frozen matrix size')
    require({m['group'] for m in plan['methods']} == set(GROUPS), 'Incorrect sensor method groups')
    require(len({c['seed'] for c in plan['cases']}) == 30, 'Duplicate sensor environment seeds')
    for group in GROUPS:
        jobs = [m for m in plan['methods'] if m['group'] == group]
        if group in SEEDED_GROUPS:
            require(sorted(m.get('seed', -1) for m in jobs) == [0, 1, 2], f'Incomplete seeds for {group}')
        else:
            require(len(jobs) == 1, f'Incorrect singleton group {group}')
    for case in plan['cases']:
        values = case['normal_residual_N']
        require(case['domain'] == 'id' and len(values) == 93 and all(len(row) == 2 for row in values),
                'Invalid sensor case or residual shape')
        require(all(math.isfinite(v) for row in values for v in row), 'Nonfinite empirical residual')
        require(residual_hash(values) == case['residual_float32_sha256'], 'Changed residual block values')
    if verify_sources:
        require(sha(OUT / 'protocol.json') == plan['main_protocol_sha256'], 'Main protocol changed')
        require(sha(OUT / 'methods.json') == plan['main_methods_sha256'], 'Main method matrix changed')
        # The frozen protocol and float32 block hashes authenticate embedded residuals.
        # The optional original prediction archive is not redistributed.
        residual_source = ROOT / plan['source']['path']
        if residual_source.is_file():
            require(sha(residual_source) == plan['source']['sha256'], 'Public residual source changed')
        for path, expected in plan['source_sha256'].items():
            require(sha(ROOT / path) == expected, f'Frozen inherited source changed: {path}')
    return plan


def ready_check(plan):
    """Final artifacts, not interim best checkpoints; never opens test outcomes."""
    rows = []
    for method in plan['methods']:
        missing, hashes, markers = [], {}, {}
        for key in ('model', 'actor', 'checkpoint'):
            if key not in method:
                continue
            path = ROOT / method[key]
            marker = path.parent / ('summary.json' if key == 'model' else 'run_metadata.json')
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(f'missing {key}: {method[key]}')
            elif not marker.is_file():
                missing.append(f'final completion marker missing: {marker.relative_to(ROOT)}')
            else:
                hashes[key] = sha(path)
                markers[key] = sha(marker)
                if key == 'actor':
                    metadata = read_json(marker)
                    if 'model_sha256' in metadata and method.get('model'):
                        if metadata['model_sha256'] != sha(ROOT / method['model']):
                            missing.append('actor was trained with a different world-model checkpoint')
                    arguments = metadata.get('arguments', {})
                    if int(arguments.get('rl_steps', -1)) != 1500:
                        missing.append('actor metadata does not confirm fixed1500 RL updates')
                elif key == 'checkpoint':
                    metadata = read_json(marker)
                    if metadata.get('completed_iql_updates') != 10000:
                        missing.append('reactive metadata does not confirm fixed10000 IQL updates')
                    recorded_hash = metadata.get(path.name + '_sha256')
                    if recorded_hash != hashes[key]:
                        missing.append('reactive checkpoint hash differs from final metadata')
        rows.append(dict(name=method['name'], group=method['group'], ready=not missing,
                         missing=missing, model_inputs=hashes, completion_marker_hashes=markers))
    return dict(ready=all(row['ready'] for row in rows), expected_methods=11, methods=rows,
                protocol_sha256=sha(PROTOCOL), outcomes_read=False)


def corrupt_observation(obs, t, block, np):
    """Copy the controller input; never mutate the simulator observation/state."""
    supplied = dict(obs)
    supplied['tactile'] = obs['tactile'].copy()
    if 42 <= t < 135:
        supplied['tactile'][[0, 3]] = np.maximum(
            0., supplied['tactile'][[0, 3]] + block[t - 42])
    return supplied


def execute_episode(env, method, case, bridge, np, longest_run):
    block = np.asarray(case['normal_residual_N'], dtype=np.float32)
    require(block.shape == (93, 2), 'Incorrect block dimensions')
    obs = env.reset(case['seed'], params=case['params'])
    if bridge:
        bridge.reset()
    states, supplied_frames = [obs], []
    actions, rewards, latencies = [], [], []
    guard_count = fallback_count = 0
    for t in range(150):
        # Call the original script exactly once, retaining common nuisance RNG.
        action = env.script_action()
        supplied = corrupt_observation(obs, t, block, np)
        supplied_frames.append(supplied['tactile'].copy())
        if bridge and 42 <= t < 135:
            predicted, reply, elapsed = bridge.action(supplied, t >= 44)
            if t >= 44:
                latencies.append(elapsed)
                guard_count += int(reply.get('guard_intervened', False))
                fallback_count += int(reply.get('fallback_to_reactive', False))
        if 44 <= t < 135:
            action[[0, 1, 3, 4, 5]] = 0.
            if bridge:
                action[[2, 6]] = predicted[[2, 6]]
            elif method['kind'] == 'force_feedback':
                action[6] = np.clip(.12 * (5. - float(supplied['tactile'][[0, 3]].mean())), -.25, .35)
            else:
                require(method['kind'] == 'script', 'Missing policy bridge')
        obs, reward, _, info = env.step(action)
        require(info['substeps'] == 25, 'Unexpected physics substep count')
        states.append(obs)
        actions.append(action.copy())
        rewards.append([reward])
    supplied_frames.append(obs['tactile'].copy())
    arrays = {key: np.stack([state[key] for state in states]) for key in states[0]}
    arrays['observed_tactile'] = np.stack(supplied_frames)
    arrays['observed_normals_N'] = arrays['observed_tactile'][:, [0, 3]].copy()
    arrays['sensor_residual_block_N'] = block.copy()
    arrays['sensor_residual_N'] = np.zeros((151, 2), dtype=np.float32)
    arrays['sensor_residual_N'][42:135] = block
    arrays['sensor_noise_active'] = np.zeros(151, dtype=bool)
    arrays['sensor_noise_active'][42:135] = True
    raw_noisy = arrays['tactile'][42:135][:, [0, 3]] + block
    arrays['sensor_clipped_at_zero'] = np.zeros((151, 2), dtype=bool)
    arrays['sensor_clipped_at_zero'][42:135] = raw_noisy < 0.
    arrays.update(action=np.stack(actions), reward=np.asarray(rewards, np.float32), seed=np.array(case['seed']),
        episode_id=np.array(case['episode_id']), split=np.array('sensor_stress_id'),
        params_json=np.array(json.dumps(env.params, sort_keys=True)), dt=np.array(.05),
        sensor_draws_json=np.array(json.dumps(case['sensor_draws'], sort_keys=True)),
        sensor_protocol_sha256=np.array(sha(PROTOCOL)),
        sensor_residual_float32_sha256=np.array(case['residual_float32_sha256']))
    require(np.max(np.abs(arrays['action'][44:135][:, [0, 1, 3, 4, 5]])) == 0., 'Action-subspace violation')
    require(np.array_equal(arrays['observed_tactile'][:, [1, 2, 4, 5]], arrays['tactile'][:, [1, 2, 4, 5]]),
            'Shear channels changed')
    require(np.array_equal(arrays['observed_normals_N'][42:135], np.maximum(0., raw_noisy)),
            'Saved observations do not match the injected residual')
    require(np.array_equal(arrays['observed_tactile'][:42], arrays['tactile'][:42]) and
            np.array_equal(arrays['observed_tactile'][135:], arrays['tactile'][135:]),
            'Observation corruption outside declared interval')
    strict_longest = longest_run(arrays['height'][1:136, 0] >= .10)
    peak = float(arrays['substep_normal_peak'].max())
    true_normals = arrays['tactile'][42:135][:, [0, 3]]
    false_positive = (true_normals == 0.) & (arrays['observed_normals_N'][42:135] > 0.)
    row = dict(seed=case['seed'], domain='id', geometry=env.params['geometry'], method=method['name'],
        group=method['group'], strict_lift=bool(strict_longest >= 10),
        joint_success=bool(strict_longest >= 10 and peak <= 8.), force_violation=bool(peak > 8.),
        peak_N=peak, max_height_m=float(arrays['height'].max()), strict_longest=strict_longest,
        setup_peak_N=float(arrays['substep_normal_peak'][:45].max()),
        intervention_peak_N=float(arrays['substep_normal_peak'][45:136].max()),
        lower_peak_N=float(arrays['substep_normal_peak'][136:].max()),
        bilateral_contact_at_takeover=bool(arrays['contact'][44].all()),
        native_success=bool(longest_run(arrays['success'][1:136, 0]) >= 10),
        native_return=float(np.sum(rewards)), guard_interventions=guard_count, model_fallbacks=fallback_count,
        policy_roundtrip_ms_mean=float(np.mean(latencies) * 1000) if latencies else None,
        policy_roundtrip_ms_p95=float(np.percentile(latencies, 95) * 1000) if latencies else None,
        timing_caveat='Concurrent local execution; IPC included; not isolated latency',
        params=env.params, sensor_protocol_sha256=sha(PROTOCOL),
        sensor_residual_float32_sha256=case['residual_float32_sha256'], sensor_draws=case['sensor_draws'],
        normal_observation_clip_count=int(arrays['sensor_clipped_at_zero'].sum()),
        normal_observation_clip_fraction=float(arrays['sensor_clipped_at_zero'][42:135].mean()),
        synthetic_no_contact_positive_count=int(false_positive.sum()),
        metrics_source='Unmodified ground-truth simulator state/substep forces, never noisy observations')
    return arrays, row


def evaluate(args, plan):
    from evaluate import Bridge, SimAdapterV2, np, longest_run
    method = next((m for m in plan['methods'] if m['name'] == args.method), None)
    require(method is not None, 'Method not included in frozen sensor stress protocol')
    readiness = ready_check(plan)
    selected_ready = next(row for row in readiness['methods'] if row['name'] == method['name'])
    require(selected_ready['ready'], 'Selected final policy is not ready: ' + '; '.join(selected_ready['missing']))
    require(args.smoke or readiness['ready'], 'All11 final methods must be ready before primary stress execution')
    cases = plan['cases']
    if args.smoke:
        validation_case = next(c for c in read_json(OUT / 'protocol.json')['collection'] if c['split'] == 'val')
        validation_case = copy.deepcopy(validation_case)
        for key in ('normal_residual_N', 'sensor_draws', 'residual_float32_sha256'):
            validation_case[key] = copy.deepcopy(cases[0][key])
        cases = [validation_case]
    destination = Path(args.out).resolve() if args.out else OUT / 'sensor_stress_evaluations' / method['name']
    if args.smoke:
        require(args.out is not None, 'Validation smoke requires explicit separate --out')
        require(destination.is_relative_to((ROOT / 'work').resolve()), 'Smoke output must be inside workspace work/')
    destination.mkdir(parents=True, exist_ok=True)
    inputs = dict(method=method, smoke=bool(args.smoke), sensor_protocol_sha256=sha(PROTOCOL),
        protocol_sha256=plan['main_protocol_sha256'], main_evaluator_sha256=plan['source_sha256']['outputs/revision_v2/evaluate.py'],
        sensor_evaluator_sha256=sha(Path(__file__)), adapter_sha256=sha(OUT / 'sim_adapter_v2.py'),
        original_adapter_sha256=sha(ROOT / 'outputs/simulation/sim_adapter.py'),
        cases=[c['seed'] for c in cases], model_inputs=selected_ready['model_inputs'],
        completion_marker_hashes=selected_ready['completion_marker_hashes'], residual_source_sha256=plan['source']['sha256'])
    inputs_path = destination / 'inputs.json'
    if inputs_path.exists():
        require(read_json(inputs_path) == inputs, 'Changed run inputs; refuse mixed resume')
    else:
        inputs_path.write_text(json.dumps(inputs, indent=2), encoding='utf-8')
    records = destination / 'episodes.jsonl'
    rows = []
    if records.exists():
        require(args.resume, 'Existing episodes require --resume')
        rows = [json.loads(line) for line in records.read_text(encoding='utf-8').splitlines() if line.strip()]
        require(len(rows) <= len(cases), 'Resume exceeds planned case count')
        for row, case in zip(rows, cases):
            require(row['seed'] == case['seed'] and row['method'] == method['name'], 'Resume case mismatch')
            require(row['sensor_residual_float32_sha256'] == case['residual_float32_sha256'], 'Resume noise changed')
            require(sha(destination / f'episode_{case["seed"]}.npz') == row['trajectory_sha256'], 'Resume trajectory changed')
    if len(rows) == len(cases):
        return dict(completed=len(rows), already_complete=True)
    bridge = Bridge(method, args.gpu_python, destination) if method['kind'] in ('world_model', 'reactive', 'adaptive') else None
    env = None
    start = time.perf_counter()
    try:
        env = SimAdapterV2(render=True)
        for index, case in list(enumerate(cases))[len(rows):]:
            arrays, row = execute_episode(env, method, case, bridge, np, longest_run)
            path = destination / f'episode_{case["seed"]}.npz'
            np.savez_compressed(path, **arrays)
            row['trajectory_sha256'] = sha(path)
            row['smoke'] = bool(args.smoke)
            with records.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            rows.append(row)
            if (index + 1) % 10 == 0 or index + 1 == len(cases):
                print(json.dumps(dict(method=method['name'], completed=index+1, total=len(cases),
                    smoke=bool(args.smoke), elapsed_seconds=round(time.perf_counter()-start, 1))), flush=True)
    finally:
        if env is not None:
            env.close()
        if bridge:
            bridge.close()
    summary = dict(completed=len(rows), inputs=inputs, bridge_ready=bridge.ready if bridge else None,
                   elapsed_seconds=time.perf_counter()-start, claim_scope=plan['scope'])
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def analyze(args, plan):
    import analyze_control as analysis
    expected = {('id', c['seed']): c for c in plan['cases']}
    matrix, jobs, progress, missing = {}, [], [], []
    params_by_env = {}
    input_sources = []
    for method in plan['methods']:
        condition_inputs = {}
        for condition in ('clean', 'stress'):
            folder = (OUT / 'evaluations' if condition == 'clean' else OUT / 'sensor_stress_evaluations') / method['name']
            path = folder / 'episodes.jsonl'
            raw_rows, in_flight = analysis.read_jsonl(path, args.allow_partial)
            job = dict(method, name=f'{condition}/{method["name"]}', group=f'{condition}:{method["group"]}')
            jobs.append(job)
            indexed = {}
            for row in raw_rows:
                if condition == 'clean' and row.get('domain') != 'id':
                    continue
                label = f'{condition}/{method["name"]} seed={row.get("seed")}'
                key = (row.get('domain'), row.get('seed'))
                require(key in expected and key not in indexed, f'{label}: unexpected/duplicate environment')
                require(row.get('method') == method['name'] and row.get('group') == method['group'], f'{label}: method mismatch')
                require(row.get('geometry') == expected[key]['params']['geometry'], f'{label}: geometry mismatch')
                analysis.validate_prescribed_params(row['params'], expected[key]['params'], label)
                analysis.validate_metric_row(row, analysis.METRICS, label)
                fingerprint = analysis.canonical(row['params'])
                if key in params_by_env:
                    require(params_by_env[key] == fingerprint, f'{label}: paired parameters differ')
                params_by_env[key] = fingerprint
                if condition == 'stress':
                    require(not row.get('smoke'), f'{label}: validation smoke cannot enter results')
                    require(row.get('sensor_protocol_sha256') == sha(PROTOCOL), f'{label}: protocol mismatch')
                    require(row.get('sensor_residual_float32_sha256') == expected[key]['residual_float32_sha256'],
                            f'{label}: residual assignment mismatch')
                    require(row.get('sensor_draws') == expected[key]['sensor_draws'], f'{label}: draw provenance mismatch')
                trajectory = folder / f'episode_{row["seed"]}.npz'
                require(trajectory.is_file() and sha(trajectory) == row.get('trajectory_sha256'),
                        f'{label}: missing or modified saved trajectory')
                indexed[key] = row
            absent = sorted(set(expected) - set(indexed))
            if absent or in_flight:
                missing.append(f'{job["name"]}: {len(absent)} missing environments; in_flight={in_flight}')
            matrix[job['name']] = indexed
            progress.append(dict(method=method['name'], group=method['group'], condition=condition,
                                 completed=len(indexed), expected=30, complete=not absent and not in_flight))
            inputs_path = folder / 'inputs.json'
            if inputs_path.exists():
                inputs = read_json(inputs_path)
                require(inputs.get('method') == method and not inputs.get('smoke'), f'{job["name"]}: input identity mismatch')
                require(inputs.get('protocol_sha256') == plan['main_protocol_sha256'], f'{job["name"]}: environment plan changed')
                if condition == 'stress':
                    require(inputs.get('sensor_protocol_sha256') == sha(PROTOCOL), f'{job["name"]}: sensor input protocol mismatch')
                    require(inputs.get('sensor_evaluator_sha256') == sha(Path(__file__)), f'{job["name"]}: stress runner source changed')
                    require(inputs.get('residual_source_sha256') == plan['source']['sha256'], f'{job["name"]}: residual source mismatch')
                condition_inputs[condition] = inputs
                input_sources.append(dict(condition=condition, method=method['name'],
                    episodes_path=str(path), episodes_sha256=sha(path) if path.exists() else None,
                    inputs_sha256=sha(inputs_path)))
            elif indexed:
                raise ValueError(f'{job["name"]}: missing input provenance')
        if len(condition_inputs) == 2:
            for key in ('model_inputs', 'adapter_sha256', 'original_adapter_sha256'):
                require(condition_inputs['clean'].get(key) == condition_inputs['stress'].get(key),
                        f'{method["name"]}: clean/stress {key} differ')
            require(condition_inputs['clean'].get('evaluator_sha256') == plan['source_sha256']['outputs/revision_v2/evaluate.py'],
                    f'{method["name"]}: clean evaluator differs from frozen source')
    destination = Path(args.out).resolve() if args.out else OUT / 'sensor_stress_analysis'
    if missing:
        require(args.allow_partial, 'Incomplete fixed sensor matrix:\n' + '\n'.join(missing))
        destination.mkdir(parents=True, exist_ok=True)
        report = dict(status='INCOMPLETE_PROGRESS_ONLY', analysis_performed=False, missing=missing, progress=progress)
        (destination / 'progress.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        analysis.write_csv(destination / 'progress.csv', progress)
        return report
    contrasts = tuple((f'stress:{group}', f'clean:{group}') for group in GROUPS)
    results = analysis.analyze_matrix(jobs, matrix, expected, ('id',), analysis.METRICS, contrasts,
                                     resamples=2000, seed=20401010)
    # The single-domain pooled result is identical; retain only explicitly ID rows.
    for key in ('group_stats', 'contrasts', 'seed_variation', 'policy_stats'):
        results[key] = [row for row in results[key] if row['domain'] == 'id']
    paired_policies, change_variation = [], []
    for group in GROUPS:
        methods = [m for m in plan['methods'] if m['group'] == group]
        for metric in analysis.METRICS:
            changes = []
            for method in methods:
                change = statistics.fmean(matrix[f'stress/{method["name"]}'][key][metric] -
                                          matrix[f'clean/{method["name"]}'][key][metric] for key in expected)
                changes.append(change)
                paired_policies.append(dict(group=group, method=method['name'], training_seed=method.get('seed'),
                                            metric=metric, stress_minus_clean=change, n_environments=30))
            change_variation.append(dict(group=group, metric=metric, n_policies=len(methods),
                mean_change=statistics.fmean(changes), sample_sd=statistics.stdev(changes) if len(changes)>1 else None,
                minimum=min(changes), maximum=max(changes)))
    results.update(paired_policy_changes=paired_policies, paired_change_seed_variation=change_variation)
    report = dict(status='COMPLETE', primary_metric='joint_success', claim_scope=plan['scope'],
        protocol_sha256=sha(PROTOCOL), source_sha256=sha(Path(__file__)), limitations=plan['limitations'],
        input_sources=input_sources, results=results)
    destination.mkdir(parents=True, exist_ok=True)
    for key in ('group_stats', 'contrasts', 'seed_variation', 'policy_stats',
                'paired_policy_changes', 'paired_change_seed_variation'):
        analysis.write_csv(destination / f'{key}.csv', results[key])
    (destination / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (destination / 'README.md').write_text(
        'Appendix diagnostic: empirical residual-block perturbation of simulated normal-force observations. '
        'The30 main ID environments and11 fixed methods are paired between clean and stress conditions. '
        'All outcome metrics use true physics, including every full-episode substep peak. '
        'Three trained policies are averaged within environment before2,000 paired environment-bootstrap '
        'replicates (seed20401010); pointwise95% percentile intervals have no multiplicity correction. '
        'Training-seed variation and paired policy changes are separate tables. '
        'The contrast sign is stress minus clean; rates also have percentage-point differences. '
        'Intervals are conditional on the fixed policies and reused residual bank.\n\n'
        'Source errors come from held-out sphere trajectories of the prespecified CNN seed17. '
        'Source normal labels span approximately0.073–1.888N, below the8N simulator budget. '
        'Two finger blocks are independently drawn from one-sensor data, and one source frame per20Hz '
        'simulator observation is an imposed index mapping with no identified real sensor rate. '
        'Contact-derived errors are also added at zero true force; resulting false positives are synthetic. '
        'This is a normal-channel sensitivity study, not validated sensor transfer, sim-to-real, hardware '
        'safety, identified noise at8N, or optical tactile policy deployment. No model or threshold is tuned '
        'from these results.\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check-ready', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--analyze', action='store_true')
    parser.add_argument('--method')
    parser.add_argument('--out')
    parser.add_argument('--gpu-python', help='Required for --run or --smoke; provide your GPU Python executable')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    if (args.run or args.smoke) and not args.gpu_python:
        parser.error('--run and --smoke require --gpu-python')
    plan = load_protocol()
    if args.check_ready:
        result = ready_check(plan)
    elif args.analyze:
        result = analyze(args, plan)
    else:
        require(args.method is not None, 'Execution requires --method')
        result = evaluate(args, plan)
    if args.check_ready:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({'status': result.get('status', 'COMPLETE'), 'out': args.out,
                          'smoke': args.smoke, 'analysis': args.analyze}), flush=True)
    return result


if __name__ == '__main__':
    main()
