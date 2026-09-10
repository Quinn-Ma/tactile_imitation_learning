"""Train independent end-to-end reactive BC, then model-free offline IQL.

Only explicitly listed train and val episodes are opened. Calibration and test
records are neither loaded nor used for normalization, bounds, or selection.
BC uses minimum validation normalized-action MSE; IQL uses the fixed final
update, with validation diagnostics only. No simulator is called by this file.

IQL source: Kostrikov, Nair, Levine, https://arxiv.org/abs/2110.06169;
official objectives checked against https://github.com/ikostrikov/implicit_q_learning
(critic.py / actor.py). This independently implemented pixel adaptation does
not claim reproduction of the official benchmark. No external code is copied.
"""
import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

# Process-local requirement for deterministic CUDA matmul; no global changes.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

from reactive import (ReactiveActor, ReactiveConfig, ReactiveTwinQ, ReactiveValue,
                      expectile_loss, load_policy, observed_reward, parameter_count,
                      policy_payload, td_target)


def canonical_split(value):
    value = str(value).lower()
    return 'val' if value in ('val', 'valid', 'validation') else value


def manifest_rows(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(obj, dict) and isinstance(obj.get('episodes'), list):
        obj = obj['episodes']
    if isinstance(obj, dict) and isinstance(obj.get('splits'), dict):
        obj = obj['splits']
    if isinstance(obj, dict):
        rows = []
        for split, entries in obj.items():
            if canonical_split(split) not in ('train', 'val', 'calibration', 'test', 'id', 'ood', 'test_id', 'test_ood'):
                continue
            if isinstance(entries, dict):
                entries = entries.get('episodes', [])
            for entry in entries:
                row = dict(entry) if isinstance(entry, dict) else {'path': entry}
                row['split'] = split
                rows.append(row)
    elif isinstance(obj, list):
        rows = [dict(row) for row in obj]
    else:
        raise ValueError('Manifest must explicitly enumerate episode paths by split')
    for row in rows:
        if 'split' not in row:
            raise ValueError('Every manifest row must name its split')
        row['split'] = canonical_split(row['split'])
        row['path'] = row.get('path', row.get('file'))
        if not isinstance(row['path'], str) or not row['path']:
            raise ValueError('Every manifest row must name its episode path')
    return rows


def resolve_episode(name, data_dir, manifest_dir):
    """Resolve an explicit name; never glob a directory or inspect other NPZs."""
    data_dir, manifest_dir, name = Path(data_dir).resolve(), Path(manifest_dir).resolve(), Path(name)
    candidates = [name] if name.is_absolute() else [data_dir/name, manifest_dir/name, Path.cwd()/name, data_dir/name.name]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            if not candidate.is_relative_to(data_dir):
                raise ValueError(f'Explicit episode is outside --data: {candidate}')
            return candidate
    raise FileNotFoundError(f'Explicit episode not found: {name}')


def _scalar(value):
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def load_training_validation(data_dir, split_file, expected_train=80, expected_val=16,
                             start_step=44, control_end=135, allow_action_projection=False):
    """Open only train/val files and reject ambiguous grouping and labels.

The default also rejects recorded transitions with other action coordinates:
their causal action cannot be represented by a two-output Q function. An
explicit projection override exists only for separately declared diagnostics.
"""
    rows = manifest_rows(split_file)
    selected = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val')}
    for split, expected in [('train', expected_train), ('val', expected_val)]:
        if not selected[split] or (expected is not None and len(selected[split]) != expected):
            raise ValueError(f'{split}: expected {expected} episodes, found {len(selected[split])}')
    # Path/ID conflicts in the manifest are rejected without opening held-out files.
    seen_names, seen_manifest_ids = set(), set()
    for row in rows:
        normalized_name = os.path.normcase(os.path.normpath(row['path']))
        if normalized_name in seen_names:
            raise ValueError(f'Duplicate manifest path: {row["path"]}')
        seen_names.add(normalized_name)
        if row.get('episode_id') is not None:
            eid = str(row['episode_id'])
            if eid in seen_manifest_ids:
                raise ValueError(f'Duplicate manifest episode_id: {eid}')
            seen_manifest_ids.add(eid)
    result, provenance, seen_paths, seen_ids, seen_hashes = {}, {}, set(), set(), set()
    for split in ('train', 'val'):
        result[split], provenance[split] = [], []
        for row in selected[split]:
            path = resolve_episode(row['path'], data_dir, Path(split_file).parent)
            if path in seen_paths:
                raise ValueError(f'An episode appears in multiple groups: {path}')
            seen_paths.add(path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha in seen_hashes:
                raise ValueError('Duplicate episode bytes across manifest entries')
            seen_hashes.add(sha)
            with np.load(path, allow_pickle=False) as source:
                required = ('rgb', 'tactile', 'proprio', 'action', 'height', 'substep_normal_peak', 'split', 'episode_id')
                if any(k not in source for k in required):
                    raise ValueError(f'{path}: missing required observation, peak, or split fields')
                embedded_split, eid = canonical_split(_scalar(source['split'])), _scalar(source['episode_id'])
                if embedded_split != split:
                    raise ValueError(f'{path}: manifest split {split} disagrees with embedded {embedded_split}')
                if row.get('episode_id') is not None and str(row['episode_id']) != eid:
                    raise ValueError(f'{path}: manifest and embedded episode IDs disagree')
                if eid in seen_ids:
                    raise ValueError(f'Duplicate episode identity: {eid}')
                seen_ids.add(eid)
                ep = {k: source[k].copy() for k in required[:6]}
                if 'done' in source:
                    ep['done'] = source['done'].copy()
            n = len(ep['action'])
            if ep['rgb'].shape != (n+1, 64, 64, 3) or ep['rgb'].dtype != np.uint8:
                raise ValueError(f'{path}: require RGB uint8 [T+1,64,64,3]')
            for key, dim, length in [('tactile', 6, n+1), ('proprio', 9, n+1),
                                     ('height', 1, n+1), ('substep_normal_peak', 2, n+1), ('action', 7, n)]:
                ep[key] = np.asarray(ep[key], dtype=np.float32).reshape(length, dim)
                if not np.isfinite(ep[key]).all():
                    raise ValueError(f'{path}: nonfinite {key}')
            if np.any(ep['substep_normal_peak'] < -1e-6) or np.any(np.abs(ep['action']) > 1.000001):
                raise ValueError(f'{path}: negative normal peak or out-of-range action')
            terminal = min(n, control_end)
            if 'done' in ep:
                done = np.asarray(ep['done'], dtype=bool).reshape(-1)
                if len(done) == n+1:
                    done = done[1:]
                if len(done) != n:
                    raise ValueError('done must be [T] action termination or [T+1] state termination')
                if done.any():
                    terminal = min(terminal, int(np.flatnonzero(done)[0])+1)
            if terminal <= start_step:
                raise ValueError(f'{path}: no intervention transition before termination')
            other_max = float(np.abs(ep['action'][start_step:terminal][:, [0, 1, 3, 4, 5]]).max())
            if other_max > 1e-6 and not allow_action_projection:
                raise ValueError(f'{path}: noncontrolled action max {other_max:.6g}; cannot fit a 2D causal Q transition')
            ep.update(terminal=terminal, episode_id=eid)
            result[split].append(ep)
            provenance[split].append({'path': str(path), 'episode_id': eid, 'split': split,
                                      'sha256': sha, 'steps': n, 'terminal_observation': terminal,
                                      'max_noncontrolled_action': other_max})
    return result, provenance


def training_normalization(episodes):
    norm = {}
    for key, floor in [('tactile', .05), ('proprio', .01)]:
        arrays = [ep[key].astype(np.float64) for ep in episodes]
        count = sum(len(a) for a in arrays)
        mean = sum(a.sum(0) for a in arrays)/count
        variance = sum(np.square(a).sum(0) for a in arrays)/count-mean*mean
        norm[key+'_mean'] = mean.tolist()
        norm[key+'_std'] = np.sqrt(np.maximum(variance, 0)).clip(floor).tolist()
    return norm


class ObservationReplay:
    """Raw recorded observation bank, not a cached learned representation.

Current and next histories never cross episode boundaries. bootstrap=0 at
the first terminal or the intervention boundary, even if lowering continues.
"""
    def __init__(self, episodes, device='cpu', history=3, start_step=44, stride=1, reward_args=None):
        if stride < 1 or history < 1 or start_step < history-1:
            raise ValueError('Invalid stride/history/intervention start')
        rgb, touch, proprio, history_rows, next_rows, actions, heights, peaks, masks = ([] for _ in range(9))
        offset = 0
        for ep in episodes:
            n = len(ep['action'])
            rgb.append(ep['rgb'].transpose(0, 3, 1, 2))
            touch.append(ep['tactile'])
            proprio.append(ep['proprio'])
            times = np.arange(start_step, ep['terminal'], stride)
            lag = np.arange(-history+1, 1)
            history_rows.append(offset+np.maximum(times[:, None]+lag, 0))
            next_rows.append(offset+np.maximum(times[:, None]+1+lag, 0))
            actions.append(ep['action'][times][:, [2, 6]])
            heights.append(ep['height'][times+1])
            peaks.append(ep['substep_normal_peak'][times+1])
            masks.append((times+1 < ep['terminal']).astype(np.float32))
            offset += n+1
        self.device = torch.device(device)
        self.rgb = torch.as_tensor(np.concatenate(rgb), device=device)
        self.tactile = torch.as_tensor(np.concatenate(touch), device=device)
        self.proprio = torch.as_tensor(np.concatenate(proprio), device=device)
        self.current = torch.as_tensor(np.concatenate(history_rows), device=device)
        self.next = torch.as_tensor(np.concatenate(next_rows), device=device)
        self.action = torch.as_tensor(np.concatenate(actions), device=device)
        self.bootstrap = torch.as_tensor(np.concatenate(masks), device=device)
        next_height = torch.as_tensor(np.concatenate(heights), device=device)
        next_peak = torch.as_tensor(np.concatenate(peaks), device=device)
        self.reward = observed_reward(next_height, next_peak, self.action, **(reward_args or {}))

    def __len__(self):
        return len(self.action)

    def observations(self, indices, next_state=False):
        rows = self.next[indices] if next_state else self.current[indices]
        return (self.rgb[rows].flatten(1, 2), self.tactile[rows], self.proprio[rows])

    def batch(self, indices):
        return {'obs': self.observations(indices), 'next_obs': self.observations(indices, True),
                'action': self.action[indices], 'reward': self.reward[indices],
                'bootstrap': self.bootstrap[indices]}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def validation_metrics(actor, replay, batch_size):
    actor.eval()
    mse, nll, n = 0., 0., len(replay)
    for start in range(0, n, batch_size):
        indices = torch.arange(start, min(start+batch_size, n), device=replay.device)
        obs, action = replay.observations(indices), replay.action[indices]
        prediction = actor(*obs)
        mse += float((actor.normalize_action(prediction)-actor.normalize_action(action)).square().mean(-1).sum())
        nll += float(-actor.log_prob(*obs, action).sum())
    return {'val_normalized_action_mse': mse/n, 'val_action_nll': nll/n}


def save_checkpoint(payload, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def optimizer_step(loss, optimizer, module, clip=10.):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite objective; stop instead of selecting an invalid policy')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(module.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return float(gradient_norm.detach())


def iql_update(actor, critic, value, target_critic, optimizers, batch, discount=.95,
               expectile=.7, beta=3., weight_cap=100., ema=.005, gradient_clip=10.):
    obs, next_obs, actions = batch['obs'], batch['next_obs'], batch['action']
    unit_action = actor.normalize_action(actions).detach()
    with torch.no_grad():
        tq1, tq2 = target_critic(*obs, unit_action)
        target_q = torch.minimum(tq1, tq2)
    current_v = value(*obs)
    value_loss = expectile_loss(target_q-current_v, expectile)
    optimizer_step(value_loss, optimizers['value'], value, gradient_clip)
    with torch.no_grad():
        advantage = target_q-value(*obs)
        weights = torch.exp((beta*advantage).clamp(max=math.log(weight_cap)))
        target = td_target(batch['reward'], batch['bootstrap'], value(*next_obs), discount)
    actor_loss = -(weights*actor.log_prob(*obs, actions)).mean()
    optimizer_step(actor_loss, optimizers['actor'], actor, gradient_clip)
    q1, q2 = critic(*obs, unit_action)
    critic_loss = ((q1-target).square()+(q2-target).square()).mean()
    optimizer_step(critic_loss, optimizers['critic'], critic, gradient_clip)
    with torch.no_grad():
        for dest, source in zip(target_critic.parameters(), critic.parameters()):
            dest.lerp_(source, ema)
    return {'actor_loss': float(actor_loss.detach()), 'value_loss': float(value_loss.detach()),
            'critic_loss': float(critic_loss.detach()), 'mean_advantage': float(advantage.mean()),
            'mean_advantage_weight': float(weights.mean()),
            'max_advantage_weight': float(weights.max()), 'mean_td_target': float(target.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=['vision', 'visuotactile'], default='visuotactile')
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--expected-train-episodes', type=int, default=80)
    parser.add_argument('--expected-val-episodes', type=int, default=16)
    parser.add_argument('--start-step', type=int, default=44)
    parser.add_argument('--control-end', type=int, default=135)
    parser.add_argument('--train-stride', type=int, default=1)
    parser.add_argument('--bc-steps', type=int, default=2000)
    parser.add_argument('--iql-steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--discount', type=float, default=.95)
    parser.add_argument('--expectile', type=float, default=.7)
    parser.add_argument('--beta', type=float, default=3.)
    parser.add_argument('--weight-cap', type=float, default=100.)
    parser.add_argument('--ema', type=float, default=.005)
    parser.add_argument('--gradient-clip', type=float, default=10.)
    parser.add_argument('--target-height', type=float, default=.10)
    parser.add_argument('--force-cap', type=float, default=8.)
    parser.add_argument('--force-weight', type=float, default=1.)
    parser.add_argument('--action-weight', type=float, default=.005)
    parser.add_argument('--action-low', type=float, nargs=2)
    parser.add_argument('--action-high', type=float, nargs=2)
    parser.add_argument('--allow-action-projection', action='store_true',
                        help='Explicit diagnostic override for records with nonzero other action axes; not default revised protocol')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.bc_steps < 1 or args.iql_steps < 0 or min(args.batch_size, args.eval_every, args.expected_train_episodes, args.expected_val_episodes) < 1:
        parser.error('Invalid training/episode budget')
    if not 0 < args.expectile < 1 or not 0 <= args.discount <= 1 or not 0 < args.ema <= 1 or args.beta < 0 or args.weight_cap < 1:
        parser.error('Invalid IQL hyperparameters')
    if bool(args.action_low) != bool(args.action_high):
        parser.error('Provide both --action-low and --action-high')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent device fallback')
    if any((args.out/name).exists() for name in ['bc_policy.pt', 'iql_policy.pt', 'run_metadata.json', 'fixed_config.json']):
        raise FileExistsError('Output contains a prior run; choose a fresh --out directory')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_seed(args.seed)
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out/'fixed_config.json').write_text(json.dumps(arguments, indent=2), encoding='utf-8')
    started = time.perf_counter()
    episodes, provenance = load_training_validation(args.data, args.splits, args.expected_train_episodes,
        args.expected_val_episodes, args.start_step, args.control_end, args.allow_action_projection)
    norm = training_normalization(episodes['train'])
    reward_args = {k: getattr(args, k) for k in ('target_height', 'force_cap', 'force_weight', 'action_weight')}
    train = ObservationReplay(episodes['train'], args.device, start_step=args.start_step,
                              stride=args.train_stride, reward_args=reward_args)
    val = ObservationReplay(episodes['val'], args.device, start_step=args.start_step, reward_args=reward_args)
    del episodes
    low = args.action_low or train.action.amin(0).cpu().tolist()
    high = args.action_high or train.action.amax(0).cpu().tolist()
    low, high = np.asarray(low), np.asarray(high)
    if (train.action.amin(0).cpu().numpy() < low-1e-6).any() or (train.action.amax(0).cpu().numpy() > high+1e-6).any():
        raise ValueError('Explicit action bounds exclude training actions')
    config = ReactiveConfig(variant=args.variant)
    actor = ReactiveActor(config, norm, low, high).to(args.device)
    critic, value = ReactiveTwinQ(config, norm).to(args.device), ReactiveValue(config, norm).to(args.device)
    target_critic = copy.deepcopy(critic).eval().requires_grad_(False)
    counts = {'actor_inference': parameter_count(actor), 'twin_critic': parameter_count(critic),
              'value': parameter_count(value), 'target_critic_nontrainable': parameter_count(target_critic)}
    counts['total_trainable_iql'] = counts['actor_inference']+counts['twin_critic']+counts['value']
    metadata = {'algorithm': 'Independent pixel-reactive BC plus task-specific offline IQL',
                'claim_scope': 'IQL objective adaptation; not official benchmark reproduction',
                'primary_source': 'https://arxiv.org/abs/2110.06169',
                'official_implementation_reference': 'https://github.com/ikostrikov/implicit_q_learning',
                'arguments': arguments, 'config': asdict(config), 'normalization': norm,
                'normalization_source': 'explicit training episodes only', 'parameters': counts,
                'train_transitions': len(train), 'validation_transitions': len(val),
                'train_terminal_transitions': int((train.bootstrap == 0).sum()),
                'action_low': low.tolist(), 'action_high': high.tolist(),
                'reward_equation': 'clip(h[t+1]/0.10,0,1)-mean((relu(peak[t+1]-8)/8)^2)-0.005*mean(action[t,[2,6]]^2), configurable recorded coefficients',
                'terminal_rule': 'no bootstrap at first recorded terminal or intervention end; success is not terminal',
                'observation_inputs': ['three RGB observations', 'three tactile observations (masked for vision)', 'three proprioceptive observations'],
                'no_world_model': True, 'random_initialization': True, 'encoder_sharing': 'none between actor, critic, and value',
                'data_augmentation': 'none', 'sources': provenance,
                'split_manifest_sha256': hashlib.sha256(args.splits.read_bytes()).hexdigest(),
                'bc_selection': 'minimum validation normalized two-action MSE; earliest checkpoint wins ties',
                'iql_selection': 'fixed final update; validation metrics are diagnostic only',
                'iql_initialization': 'selected reactive BC actor; independent fresh critic/value',
                'device': args.device, 'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
                'gpu': torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,
                'deterministic_algorithms': True,
                'train_reward_mean': float(train.reward.mean()), 'train_reward_min': float(train.reward.min()),
                'train_reward_max': float(train.reward.max()),
                'validation_actions_outside_training_bounds_fraction': float(((val.action < actor.action_low) | (val.action > actor.action_high)).any(-1).float().mean())}
    (args.out/'data_audit.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    optimizers = {key: torch.optim.Adam(module.parameters(), lr=args.learning_rate)
                  for key, module in [('actor', actor), ('critic', critic), ('value', value)]}
    logs, best_mse, best_step = [], math.inf, None
    bc_started = time.perf_counter()
    for step in range(1, args.bc_steps+1):
        actor.train()
        idx = torch.randint(len(train), (args.batch_size,), device=train.device)
        obs, actions = train.observations(idx), train.action[idx]
        loss = -actor.log_prob(*obs, actions).mean()
        optimizer_step(loss, optimizers['actor'], actor, args.gradient_clip)
        if step % args.eval_every == 0 or step == args.bc_steps:
            metrics = validation_metrics(actor, val, args.batch_size)
            row = {'phase': 'bc', 'step': step, 'actor_loss': float(loss.detach()), **metrics}
            logs.append(row)
            if metrics['val_normalized_action_mse'] < best_mse:
                best_mse, best_step = metrics['val_normalized_action_mse'], step
                save_checkpoint(policy_payload(actor, kind='reactive_bc', seed=args.seed,
                    selected_step=step, selection=metadata['bc_selection'], validation=metrics), args.out/'bc_policy.pt')
            print(json.dumps(row), flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    metadata.update(bc_seconds=time.perf_counter()-bc_started, bc_selected_step=best_step,
                    bc_selected_validation_mse=best_mse)
    actor, _ = load_policy(args.out/'bc_policy.pt', args.device)
    optimizers['actor'] = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    iql_started = time.perf_counter()
    for step in range(1, args.iql_steps+1):
        actor.train()
        idx = torch.randint(len(train), (args.batch_size,), device=train.device)
        losses = iql_update(actor, critic, value, target_critic, optimizers, train.batch(idx),
            args.discount, args.expectile, args.beta, args.weight_cap, args.ema, args.gradient_clip)
        if step % args.eval_every == 0 or step == args.iql_steps:
            row = {'phase': 'iql', 'step': step, **losses, **validation_metrics(actor, val, args.batch_size)}
            logs.append(row)
            print(json.dumps(row), flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    metadata.update(iql_seconds=time.perf_counter()-iql_started,
                    elapsed_seconds=time.perf_counter()-started,
                    completed_bc_updates=args.bc_steps, completed_iql_updates=args.iql_steps,
                    bc_samples_drawn=args.bc_steps*args.batch_size,
                    iql_transition_samples_drawn=args.iql_steps*args.batch_size)
    if args.iql_steps:
        save_checkpoint(policy_payload(actor, kind='reactive_iql', seed=args.seed,
            selected_step=args.iql_steps, selection=metadata['iql_selection'],
            initialized_from_bc_step=best_step), args.out/'iql_policy.pt')
        save_checkpoint({'actor': actor.state_dict(), 'critic': critic.state_dict(),
            'value': value.state_dict(), 'target_critic': target_critic.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'metadata': metadata}, args.out/'training_state.pt')
    with (args.out/'training_log.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in logs for k in row)))
        writer.writeheader()
        writer.writerows(logs)
    for name in ('bc_policy.pt', 'iql_policy.pt'):
        path = args.out/name
        if path.exists():
            metadata[name+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out/'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ('parameters', 'train_transitions', 'bc_selected_step',
        'completed_bc_updates', 'completed_iql_updates', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
