"""Bounded model-based RL pilot using a frozen learned world model.

Behavior-cloned initialization is followed by differentiable imagined actor
updates and a TD value regression. This is an adaptation, not the official
TD-MPC2 learning algorithm. No simulator labels enter the actor observation.
The actor controls vertical motion and continuous gripper rate after a shared
scripted approach. Test-simulator interaction must be reported separately.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from model import load_model


class Actor(nn.Module):
    def __init__(self, latent_dim=128, action_low=None, action_high=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.ELU(),
                                 nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 2))
        self.register_buffer('action_low', torch.tensor(action_low if action_low is not None else [-1., -1.]))
        self.register_buffer('action_high', torch.tensor(action_high if action_high is not None else [1., 1.]))

    def forward(self, latent):
        unit = (self.net(latent).tanh() + 1.) / 2.
        return self.action_low + unit * (self.action_high - self.action_low)


class Value(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.ELU(),
                                 nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, latent):
        return self.net(latent).squeeze(-1)


def expand_action(action):
    """Map [vertical,gripper] to benchmark's [xyz,rotation_xyz,gripper]."""
    out = action.new_zeros(*action.shape[:-1], 7)
    out[..., 2] = action[..., 0]
    out[..., 6] = action[..., 1]
    return out


def load_episode(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    aliases = {'rgb': ('rgb', 'images', 'image'), 'tactile': ('tactile',),
               'proprio': ('proprio',), 'action': ('action', 'actions')}
    for name, choices in aliases.items():
        key = next((key for key in choices if key in result), None)
        if key is None:
            raise ValueError(f'{path}: missing {name}; keys={list(result)}')
        result[name] = result[key]
    return result


def list_training_files(split_file, data_dir):
    split = json.loads(Path(split_file).read_text(encoding='utf-8'))
    if isinstance(split, list):
        names = [row['path'] for row in split if row['split'] == 'train']
    else:
        names = split.get('train', split.get('splits', {}).get('train', []))
    if not names:
        raise ValueError('Split file must contain explicit train file names, never infer from test.')
    if isinstance(names, dict):
        names = names.get('episodes', [])
    paths = []
    for name in names:
        if isinstance(name, dict):
            name = name.get('file', name.get('path'))
        path = Path(name)
        if not path.exists():
            path = data_dir / path
        if not path.exists():
            path = data_dir / Path(name).name
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def encode_replay(model, files, device, start_step, horizon=5, control_end=135, stride=2):
    latent_chunks, action_chunks, bootstrap_chunks, provenance = [], [], [], []
    hist = model.config.history
    for path in files:
        ep = load_episode(path)
        terminal = min(len(ep['action']), len(ep['rgb']) - 1, control_end)
        stop = terminal - horizon + 1
        indexes = np.arange(max(hist - 1, start_step), stop, stride)
        if not len(indexes):
            continue
        # Deliberately no reward, object pose, mass, friction, or support labels
        # in these inputs. Those may be outputs/diagnostics of the world model.
        hs = np.maximum(0, indexes[:, None] - np.arange(hist - 1, -1, -1)[None])
        frames = ep['rgb'][hs]
        if frames.shape[-1] == 3:
            frames = frames.transpose(0, 1, 4, 2, 3)
        frames = frames.reshape(len(indexes), hist * 3, 64, 64)
        with torch.no_grad():
            for offset in range(0, len(indexes), 128):
                batch = slice(offset, offset + 128)
                latent_chunks.append(model.encode(
                    torch.as_tensor(frames[batch], device=device),
                    torch.as_tensor(ep['tactile'][hs[batch]], dtype=torch.float32, device=device),
                    torch.as_tensor(ep['proprio'][hs[batch]], dtype=torch.float32, device=device)).detach())
        action_chunks.append(torch.as_tensor(ep['action'][indexes][:, [2, 6]], dtype=torch.float32, device=device))
        bootstrap_chunks.append(torch.as_tensor(indexes + horizon < terminal, dtype=torch.float32, device=device))
        provenance.append({'file': str(path), 'num_starts': len(indexes),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if not latent_chunks:
        raise ValueError('No training states after start_step; review episode lengths.')
    return torch.cat(latent_chunks), torch.cat(action_chunks), torch.cat(bootstrap_chunks), provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--start-step', type=int, default=44)
    p.add_argument('--control-end', type=int, default=135)
    p.add_argument('--bc-steps', type=int, default=1000)
    p.add_argument('--rl-steps', type=int, default=1500)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--force-cap', type=float, default=8.)
    p.add_argument('--force-margin', type=float, default=0.)
    p.add_argument('--force-weight', type=float, default=1.)
    p.add_argument('--task-reward', choices=['native', 'height'], default='native')
    p.add_argument('--target-height', type=float, default=.10)
    p.add_argument('--behavior-weight', type=float, default=2.)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently change experimental device.')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    started = time.perf_counter()
    model, payload = load_model(args.model, args.device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    files = list_training_files(args.splits, args.data)
    replay_z, replay_a, replay_bootstrap, provenance = encode_replay(
        model, files, args.device, args.start_step, args.horizon, args.control_end)
    action_low = replay_a.amin(0).clamp(-1, 1).cpu().tolist()
    action_high = replay_a.amax(0).clamp(-1, 1).cpu().tolist()
    actor = Actor(model.config.latent_dim, action_low, action_high).to(args.device)
    value = Value(model.config.latent_dim).to(args.device)
    target_value = copy.deepcopy(value).eval()
    for param in target_value.parameters():
        param.requires_grad_(False)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=2e-4)
    value_opt = torch.optim.Adam(value.parameters(), lr=3e-4)
    logs = []
    def sample():
        idx = torch.randint(len(replay_z), (args.batch_size,), device=args.device)
        return replay_z[idx], replay_a[idx], replay_bootstrap[idx]
    for step in range(args.bc_steps):
        z, action, _ = sample()
        loss = (actor(z) - action).square().mean()
        actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        actor_opt.step()
        if step % 100 == 0:
            logs.append({'phase': 'bc', 'step': step, 'loss': float(loss.detach()), 'value_loss': 0.})
    prior = copy.deepcopy(actor).eval()
    for param in prior.parameters():
        param.requires_grad_(False)
    torch.save({'actor': actor.state_dict(), 'latent_dim': model.config.latent_dim,
                'world_model': str(args.model.resolve()), 'kind': 'behavior_clone',
                'start_step': args.start_step}, args.out / 'bc_actor.pt')
    gamma = .95
    for step in range(args.rl_steps):
        z, _, bootstrap = sample()
        states, rewards, regularizers = [z], [], []
        for k in range(args.horizon):
            action2 = actor(z)
            reference = prior(z).detach()
            regularizers.append((action2 - reference).square().mean(-1))
            z = model.next(z, expand_action(action2))
            physical = model.decode(z)
            if 'peak' not in physical:
                raise ValueError('World model must predict per-interval substep force peaks for this RL objective.')
            normals = physical['peak'].clamp_min(0)
            excess = ((normals + args.force_margin - args.force_cap).clamp_min(0) / max(args.force_cap, 1e-6)).square().mean(-1)
            # The native reward saturates below our strict 10cm evaluation goal.
            # The separately reported goal-aligned experiment rewards predicted
            # height progress at every step, encouraging continued attainment.
            # This remains a learned-model objective, never an execution oracle.
            task_reward = physical['reward'].squeeze(-1) if args.task_reward == 'native' else (
                physical['height'].squeeze(-1) / args.target_height).clamp(0, 1)
            reward = task_reward - args.force_weight * excess - .005 * action2.square().mean(-1)
            rewards.append(reward)
            states.append(z)
        # Freeze target critic weights, retaining the terminal-state gradient
        # for imagined actor learning; critic regression targets detach below.
        target = target_value(states[-1]) * bootstrap
        returns = []
        for reward in reversed(rewards):
            target = reward + gamma * target
            returns.append(target)
        returns.reverse()
        actor_loss = -returns[0].mean() + args.behavior_weight * torch.stack(regularizers).mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.)
        actor_opt.step()
        value_targets = torch.stack(returns).detach()
        values = value(torch.stack(states[:-1]).detach())
        value_loss = (values - value_targets).square().mean()
        value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(value.parameters(), 10.)
        value_opt.step()
        with torch.no_grad():
            for dest, source in zip(target_value.parameters(), value.parameters()):
                dest.lerp_(source, .01)
        if step % 100 == 0:
            row = {'phase': 'imagination_rl', 'step': step, 'loss': float(actor_loss.detach()), 'value_loss': float(value_loss.detach())}
            logs.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize() if args.device == 'cuda' else None
    checkpoint = {'actor': actor.state_dict(), 'value': value.state_dict(), 'target_value': target_value.state_dict(),
                  'prior': prior.state_dict(), 'latent_dim': model.config.latent_dim,
                  'world_model': str(args.model.resolve()), 'kind': 'imagined_actor_value',
                  'start_step': args.start_step, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    torch.save(checkpoint, args.out / 'rl_actor.pt')
    with (args.out / 'training_log.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)
    metadata = {'claim_scope': 'frozen learned-model imagination RL pilot; simulator evaluation required',
                'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
                'replay_states': len(replay_z), 'training_episodes': provenance,
                'action_low': action_low, 'action_high': action_high,
                'actor_parameters': sum(p.numel() for p in actor.parameters()),
                'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
                'elapsed_seconds': time.perf_counter() - started, 'python': sys.executable,
                'torch': torch.__version__, 'arguments': checkpoint['arguments']}
    (args.out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'training_episodes'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
