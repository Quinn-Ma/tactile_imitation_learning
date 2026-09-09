"""Episode-disjoint data loading for RGB/action/contact simulation records."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset


def scalar_string(value):
    result = np.asarray(value).item()
    return result.decode() if isinstance(result, bytes) else str(result)


def read_episodes(directory, calibration_fraction=0.15, split_seed=123):
    episodes = []
    for path in sorted(Path(directory).rglob('*.npz')):
        with np.load(path, allow_pickle=False) as source:
            if not all(k in source for k in ('rgb', 'tactile', 'proprio', 'action', 'height', 'support', 'contact')):
                continue
            ep = {k: source[k].copy() for k in source.files}
        ep['episode_id'] = scalar_string(ep.get('episode_id', path.stem))
        ep['split'] = scalar_string(ep.get('split', 'train'))
        ep['path'] = str(path.resolve())
        n = len(ep['action'])
        if ep['rgb'].shape != (n+1, 64, 64, 3):
            raise ValueError(f'{path}: RGB shape {ep["rgb"].shape}, actions={n}')
        for k in ('tactile', 'proprio', 'height', 'support', 'contact'):
            if len(ep[k]) != n+1:
                raise ValueError(f'{path}: {k} must have T+1 observations')
            ep[k] = np.asarray(ep[k], dtype=np.float32).reshape(n+1, -1)
            if not np.isfinite(ep[k]).all():
                raise ValueError(f'{path}: non-finite {k}')
        ep['action'] = ep['action'].astype(np.float32).reshape(n, -1)
        if 'substep_normal_peak' not in ep:
            raise ValueError(f'{path}: substep_normal_peak is required; endpoint forces cannot label peak safety')
        ep['peak'] = np.asarray(ep['substep_normal_peak'], dtype=np.float32).reshape(-1, 2)
        if len(ep['peak']) != n+1 or not np.isfinite(ep['peak']).all():
            raise ValueError(f'{path}: invalid substep_normal_peak labels, expected [T+1,2]')
        if 'reward' in ep:
            reward = ep['reward'].astype(np.float32).reshape(-1, 1)
            if len(reward) == n:
                reward = np.concatenate((np.zeros((1, 1), np.float32), reward), axis=0)
            if len(reward) != n+1:
                raise ValueError(f'{path}: reward must have T or T+1 entries')
            ep['reward'] = reward
            ep['has_reward'] = True
        else:
            ep['reward'] = np.zeros((n+1, 1), np.float32)
            ep['has_reward'] = False
        episodes.append(ep)
    if not episodes:
        raise ValueError(f'No schema-compatible episode npz files under {directory}')
    ids = [x['episode_id'] for x in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate episode_id values; group identity is ambiguous.')
    if not any(e['split'] == 'calibration' for e in episodes):
        train = [e for e in episodes if e['split'] == 'train']
        if len(train) >= 3:
            ordered = sorted(train, key=lambda e: hashlib.sha256(
                f'{split_seed}/{e["episode_id"]}'.encode()).hexdigest())
            amount = max(1, min(len(train)-2, round(len(train)*calibration_fraction)))
            for e in ordered[:amount]:
                e['split'] = 'calibration'
    return episodes


def normalization_from_training(episodes):
    train = [e for e in episodes if e['split'] == 'train']
    if not train:
        raise ValueError('No training episodes.')
    result = {}
    for name, floor in [('tactile', .05), ('proprio', .01), ('height', .01), ('reward', .05), ('peak', .05)]:
        count = sum(len(e[name]) for e in train)
        total = sum(e[name].sum(0, dtype=np.float64) for e in train)
        second = sum(np.square(e[name].astype(np.float64)).sum(0) for e in train)
        mean = total / count
        std = np.sqrt(np.maximum(second / count - mean*mean, 0)).clip(floor)
        result[name+'_mean'] = mean.tolist()
        result[name+'_std'] = std.tolist()
    return result


class SequenceDataset(Dataset):
    def __init__(self, episodes, split, horizon=5, history=3, stride=1,
                 max_windows_per_episode=None, shuffle_tactile=False):
        self.episodes = [e for e in episodes if e['split'] == split]
        self.horizon = horizon
        self.history = history
        self.shuffle_tactile = shuffle_tactile
        self.indices = []
        for j, e in enumerate(self.episodes):
            starts = list(range(0, len(e['action'])-horizon+1, stride))
            if max_windows_per_episode and len(starts) > max_windows_per_episode:
                starts = np.linspace(0, len(starts)-1, max_windows_per_episode).astype(int)
                starts = [list(range(0, len(e['action'])-horizon+1, stride))[i] for i in starts]
            self.indices.extend((j, start) for start in starts)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode_index, start = self.indices[index]
        e = self.episodes[episode_index]
        times = np.arange(start, start+self.horizon+1)
        history = np.maximum(times[:, None] + np.arange(-self.history+1, 1)[None, :], 0)
        rgb = e['rgb'][history].transpose(0, 1, 4, 2, 3).reshape(-1, self.history*3, 64, 64)
        tactile = e['tactile'][history].copy()
        if self.shuffle_tactile:
            if len(self.episodes) > 1:
                other = self.episodes[(episode_index+1) % len(self.episodes)]
                tactile = other['tactile'][history % len(other['tactile'])].copy()
            else:
                tactile[:] = 0
        values = {'rgb': np.ascontiguousarray(rgb), 'tactile': tactile,
                  'proprio': e['proprio'][history].copy(),
                  'action': e['action'][start:start+self.horizon].copy(),
                  'episode_index': np.int64(episode_index), 'start': np.int64(start),
                  'has_reward': np.float32(e['has_reward'])}
        for k in ('tactile', 'height', 'support', 'contact', 'reward', 'peak'):
            values['target_'+k] = e[k][times].copy()
        return {k: torch.as_tensor(v) for k, v in values.items()}


def split_manifest(episodes):
    return [{'episode_id': e['episode_id'], 'split': e['split'], 'path': e['path'],
             'steps': len(e['action']), 'has_reward': e['has_reward']} for e in episodes]
