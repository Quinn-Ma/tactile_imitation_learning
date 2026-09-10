"""Independent pixel-reactive BC / offline IQL components.

No world-model weights, latent transition, synthetic transition, or imagined
reward is used. Actor, twin-Q critic, and value each have a separate encoder.
Only the actor is needed at inference. This task-specific pixel implementation
follows the IQL objectives, not an official benchmark reproduction:
https://arxiv.org/abs/2110.06169
https://github.com/ikostrikov/implicit_q_learning
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass
class ReactiveConfig:
    history: int = 3
    tactile_dim: int = 6
    proprio_dim: int = 9
    feature_dim: int = 128
    channels: int = 16
    variant: str = 'visuotactile'


def dense(in_dim, widths, out_dim):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(in_dim, width), nn.ELU()))
        in_dim = width
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class ReactiveEncoder(nn.Module):
    """Three observed frames -> feature; never predicts a future feature."""
    def __init__(self, config=None, normalization=None):
        super().__init__()
        self.config = config if isinstance(config, ReactiveConfig) else ReactiveConfig(**(config or {}))
        c = self.config
        if c.variant not in ('vision', 'visuotactile'):
            raise ValueError(c.variant)
        self.visual = nn.Sequential(
            nn.Conv2d(3*c.history, c.channels, 7, stride=2), nn.ReLU(),
            nn.Conv2d(c.channels, c.channels, 5, stride=2), nn.ReLU(),
            nn.Conv2d(c.channels, c.channels, 3, stride=2), nn.ReLU(),
            nn.Conv2d(c.channels, c.channels, 3), nn.ReLU(), nn.Flatten())
        self.touch = dense(c.history*c.tactile_dim, [64], 64)
        self.proprio = dense(c.history*c.proprio_dim, [64], 32)
        self.fusion = nn.Sequential(nn.Linear(16*c.channels+96, 256), nn.LayerNorm(256),
                                    nn.ELU(), nn.Linear(256, c.feature_dim),
                                    nn.LayerNorm(c.feature_dim), nn.ELU())
        norm = normalization or {}
        for name, dim in [('tactile', c.tactile_dim), ('proprio', c.proprio_dim)]:
            for stat in ('mean', 'std'):
                default = [0. if stat == 'mean' else 1.]*dim
                value = torch.as_tensor(norm.get(name+'_'+stat, default), dtype=torch.float32)
                if value.shape != (dim,) or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid normalization {name}_{stat}')
                if stat == 'std' and (value <= 0).any():
                    raise ValueError('Normalization scales must be positive')
                self.register_buffer(name+'_'+stat, value)

    def forward(self, rgb, tactile, proprio):
        c = self.config
        if rgb.ndim != 4 or rgb.shape[1:] != (3*c.history, 64, 64):
            raise ValueError(f'Expected RGB [B,{3*c.history},64,64], got {tuple(rgb.shape)}')
        if tactile.shape[1:] != (c.history, c.tactile_dim):
            raise ValueError('Tactile history shape mismatch')
        if proprio.shape[1:] != (c.history, c.proprio_dim):
            raise ValueError('Proprioception history shape mismatch')
        touch = (tactile-self.tactile_mean)/self.tactile_std
        if c.variant == 'vision':
            touch = torch.zeros_like(touch)
        position = (proprio-self.proprio_mean)/self.proprio_std
        return self.fusion(torch.cat((self.visual(rgb.float()/255.-.5),
                                      self.touch(touch.flatten(1)),
                                      self.proprio(position.flatten(1))), dim=-1))


class ReactiveActor(nn.Module):
    """Gaussian likelihood for BC/AWR; bounded deterministic mean at inference.

As in the official IQL policy's unsquashed-distribution option, the Gaussian
mean is tanh-bounded and its state-independent log std is learned. Likelihoods
are in training-normalized two-action units; deployment uses the mean only.
"""
    def __init__(self, config=None, normalization=None, action_low=None, action_high=None):
        super().__init__()
        self.encoder = ReactiveEncoder(config, normalization)
        self.head = dense(self.encoder.config.feature_dim, [128, 128], 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        low = torch.tensor(action_low if action_low is not None else [-1., -1.], dtype=torch.float32)
        high = torch.tensor(action_high if action_high is not None else [1., 1.], dtype=torch.float32)
        if low.shape != (2,) or high.shape != (2,) or not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError('Expected two finite action bounds')
        if not ((high > low).all() and (low >= -1).all() and (high <= 1).all()):
            raise ValueError('Action bounds must be nondegenerate within [-1,1]')
        self.register_buffer('action_low', low)
        self.register_buffer('action_high', high)

    def unit_mean(self, rgb, tactile, proprio):
        return self.head(self.encoder(rgb, tactile, proprio)).tanh()

    def normalize_action(self, action):
        return 2*(action-self.action_low)/(self.action_high-self.action_low)-1

    def forward(self, rgb, tactile, proprio):
        unit = self.unit_mean(rgb, tactile, proprio)
        return self.action_low+(unit+1)*.5*(self.action_high-self.action_low)

    def log_prob(self, rgb, tactile, proprio, action):
        mean = self.unit_mean(rgb, tactile, proprio)
        log_std = self.log_std.clamp(-5., 2.)
        unit = self.normalize_action(action)
        return (-.5*((unit-mean)/log_std.exp()).square()-log_std-.5*math.log(2*math.pi)).sum(-1)


class ReactiveTwinQ(nn.Module):
    def __init__(self, config=None, normalization=None):
        super().__init__()
        self.encoder = ReactiveEncoder(config, normalization)
        self.q1 = dense(self.encoder.config.feature_dim+2, [256, 256], 1)
        self.q2 = dense(self.encoder.config.feature_dim+2, [256, 256], 1)

    def forward(self, rgb, tactile, proprio, unit_action):
        x = torch.cat((self.encoder(rgb, tactile, proprio), unit_action), -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReactiveValue(nn.Module):
    def __init__(self, config=None, normalization=None):
        super().__init__()
        self.encoder = ReactiveEncoder(config, normalization)
        self.head = dense(self.encoder.config.feature_dim, [256, 256], 1)

    def forward(self, rgb, tactile, proprio):
        return self.head(self.encoder(rgb, tactile, proprio)).squeeze(-1)


def expectile_loss(residual, expectile=.7):
    if not 0 < expectile < 1:
        raise ValueError('Expectile must lie in (0,1)')
    return (torch.where(residual > 0, expectile, 1-expectile)*residual.square()).mean()


def td_target(reward, bootstrap, next_value, discount=.95):
    return reward+discount*bootstrap*next_value


def observed_reward(next_height, next_peak, action, target_height=.10,
                    force_cap=8., force_weight=1., action_weight=.005):
    """Same height/peak/action stage reward as WM height RL, with REAL labels.

next_peak is the two-finger peak over the physics substeps caused by the
current action. Endpoint force is deliberately not an acceptable substitute.
No reward clipping/normalization beyond height clipping is introduced.
"""
    if target_height <= 0 or force_cap <= 0:
        raise ValueError('Height and force denominators must be positive')
    task = (next_height.reshape(-1)/target_height).clamp(0, 1)
    excess = ((next_peak.clamp_min(0)-force_cap).clamp_min(0)/force_cap).square().mean(-1)
    return task-force_weight*excess-action_weight*action.square().mean(-1)


def expand_action(action):
    out = action.new_zeros(*action.shape[:-1], 7)
    out[..., 2], out[..., 6] = action[..., 0], action[..., 1]
    return out


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def policy_payload(actor, **metadata):
    norm = {name: buffer.detach().cpu().tolist() for name, buffer in actor.encoder.named_buffers()
            if name.endswith('_mean') or name.endswith('_std')}
    return {'format': 'reactive_pixel_policy_v1', 'config': asdict(actor.encoder.config),
            'normalization': norm, 'action_low': actor.action_low.detach().cpu().tolist(),
            'action_high': actor.action_high.detach().cpu().tolist(),
            'actor': {k: v.detach().cpu() for k, v in actor.state_dict().items()},
            'actor_parameters': parameter_count(actor), 'no_world_model': True, **metadata}


def load_policy(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get('format') != 'reactive_pixel_policy_v1' or payload.get('no_world_model') is not True:
        raise ValueError('Not a reactive policy checkpoint')
    actor = ReactiveActor(payload['config'], payload['normalization'],
                          payload['action_low'], payload['action_high']).to(device)
    actor.load_state_dict(payload['actor'])
    return actor.eval(), payload
