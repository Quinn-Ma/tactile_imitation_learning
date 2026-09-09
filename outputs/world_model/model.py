"""Action-conditioned visual/tactile latent model adapted from TD-MPC2 layers.

Model-only adaptation: no TD-MPC2 value learning, policy prior, or official RL
result is reproduced here. Simulator state is used as a supervised target only.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from tdmpc2_layers import SimNorm, conv, mlp


@dataclass
class ModelConfig:
    proprio_dim: int = 9
    tactile_dim: int = 6
    action_dim: int = 7
    history: int = 3
    latent_dim: int = 128
    hidden_dim: int = 384
    channels: int = 16
    variant: str = 'visuotactile'


class SmallWorldModel(nn.Module):
    def __init__(self, config=None, normalization=None):
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**(config or {}))
        c = self.config
        if c.variant not in ('vision', 'visuotactile'):
            raise ValueError(c.variant)
        self.visual = conv(3 * c.history, c.channels)
        self.touch = mlp(c.tactile_dim * c.history, [64], 64)
        self.proprio = mlp(c.proprio_dim * c.history, [64], 32)
        self.fusion = mlp(16 * c.channels + 96, [256], c.latent_dim, SimNorm())
        self.dynamics = mlp(c.latent_dim + c.action_dim, [c.hidden_dim] * 2,
                            c.latent_dim, SimNorm())
        # Endpoint forces (6), height (1), support (1), contacts (2), reward (1),
        # and the two normal-force maxima over the preceding physical substeps.
        self.physical_head = mlp(c.latent_dim, [128], c.tactile_dim + 7)
        # Low-resolution future RGB is an auxiliary target; no full image generator claim.
        self.image_head = mlp(c.latent_dim, [256], 3 * 16 * 16)
        norm = normalization or {}
        for name, dimension in [('tactile', c.tactile_dim), ('proprio', c.proprio_dim),
                                ('height', 1), ('reward', 1), ('peak', 2)]:
            mean = torch.as_tensor(norm.get(name + '_mean', [0.] * dimension), dtype=torch.float32)
            std = torch.as_tensor(norm.get(name + '_std', [1.] * dimension), dtype=torch.float32)
            self.register_buffer(name + '_mean', mean)
            self.register_buffer(name + '_std', std.clamp_min(1e-3))

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, rgb, tactile, proprio):
        """rgb[B,3*history,64,64], tactile[B,history,D], proprio likewise.

        For a single latest touch/proprio frame, repeat explicitly at the caller.
        RGB must remain uint8 or float values in [0,255], never normalized twice.
        """
        c = self.config
        if rgb.shape[-2:] != (64, 64) or rgb.shape[-3] != c.history * 3:
            raise ValueError(f'Expected stacked RGB[B,{3*c.history},64,64], got {rgb.shape}')
        if tactile.ndim == 2:
            tactile = tactile[:, None].expand(-1, c.history, -1)
        if proprio.ndim == 2:
            proprio = proprio[:, None].expand(-1, c.history, -1)
        t = (tactile - self.tactile_mean) / self.tactile_std
        if c.variant == 'vision':
            t = torch.zeros_like(t)  # same architecture and parameter count
        p = (proprio - self.proprio_mean) / self.proprio_std
        z = torch.cat((self.visual(rgb), self.touch(t.flatten(1)), self.proprio(p.flatten(1))), -1)
        return self.fusion(z)

    def next(self, latent, action):
        return self.dynamics(torch.cat((latent, action), -1))

    def decode(self, latent):
        raw = self.physical_head(latent)
        d = self.config.tactile_dim
        result = {
            'tactile_normalized': raw[..., :d],
            'height_normalized': raw[..., d:d+1],
            'support_logit': raw[..., d+1:d+2],
            'contact_logit': raw[..., d+2:d+4],
            'reward_normalized': raw[..., d+4:d+5],
            'peak_normalized': raw[..., d+5:d+7],
            'rgb16': self.image_head(latent).sigmoid().reshape(*latent.shape[:-1], 3, 16, 16),
        }
        result['tactile'] = result['tactile_normalized'] * self.tactile_std + self.tactile_mean
        result['height'] = result['height_normalized'] * self.height_std + self.height_mean
        result['reward'] = result['reward_normalized'] * self.reward_std + self.reward_mean
        result['peak'] = result['peak_normalized'] * self.peak_std + self.peak_mean
        result['support'] = result['support_logit'].sigmoid()
        result['contact'] = result['contact_logit'].sigmoid()
        return result

    def rollout(self, latent, actions, include_initial=False):
        """Differentiable rollout: actions[B,H,A]; returns [B,H,...] outputs."""
        states = [latent] if include_initial else []
        for action in actions.unbind(1):
            latent = self.next(latent, action)
            states.append(latent)
        z = torch.stack(states, 1)
        result = self.decode(z)
        result['latent'] = z
        return result

    @torch.no_grad()
    def plan_candidates(self, rgb, tactile, proprio, candidates, force_cap=15.0,
                        force_margin=0.0, height_margin=0.0, force_weight=0.002,
                        use_reward=False):
        """Score behavior-supported action snippets [K,H,A] for ONE observation.

        Hard masks only bound predicted forces, not true safety. Calibration is
        valid only under its stated trajectory-distribution assumptions. Caller
        supplies supported candidates; this method invents no action bounds.
        Returns index=-1 if all candidates fail the predicted force envelope.
        """
        if rgb.shape[0] != 1:
            raise ValueError('Planner expects exactly one initial observation.')
        z = self.encode(rgb, tactile, proprio).expand(len(candidates), -1)
        pred = self.rollout(z, candidates)
        normal = pred['peak'].clamp_min(0)
        worst_force = (normal + force_margin).amax(dim=(1, 2))
        feasible = worst_force <= force_cap
        progress = pred['height'][:, -1, 0] - height_margin
        if use_reward:
            progress = pred['reward'][..., 0].sum(1)
        score = progress - force_weight * normal.sum(dim=(1, 2))
        score = score.masked_fill(~feasible, -torch.inf)
        index = int(score.argmax()) if bool(feasible.any()) else -1
        return {'index': index, 'scores': score, 'feasible': feasible,
                'predicted': pred, 'action': candidates[index, 0] if index >= 0 else None}


def load_model(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SmallWorldModel(payload['config'], payload.get('normalization')).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, payload


def checkpoint_payload(model, **metadata):
    return {'config': asdict(model.config), 'model': model.state_dict(),
            'normalization': {key: value.detach().cpu().tolist()
                              for key, value in model.named_buffers()
                              if key.endswith('_mean') or key.endswith('_std')}, **metadata}
