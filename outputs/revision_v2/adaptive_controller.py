"""Fixed exploratory model-assisted controller and action-query interface.

The matched reactive ablation uses exactly the same measured-force controller,
end-effector lift schedule, action limits, and optional measured-force guard.
Model assistance only selects a local candidate using a frozen learned model.
No object pose, measured object height, mass, friction, support, success label,
simulation phase, or simulator state is accepted by the policy interface.

This is an empirical controller, not a safety certificate. Force anchoring is a
local offset correction, not conformal calibration. A feasible *prediction*
does not imply that the executed action respects the true force budget.

Stdio: one JSON request per line. Observations contain rgb_base64 (64x64x3
uint8), tactile[6], proprio[9]. control_active=false warms history without
starting the lift schedule; its returned action must not be executed. On the
first active observation the measured end-effector z becomes the lift origin.
Modes: model_guard, model_no_guard, reactive_guard, reactive_no_guard.
Commands: reset; quit; forecast with actions[K,H,7] queries the last observed
history without advancing it or the controller's schedule. Forecast commands
are suitable for a paired alternate-action audit; outputs are predictions only.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
    # Fixed before revised test evaluation; seconds = steps * 0.05.
    horizon: int = 5
    target_force_N: float = 5.0
    force_gain: float = 0.12
    force_cap_N: float = 8.0
    guard_buffer_N: float = 0.75
    force_margin_N: float = 0.0
    grip_low: float = -0.25
    grip_high: float = 0.35
    vertical_limit: float = 0.6
    vertical_gain: float = 12.0
    target_eef_lift_m: float = 0.14
    grasp_steps: int = 18
    lift_ramp_steps: int = 50
    min_bilateral_normal_N: float = 0.25
    vertical_offsets: tuple = (0.0, -0.06, 0.06)
    gripper_offsets: tuple = (0.0, -0.10, 0.10)
    scoring_target_height_m: float = 0.10
    force_cost: float = 0.02
    action_deviation_cost: float = 0.002
    force_anchor: bool = True

    def __post_init__(self):
        if not 1 <= self.horizon <= 5:
            raise ValueError('The existing world model was trained only to horizon five.')
        if not 0 < self.target_force_N < self.force_cap_N:
            raise ValueError('Target force must lie strictly within the force budget.')
        if not 0 <= self.guard_buffer_N < self.force_cap_N:
            raise ValueError('Invalid measured-force guard buffer.')
        if self.force_margin_N < 0:
            raise ValueError('Force margin cannot be negative.')
        if self.grasp_steps < 0 or self.lift_ramp_steps < 1:
            raise ValueError('Invalid schedule.')


def _physical_action(action):
    """Reject accidentally privileged/non-comparable action subspaces."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1:] != (7,) or not np.isfinite(action).all():
        raise ValueError('Actions must be finite and have final dimension seven.')
    if np.any(np.abs(action) > 1.000001):
        raise ValueError('Action outside normalized actuator range.')
    if np.any(action[..., [0, 1, 3, 4, 5]] != 0):
        raise ValueError('Only vertical and gripper actions are allowed.')
    return action


class AdaptiveController:
    MODES = ('model_guard', 'model_no_guard', 'reactive_guard', 'reactive_no_guard')

    def __init__(self, model=None, device='cpu', config=None, mode='model_guard'):
        if mode not in self.MODES:
            raise ValueError(mode)
        if mode.startswith('model_') and model is None:
            raise ValueError('Model-assisted mode requires a frozen world model.')
        self.model = model
        self.device = device
        self.config = config or ControllerConfig()
        self.mode = mode
        self.use_model = mode.startswith('model_')
        self.use_guard = mode.endswith('_guard') and not mode.endswith('_no_guard')
        self.history_size = model.config.history if model is not None else 3
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.history_size)
        self.active_steps = 0
        self.eef_origin_z = None

    def _observation(self, rgb, tactile, proprio):
        rgb = np.asarray(rgb)
        tactile = np.asarray(tactile, np.float32)
        proprio = np.asarray(proprio, np.float32)
        if rgb.shape != (64, 64, 3) or rgb.dtype != np.uint8:
            raise ValueError('RGB must be uint8 with shape 64x64x3.')
        if tactile.shape != (6,) or proprio.shape != (9,):
            raise ValueError('Expected tactile[6] and proprio[9].')
        if not np.isfinite(tactile).all() or not np.isfinite(proprio).all():
            raise ValueError('Nonfinite available observation.')
        return rgb.copy(), tactile.copy(), proprio.copy()

    def _base_action(self, tactile, proprio):
        c = self.config
        normal = np.maximum(tactile[[0, 3]], 0.0)
        lift_fraction = np.clip((self.active_steps-c.grasp_steps)/c.lift_ramp_steps, 0, 1)
        target_z = self.eef_origin_z + c.target_eef_lift_m * lift_fraction
        action = np.zeros(7, np.float32)
        action[2] = np.clip(c.vertical_gain * (target_z-proprio[2]),
                            -c.vertical_limit, c.vertical_limit)
        if normal.min() < c.min_bilateral_normal_N:
            # No object/contact labels: the decision uses only the two normal
            # tactile channels. Keep the current arm height until contact exists.
            action[2] = min(0.0, action[2])
        action[6] = np.clip(c.force_gain * (c.target_force_N-normal.mean()),
                            c.grip_low, c.grip_high)
        return action

    def guard(self, action, tactile):
        """Same memoryless measured-force intervention in each guarded mode."""
        action = _physical_action(action).copy()
        threshold = self.config.force_cap_N-self.config.guard_buffer_N
        intervention = bool(self.use_guard and np.max(tactile[[0, 3]]) >= threshold)
        if intervention:
            action[..., 6] = self.config.grip_low
            action[..., 2] = np.minimum(action[..., 2], 0.0)
        return action, intervention

    def _candidate_actions(self, base, tactile):
        c = self.config
        candidates = []
        for z_offset in c.vertical_offsets:
            for grip_offset in c.gripper_offsets:
                action = base.copy()
                action[2] = np.clip(action[2]+z_offset, -c.vertical_limit, c.vertical_limit)
                # Preserve the identical pre-lift/contact condition in every
                # candidate, rather than letting a model bypass common setup.
                if self.active_steps < c.grasp_steps or min(tactile[0], tactile[3]) < c.min_bilateral_normal_N:
                    action[2] = base[2]
                action[6] = np.clip(action[6]+grip_offset, c.grip_low, c.grip_high)
                action, _ = self.guard(action, tactile)
                if not any(np.array_equal(action, previous) for previous in candidates):
                    candidates.append(action)
        return np.stack(candidates)

    def _encode(self):
        import torch
        if not self.history:
            raise ValueError('Observe a state before querying actions.')
        rgb = np.stack([entry[0] for entry in self.history]).transpose(0, 3, 1, 2)
        tactile = np.stack([entry[1] for entry in self.history])[None]
        proprio = np.stack([entry[2] for entry in self.history])[None]
        return self.model.encode(torch.as_tensor(rgb.reshape(1, -1, 64, 64), device=self.device),
            torch.as_tensor(tactile, device=self.device),
            torch.as_tensor(proprio, device=self.device))

    def forecast_action_sequences(self, actions):
        """Forecast supplied actions from last history, with no state mutation.

        Each horizon entry is a raw 0.05s action, not a future closed-loop guard
        rollout. Both raw forecasts and current-force-anchored forecasts are
        returned; only the latter are used by the default candidate selector.
        """
        import torch
        if self.model is None:
            raise ValueError('Forecast requires a world model, even in reactive mode.')
        actions = _physical_action(actions)
        if actions.ndim != 3 or not 1 <= actions.shape[1] <= 5 or not 1 <= len(actions) <= 64:
            raise ValueError('Expected actions[K,H,7] with K in 1..64, H in 1..5.')
        with torch.no_grad():
            latent = self._encode()
            current = self.model.decode(latent)
            prediction = self.model.rollout(latent.expand(len(actions), -1),
                                           torch.as_tensor(actions, device=self.device))
        decoded_normal = current['tactile'][0, [0, 3]].cpu().numpy()
        measured_normal = np.maximum(self.history[-1][1][[0, 3]], 0)
        offset = measured_normal-decoded_normal if self.config.force_anchor else np.zeros(2, np.float32)
        raw_peak = prediction['peak'].cpu().numpy()
        raw_tactile = prediction['tactile'].cpu().numpy()
        anchored_tactile = raw_tactile.copy()
        anchored_tactile[..., [0, 3]] = np.maximum(0, anchored_tactile[..., [0, 3]]+offset)
        result = {
            'actions': actions.tolist(),
            'raw_peak_N': raw_peak.tolist(),
            'anchored_peak_N': np.maximum(0, raw_peak+offset).tolist(),
            'raw_tactile_N': raw_tactile.tolist(),
            'anchored_tactile_N': anchored_tactile.tolist(),
            'height_m': prediction['height'][..., 0].cpu().tolist(),
            'native_reward': prediction['reward'][..., 0].cpu().tolist(),
            'current_decoded_normal_N': decoded_normal.tolist(),
            'measured_normal_N': measured_normal.tolist(),
            'force_anchor_offset_N': offset.tolist(),
            'interpretation': 'Learned predictions for fixed action sequences; not executed outcomes or safety bounds.'}
        if any(not np.isfinite(np.asarray(result[key])).all() for key in (
                'raw_peak_N', 'anchored_peak_N', 'raw_tactile_N', 'height_m', 'native_reward')):
            raise FloatingPointError('World model returned nonfinite forecasts.')
        return result

    def step(self, rgb, tactile, proprio, control_active=True):
        entry = self._observation(rgb, tactile, proprio)
        self.history.append(entry)
        while len(self.history) < self.history_size:
            self.history.appendleft(entry)
        if not control_active:
            return {'action': [0.0]*7, 'warmup': True, 'mode': self.mode,
                    'note': 'History only; do not execute this action.'}
        if self.eef_origin_z is None:
            self.eef_origin_z = float(entry[2][2])
        raw_base = self._base_action(entry[1], entry[2])
        base, guarded = self.guard(raw_base, entry[1])
        result = {'mode': self.mode, 'active_step': self.active_steps,
                  'base_action_before_guard': raw_base.tolist(), 'base_action': base.tolist(),
                  'guard_intervened': guarded, 'eef_reference_z_m': self.eef_origin_z,
                  'force_anchor_enabled': bool(self.config.force_anchor and self.use_model)}
        action = base
        if self.use_model:
            candidates = self._candidate_actions(raw_base, entry[1])
            sequences = np.repeat(candidates[:, None, :], self.config.horizon, axis=1)
            forecast = self.forecast_action_sequences(sequences)
            force = np.asarray(forecast['anchored_peak_N'])
            height = np.asarray(forecast['height_m'])
            worst = force.max(axis=(1, 2)) + self.config.force_margin_N
            feasible = worst <= self.config.force_cap_N
            progress = np.clip(height[:, -1]/self.config.scoring_target_height_m, 0, 1)
            cost = self.config.force_cost * force.mean(axis=(1, 2))/self.config.force_cap_N
            cost += self.config.action_deviation_cost * np.square(candidates-base).sum(axis=1)
            scores = progress-cost
            selected = int(np.where(feasible, scores, -np.inf).argmax()) if feasible.any() else -1
            if selected >= 0:
                action = candidates[selected]
            result.update(candidate_actions=candidates.tolist(), forecast=forecast,
                          candidate_scores=scores.tolist(), predicted_feasible=feasible.tolist(),
                          candidate_worst_anchored_force_plus_margin_N=worst.tolist(),
                          selected_index=selected, fallback_to_reactive=selected < 0)
            if selected >= 0:
                result.update(predicted_peak=forecast['raw_peak_N'][selected][0],
                              predicted_peak_anchored=forecast['anchored_peak_N'][selected][0],
                              predicted_tactile=forecast['raw_tactile_N'][selected][0],
                              predicted_height=forecast['height_m'][selected][0],
                              predicted_reward=forecast['native_reward'][selected][0])
            # A fallback may not be present in a custom candidate grid. Its
            # forecast is queried explicitly, keeping the action-audit exact.
            else:
                fallback = self.forecast_action_sequences(action[None, None, :])
                result.update(predicted_peak=fallback['raw_peak_N'][0][0],
                              predicted_peak_anchored=fallback['anchored_peak_N'][0][0],
                              predicted_tactile=fallback['raw_tactile_N'][0][0],
                              predicted_height=fallback['height_m'][0][0],
                              predicted_reward=fallback['native_reward'][0][0])
        self.active_steps += 1
        result['action'] = _physical_action(action).tolist()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--mode', choices=AdaptiveController.MODES, default='model_guard')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--config', type=Path, help='Frozen configuration JSON; do not tune on test data.')
    args = parser.parse_args()
    config = ControllerConfig(**json.loads(args.config.read_text(encoding='utf-8'))) if args.config else ControllerConfig()
    model = None
    if args.model is not None:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'world_model'))
        from model import load_model
        torch.set_num_threads(4)
        model, _ = load_model(args.model, args.device)
    if args.mode.startswith('model_') and model is None:
        parser.error('Model-assisted mode requires --model.')
    controller = AdaptiveController(model, args.device, config, args.mode)
    ready = {'ready': True, 'mode': args.mode, 'config': asdict(config),
             'model_parameters': model.num_parameters if model is not None else 0,
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model else None}
    print(json.dumps(ready, allow_nan=False), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get('command')
            if command == 'quit':
                break
            if command == 'reset':
                controller.reset()
                result = {'reset': True}
            elif command == 'forecast':
                result = controller.forecast_action_sequences(request['actions'])
            elif command is not None:
                raise ValueError('Unknown command: '+str(command))
            else:
                allowed = {'rgb_base64', 'rgb', 'tactile', 'proprio', 'control_active', 'reset'}
                unexpected = set(request)-allowed
                if unexpected:
                    raise ValueError('Unsupported observation fields: '+', '.join(sorted(unexpected)))
                if request.get('reset'):
                    controller.reset()
                if 'rgb_base64' in request:
                    rgb = np.frombuffer(base64.b64decode(request['rgb_base64'], validate=True), np.uint8).reshape(64, 64, 3)
                else:
                    raw = np.asarray(request['rgb'])
                    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 255):
                        raise ValueError('RGB outside uint8 range.')
                    rgb = raw.astype(np.uint8)
                result = controller.step(rgb, request['tactile'], request['proprio'], request.get('control_active', True))
            print(json.dumps(result, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({'error': type(error).__name__+': '+str(error)}), flush=True)


if __name__ == '__main__':
    main()
