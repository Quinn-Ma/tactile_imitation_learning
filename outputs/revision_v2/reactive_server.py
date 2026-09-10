"""Local stdio actor-only service compatible with PolicyBridge observations.

Launch: python reactive_server.py --checkpoint .../iql_policy.pt --device cuda
No value/Q modules or world model are loaded; the reply contains no forecasts.
"""
import argparse
import base64
from collections import deque
import json
import sys

import numpy as np
import torch
from reactive import expand_action, load_policy, parameter_count


class ReactiveSession:
    def __init__(self, actor, device):
        self.actor, self.device = actor, device
        self.config = actor.encoder.config
        self.history = deque(maxlen=self.config.history)

    def reset(self):
        self.history.clear()

    @torch.inference_mode()
    def request(self, request):
        if request.get('command') == 'reset':
            self.reset()
            return {'reset': True}
        if request.get('reset'):
            self.reset()
        if 'rgb_base64' in request:
            rgb = np.frombuffer(base64.b64decode(request['rgb_base64'], validate=True),
                                np.uint8).reshape(64, 64, 3)
        else:
            raw = np.asarray(request['rgb'])
            if raw.shape != (64, 64, 3) or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
                raise ValueError('Invalid RGB frame')
            rgb = raw.astype(np.uint8)
        touch = np.asarray(request['tactile'], dtype=np.float32)
        proprio = np.asarray(request['proprio'], dtype=np.float32)
        if touch.shape != (self.config.tactile_dim,) or proprio.shape != (self.config.proprio_dim,):
            raise ValueError('Observation dimension mismatch')
        if not np.isfinite(touch).all() or not np.isfinite(proprio).all():
            raise ValueError('Nonfinite observation')
        entry = rgb.copy(), touch.copy(), proprio.copy()
        self.history.append(entry)
        while len(self.history) < self.config.history:
            self.history.appendleft(entry)
        images = np.stack([x[0] for x in self.history]).transpose(0, 3, 1, 2).reshape(1, -1, 64, 64)
        touches = np.stack([x[1] for x in self.history])[None]
        positions = np.stack([x[2] for x in self.history])[None]
        action = expand_action(self.actor(torch.as_tensor(images, device=self.device),
                                          torch.as_tensor(touches, device=self.device),
                                          torch.as_tensor(positions, device=self.device)))
        if not torch.isfinite(action).all():
            raise ValueError('Nonfinite policy output')
        return {'action': action[0].cpu().tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent device fallback')
    torch.set_num_threads(4)
    actor, payload = load_policy(args.checkpoint, args.device)
    session = ReactiveSession(actor, args.device)
    print(json.dumps({'ready': True, 'model_parameters': parameter_count(actor),
                      'kind': payload.get('kind'), 'forecasts': False}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('command') == 'quit':
                break
            print(json.dumps(session.request(request), allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({'error': type(error).__name__+': '+str(error)}), flush=True)


if __name__ == '__main__':
    main()
