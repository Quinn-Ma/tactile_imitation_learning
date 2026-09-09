"""Persistent local stdio bridge from a simulator Python to GPU policy inference.

Protocol: one JSON request per line; rgb_base64 is 64x64x3 uint8. The simulator
provides only available RGB, tactile and proprio observations. No external server.
Returned forecasts correspond to the complete returned seven-dimensional action:
only vertical motion and gripper rate are nonzero. predicted_peak contains the
two per-finger normal-force peaks in Newtons over that next control interval;
these are raw model forecasts, not calibrated upper bounds or measured outcomes.
"""
import argparse
import base64
from collections import deque
import json
import sys

import numpy as np
import torch
from model import load_model
from train_imagination_rl import Actor, expand_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--actor', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    model, _ = load_model(args.model, args.device)
    payload = torch.load(args.actor, map_location=args.device, weights_only=False)
    actor = Actor(model.config.latent_dim).to(args.device).eval()
    actor.load_state_dict(payload['actor'])
    history = deque(maxlen=model.config.history)
    print(json.dumps({'ready': True, 'model_parameters': model.num_parameters}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('command') == 'quit':
                break
            if request.get('command') == 'reset':
                history.clear()
                print(json.dumps({'reset': True}), flush=True)
                continue
            if request.get('reset'):
                history.clear()
            if 'rgb_base64' in request:
                rgb = np.frombuffer(base64.b64decode(request['rgb_base64']), np.uint8).reshape(64, 64, 3)
            else:
                rgb = np.asarray(request['rgb'], dtype=np.uint8)
            tactile = np.asarray(request['tactile'], dtype=np.float32)
            proprio = np.asarray(request['proprio'], dtype=np.float32)
            entry = (rgb, tactile, proprio)
            history.append(entry)
            while len(history) < model.config.history:
                history.appendleft(entry)
            images = np.stack([x[0] for x in history]).transpose(0, 3, 1, 2).reshape(1, -1, 64, 64)
            touches = np.stack([x[1] for x in history])[None]
            positions = np.stack([x[2] for x in history])[None]
            with torch.no_grad():
                z = model.encode(torch.as_tensor(images, device=args.device),
                                 torch.as_tensor(touches, device=args.device),
                                 torch.as_tensor(positions, device=args.device))
                action = expand_action(actor(z))
                pred = model.decode(model.next(z, action))
            result = {'action': action[0].cpu().tolist(),
                      'predicted_tactile': pred['tactile'][0].cpu().tolist(),
                      'predicted_peak': pred['peak'][0].cpu().tolist(),
                      'predicted_height': float(pred['height'][0, 0].cpu()),
                      'predicted_reward': float(pred['reward'][0, 0].cpu())}
            print(json.dumps(result), flush=True)
        except Exception as error:
            print(json.dumps({'error': type(error).__name__ + ': ' + str(error)}), flush=True)


if __name__ == '__main__':
    main()
