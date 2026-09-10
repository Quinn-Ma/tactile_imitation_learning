"""Scientific-contract checks; runs on synthetic episodes, never held-out data."""
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from reactive import (ReactiveActor, ReactiveConfig, expand_action, load_policy,
                      observed_reward, policy_payload, td_target)
from reactive_server import ReactiveSession
from train_reactive import (ObservationReplay, load_training_validation,
                            training_normalization)


def make_episode(path, split='train', value=1, noncontrolled=False, done_at=None):
    n = 8
    rng = np.random.default_rng(value)
    action = np.zeros((n, 7), np.float32)
    action[:, 2] = np.linspace(-.1, .3, n)
    action[:, 6] = np.linspace(-.2, 1., n)
    if noncontrolled:
        action[4, 0] = .1
    payload = dict(rgb=rng.integers(0, 255, (n+1, 64, 64, 3), dtype=np.uint8),
        tactile=np.full((n+1, 6), value, np.float32),
        proprio=np.full((n+1, 9), value*.1, np.float32), action=action,
        height=np.arange(n+1, dtype=np.float32)[:, None]*.02,
        substep_normal_peak=np.full((n+1, 2), value, np.float32),
        split=np.array(split), episode_id=np.array(path.stem))
    if done_at is not None:
        done = np.zeros(n, bool)
        done[done_at] = True
        payload['done'] = done
    np.savez_compressed(path, **payload)


class ReactiveContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        for name, split, value in [('a', 'train', 1), ('b', 'train', 3), ('c', 'val', 100)]:
            make_episode(self.root/f'{name}.npz', split, value)
            self.rows.append({'path': name+'.npz', 'split': split})
        self.rows.append({'path': 'heldout_must_not_be_opened.npz', 'split': 'test'})
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps(self.rows))

    def load(self, **kwargs):
        return load_training_validation(self.root, self.manifest, 2, 1, 3, 7, **kwargs)

    def test_heldout_is_unopened_and_normalization_is_train_only(self):
        real_load = np.load
        opened = []
        def checked(path, *args, **kwargs):
            opened.append(Path(path).name)
            self.assertNotIn('heldout', str(path))
            return real_load(path, *args, **kwargs)
        with mock.patch('train_reactive.np.load', side_effect=checked):
            episodes, _ = self.load()
        self.assertEqual(opened, ['a.npz', 'b.npz', 'c.npz'])
        norm = training_normalization(episodes['train'])
        np.testing.assert_allclose(norm['tactile_mean'], [2]*6)
        np.testing.assert_allclose(norm['tactile_std'], [1]*6)

    def test_mislabeled_heldout_and_noncausal_actions_are_rejected(self):
        make_episode(self.root/'a.npz', 'test', 1)
        with self.assertRaisesRegex(ValueError, 'disagrees with embedded'):
            self.load()
        make_episode(self.root/'a.npz', 'train', 1, noncontrolled=True)
        with self.assertRaisesRegex(ValueError, '2D causal Q'):
            self.load()

    def test_duplicate_train_validation_identity_is_rejected(self):
        # Different path and bytes cannot evade the episode-ID grouping check.
        with np.load(self.root/'c.npz') as src:
            payload = {k: src[k] for k in src.files}
        payload['episode_id'] = np.array('a')
        np.savez_compressed(self.root/'c.npz', **payload)
        with self.assertRaisesRegex(ValueError, 'Duplicate episode identity'):
            self.load()

    def test_real_next_reward_and_terminal_mask_alignment(self):
        make_episode(self.root/'a.npz', 'train', 1, done_at=4)
        episodes, _ = self.load()
        replay = ObservationReplay(episodes['train'], start_step=3)
        # A has transitions 3->4 and 4->5; B has 3->4,...,6->7.
        self.assertEqual(len(replay), 6)
        np.testing.assert_array_equal(replay.bootstrap.numpy(), [1, 0, 1, 1, 1, 0])
        self.assertEqual(replay.next[1].tolist(), [3, 4, 5])
        self.assertEqual(replay.current[2].tolist(), [10, 11, 12])
        # Terminal continuation is exactly suppressed, even for a huge V.
        target = td_target(replay.reward, replay.bootstrap, torch.full((6,), 1000.))
        self.assertEqual(float(target[1]), float(replay.reward[1]))
        self.assertEqual(float(target[-1]), float(replay.reward[-1]))
        expected_first = .08/.10-.005*float(replay.action[0].square().mean())
        self.assertAlmostEqual(float(replay.reward[0]), expected_first, places=6)
        measured = observed_reward(torch.tensor([[.05]]), torch.tensor([[12., 4.]]), torch.tensor([[.2, .4]]))
        self.assertAlmostEqual(float(measured[0]), .5-.125-.005*.1, places=6)

    def test_vision_invariance_action_subspace_and_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        actor = ReactiveActor(ReactiveConfig(variant='vision'), action_low=[-.1, -.2], action_high=[.3, 1.]).eval()
        obs = (torch.randint(0, 256, (3, 9, 64, 64), dtype=torch.uint8),
               torch.randn(3, 3, 6), torch.randn(3, 3, 9))
        with torch.no_grad():
            actions = actor(*obs)
            changed_touch = actor(obs[0], obs[1]+1000, obs[2])
            self.assertTrue(torch.equal(actions, changed_touch))
            full = expand_action(actions)
            self.assertTrue(torch.equal(full[:, [0, 1, 3, 4, 5]], torch.zeros(3, 5)))
            self.assertTrue(((actions >= actor.action_low) & (actions <= actor.action_high)).all())
            # Extreme head logits still respect bounds.
            actor.head[-1].bias.copy_(torch.tensor([1e4, -1e4]))
            extreme = actor(*obs)
            self.assertTrue(torch.equal(extreme[:, 0], actor.action_high[0].expand(3)))
            self.assertTrue(torch.equal(extreme[:, 1], actor.action_low[1].expand(3)))
        path = self.root/'policy.pt'
        torch.save(policy_payload(actor, kind='reactive_bc'), path)
        loaded, _ = load_policy(path)
        with torch.no_grad():
            self.assertTrue(torch.equal(actor(*obs), loaded(*obs)))

    def test_stdio_reset_protocol_returns_only_measured_input_policy_action(self):
        torch.manual_seed(1)
        actor = ReactiveActor().eval()
        path = self.root/'policy.pt'
        torch.save(policy_payload(actor, kind='reactive_bc'), path)
        request = {'rgb_base64': base64.b64encode(np.zeros((64, 64, 3), np.uint8).tobytes()).decode(),
                   'tactile': [1.]*6, 'proprio': [.1]*9}
        alternate = dict(request, tactile=[5.]*6)
        messages = [{'command': 'reset'}, request, alternate, {'command': 'reset'}, request, {'command': 'quit'}]
        proc = subprocess.run([sys.executable, str(Path(__file__).with_name('reactive_server.py')),
            '--checkpoint', str(path), '--device', 'cpu'], input='\n'.join(map(json.dumps, messages))+'\n',
            text=True, capture_output=True, timeout=60,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        output = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertTrue(output[0]['ready'])
        self.assertEqual(output[1], {'reset': True})
        self.assertEqual(output[2], output[5])
        self.assertEqual(set(output[2]), {'action'})
        self.assertEqual(len(output[2]['action']), 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
