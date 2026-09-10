"""Synthetic and protocol-only forecast audit; never reads new test outcomes."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import forecast_eval as fe


def fake_record():
    n, h = 60, 5
    record = {'episode_index': np.repeat([0, 1], 30), 'start': np.tile(np.arange(0, 146, 5), 2),
              'image_mse': np.zeros((n, h), np.float32)}
    for key, dim in [('tactile', 6), ('peak', 2), ('height', 1), ('contact', 2), ('support', 1), ('reward', 1)]:
        record['pred_'+key] = np.zeros((n, h, dim), np.float32)
        record['true_'+key] = np.zeros((n, h, dim), np.float32)
    record['true_tactile'][:30] = 1
    record['true_tactile'][30:] = 3
    record['true_peak'][:30] = 2
    record['true_peak'][30:] = 4
    record['true_height'][:30] = .01
    record['true_height'][30:] = .03
    record['true_contact'][:30] = 1
    record['true_contact'][30, 0] = 1
    return record


def fake_episode(seed):
    return {'rgb': np.zeros((151, 64, 64, 3), np.uint8),
        'tactile': np.full((151, 6), seed, np.float32), 'proprio': np.zeros((151, 9), np.float32),
        'action': np.zeros((150, 7), np.float32), 'height': np.zeros((151, 1), np.float32),
        'contact': np.zeros((151, 2), np.float32), 'support': np.zeros((151, 1), np.float32),
        'reward': np.zeros((151, 1), np.float32), 'peak': np.zeros((151, 2), np.float32),
        'has_reward': True, 'split': 'test_id', 'environment_seed': seed, 'episode_id': str(seed)}


class ForecastContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_compiled_cube_refines_only_conservative_width_bounds(self):
        expected={'mass':.12,'geometry':'legacy_cube','feasibility':{
            'feasible':True,'horizontal_width_bound_m':.0623,
            'required_aperture_m':.0713,'safety_ratio':8.15}}
        actual=copy.deepcopy(expected)
        actual['feasibility']['horizontal_width_bound_m']=.0607
        actual['feasibility']['required_aperture_m']=.0697
        self.assertTrue(fe.parameters_match(expected,actual))
        actual['mass']=.25
        self.assertFalse(fe.parameters_match(expected,actual))
        actual=copy.deepcopy(expected);actual['feasibility']['required_aperture_m']=.08
        self.assertFalse(fe.parameters_match(expected,actual))
        actual=copy.deepcopy(expected);actual['feasibility']['safety_ratio']=1.
        self.assertFalse(fe.parameters_match(expected,actual))

    def test_prospective_protocol_is_disjoint_before_trajectory_access(self):
        # This is design metadata only, containing no measured test outcomes.
        protocol = json.loads((fe.ROOT/'outputs/revision_v2/protocol.json').read_text())
        self.assertEqual(len(fe.validate_protocol(protocol)), 120)
        protocol['control_test'][0]['seed'] = protocol['collection'][0]['seed']
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            fe.validate_protocol(protocol)

    def test_precomputed_calibration_count_and_rank_are_enforced(self):
        calibration = {'available': True, 'calibration_episodes': 48, 'alpha_joint': .1,
                       'order_statistic_rank': 47, 'infinite_radius': False,
                       'peak_radius_N': 3., 'height_radius_m': .02}
        self.assertIs(fe.validate_calibration(calibration), calibration)
        for change in ({'calibration_episodes': 120}, {'order_statistic_rank': 48}, {'peak_radius_N': None}):
            with self.assertRaises(ValueError):
                fe.validate_calibration({**calibration, **change})

    def test_joint_coverage_counts_whole_episodes_and_peak_target(self):
        record = fake_record()
        episodes = [{'environment_seed': 101}, {'environment_seed': 102}]
        stats, original = fe.sufficient_statistics(record, episodes, {'peak_radius_N': 3., 'height_radius_m': .02})
        self.assertEqual(stats[101]['joint_episode_coverage'], (1, 1))
        self.assertEqual(stats[102]['joint_episode_coverage'], (0, 1))
        self.assertAlmostEqual(original['force_mae_N'], 2.)
        self.assertAlmostEqual(original['substep_peak_mae_N'], 3.)
        # One bad substep-peak forecast must fail the whole otherwise-good episode.
        record['true_peak'][0, 0, 0] = 3.001
        stats, _ = fe.sufficient_statistics(record, episodes, {'peak_radius_N': 3., 'height_radius_m': .02})
        self.assertEqual(stats[101]['joint_episode_coverage'], (0, 1))
        self.assertEqual(stats[101]['height_episode_coverage'], (1, 1))
        diagnostic, _ = fe.sufficient_statistics(record, episodes, None)
        self.assertNotIn('joint_episode_coverage', diagnostic[101])

    def test_bootstrap_keeps_fitted_seeds_together_and_weights_contact_counts(self):
        first = {101: {'error': (0., 1)}, 102: {'error': (10., 1)}}
        second = {101: {'error': (2., 1)}, 102: {'error': (12., 1)}}
        draws = np.array([[0, 0], [1, 1], [0, 1]])
        result = fe.grouped_environment_statistics([first, second], 'error', draws)
        self.assertEqual(result['environment_episodes'], 2)
        self.assertEqual(result['estimate'], 6.)
        np.testing.assert_allclose([result['ci_low'], result['ci_high']], np.quantile([1., 11., 6.], [.025, .975]))
        single = fe.grouped_environment_statistics([first], 'error', draws)
        duplicated = fe.grouped_environment_statistics([first, copy.deepcopy(first)], 'error', draws)
        self.assertEqual(single['ci_low'], duplicated['ci_low'])
        self.assertEqual(single['ci_high'], duplicated['ci_high'])
        conditional = {101: {'error': (2., 2)}, 102: {'error': (90., 10)}}
        result = fe.grouped_environment_statistics([conditional], 'error', draws)
        self.assertAlmostEqual(result['estimate'], 92/12)
        delta = fe.paired_difference(second, first, 'error')
        difference = fe.grouped_environment_statistics([delta], 'error', draws)
        self.assertEqual(difference['estimate'], 2.)
        self.assertEqual(difference['ci_low'], 2.)
        self.assertEqual(difference['ci_high'], 2.)

    def test_original_persistence_and_shuffling_preserve_target_semantics(self):
        from model import SmallWorldModel, ModelConfig
        model = SmallWorldModel(ModelConfig(variant='visuotactile')).eval()
        episodes = [fake_episode(1), fake_episode(2)]
        persistence, group = fe.forecast_records(model, episodes, 'test_id', 'cpu', 32, 'persistence')
        self.assertEqual(persistence['pred_tactile'].shape, (60, 5, 6))
        np.testing.assert_array_equal(persistence['pred_tactile'][:30], 1.)
        np.testing.assert_array_equal(persistence['pred_tactile'][30:], 2.)
        # Batched BLAS can round identical decoded rows at float32 precision.
        np.testing.assert_allclose(persistence['pred_peak'][:, 1:],
            np.repeat(persistence['pred_peak'][:, :1], 4, axis=1), rtol=1e-5, atol=1e-6)
        plain = fe.SequenceDataset(episodes, 'test_id', 5, 3, stride=5)
        shuffled = fe.SequenceDataset(episodes, 'test_id', 5, 3, stride=5, shuffle_tactile=True)
        a, b = plain[0], shuffled[0]
        self.assertTrue(torch.equal(b['tactile'], torch.full_like(b['tactile'], 2.)))
        for name in ('rgb', 'proprio', 'action', 'target_tactile', 'target_peak', 'target_height'):
            self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_cli_refuses_any_test_access_without_explicit_run_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, str(Path(fe.__file__)), '--protocol', 'nonexistent_protocol',
                '--trajectories', 'nonexistent/evaluations/script', '--out', str(Path(tmp)/'out')],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 2)
            self.assertIn('No test files opened', proc.stderr)
            self.assertFalse((Path(tmp)/'out').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
