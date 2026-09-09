"""Train/evaluate the small action-conditioned world model; no robot rollout claims."""
import argparse
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model import ModelConfig, SmallWorldModel, checkpoint_payload
from data import read_episodes, normalization_from_training, SequenceDataset, split_manifest


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def encode_sequence(model, batch):
    b, length = batch['rgb'].shape[:2]
    z = model.encode(batch['rgb'].flatten(0, 1), batch['tactile'].flatten(0, 1),
                     batch['proprio'].flatten(0, 1))
    return z.reshape(b, length, -1)


def supervised_loss(model, pred, batch):
    force = (batch['target_tactile']-model.tactile_mean)/model.tactile_std
    height = (batch['target_height']-model.height_mean)/model.height_std
    reward = (batch['target_reward']-model.reward_mean)/model.reward_std
    peak = (batch['target_peak']-model.peak_mean)/model.peak_std
    # Train current observer and future predictions; future positions cannot enter input.
    losses = {
        'force': F.smooth_l1_loss(pred['tactile_normalized'], force),
        'height': F.smooth_l1_loss(pred['height_normalized'], height),
        'peak': F.smooth_l1_loss(pred['peak_normalized'], peak),
        'support': F.binary_cross_entropy_with_logits(pred['support_logit'], batch['target_support']),
        'contact': F.binary_cross_entropy_with_logits(pred['contact_logit'], batch['target_contact']),
    }
    mask = batch['has_reward'][:, None, None]
    reward_error = F.smooth_l1_loss(pred['reward_normalized'], reward, reduction='none')
    losses['reward'] = (reward_error * mask).sum() / (mask.sum()*reward.shape[1]).clamp_min(1)
    latest = batch['rgb'][:, :, -3:].float().flatten(0, 1) / 255.
    visual_target = F.interpolate(latest, size=(16, 16), mode='area').reshape_as(pred['rgb16'])
    losses['image'] = F.mse_loss(pred['rgb16'], visual_target)
    total = losses['force'] + losses['height'] + losses['peak'] + .25*(losses['support']+losses['contact'])
    total = total + .5*losses['reward'] + .5*losses['image']
    return total, losses


@torch.no_grad()
def update_ema(target, model, rate=.01):
    for left, right in zip(target.parameters(), model.parameters()):
        left.lerp_(right, rate)


def do_epoch(model, target, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    target.eval()
    sums = {}
    samples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move(batch, device)
            z0 = model.encode(batch['rgb'][:, 0], batch['tactile'][:, 0], batch['proprio'][:, 0])
            pred = model.rollout(z0, batch['action'], include_initial=True)
            loss, pieces = supervised_loss(model, pred, batch)
            with torch.no_grad():
                target_z = encode_sequence(target, batch)
            latent_loss = F.mse_loss(pred['latent'][:, 1:], target_z[:, 1:])
            loss = loss + 20 * latent_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 20.)
                optimizer.step()
                update_ema(target, model)
            count = len(batch['rgb'])
            samples += count
            for key, value in {**pieces, 'latent': latent_loss, 'total': loss}.items():
                sums[key] = sums.get(key, 0.) + float(value.detach())*count
    return {key: value/max(samples, 1) for key, value in sums.items()}


@torch.no_grad()
def predictions(model, loader, device, persistence=False):
    model.eval()
    storage = {}
    for batch in loader:
        batch = move(batch, device)
        z0 = model.encode(batch['rgb'][:, 0], batch['tactile'][:, 0], batch['proprio'][:, 0])
        if persistence:
            z = z0[:, None].expand(-1, batch['action'].shape[1], -1)
            pred = model.decode(z)
            # Persistence of an available observed force, rather than an oracle future force.
            if model.config.variant == 'visuotactile':
                pred['tactile'] = batch['tactile'][:, 0, -1, None].expand(-1, z.shape[1], -1)
        else:
            pred = model.rollout(z0, batch['action'])
        records = {'episode_index': batch['episode_index'], 'start': batch['start']}
        for key in ('tactile', 'height', 'support', 'contact', 'reward', 'peak'):
            records['pred_'+key] = pred[key]
            records['true_'+key] = batch['target_'+key][:, 1:]
        latest = batch['rgb'][:, 1:, -3:].float().flatten(0, 1) / 255.
        target_rgb = F.interpolate(latest, (16,16), mode='area').reshape_as(pred['rgb16'])
        records['image_mse'] = (pred['rgb16']-target_rgb).square().mean((-1, -2, -3))
        for key, value in records.items():
            storage.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(value) for key, value in storage.items()} if storage else {}


def metrics(record):
    if not record:
        return {'windows': 0}
    ferr = np.abs(record['pred_tactile']-record['true_tactile'])
    herr = np.abs(record['pred_height']-record['true_height'])
    perr = np.abs(record['pred_peak']-record['true_peak'])
    active = record['true_contact'].max(-1) > .5
    metrics = {
        'windows': len(ferr), 'episodes': int(len(np.unique(record['episode_index']))),
        'force_mae_N': float(ferr.mean()), 'force_mae_components_N': ferr.mean((0,1)).tolist(),
        'height_mae_m': float(herr.mean()), 'rgb16_mse': float(record['image_mse'].mean()),
        'substep_peak_mae_N': float(perr.mean()),
        'substep_peak_mae_by_horizon_N': perr.mean((0,2)).tolist(),
        'substep_peak_underprediction_p95_N': float(np.quantile((record['true_peak']-record['pred_peak']).clip(0), .95)),
        'support_accuracy': float(((record['pred_support']>.5)==(record['true_support']>.5)).mean()),
        'contact_accuracy': float(((record['pred_contact']>.5)==(record['true_contact']>.5)).mean()),
        'reward_mae': float(np.abs(record['pred_reward']-record['true_reward']).mean()),
        'force_mae_by_horizon_N': ferr.mean((0,2)).tolist(),
        'height_mae_by_horizon_m': herr.mean((0,2)).tolist(),
        'contact_forecasts': int(active.sum()),
    }
    if active.any():
        metrics['contact_phase_force_mae_N'] = float(ferr[active].mean())
        metrics['contact_phase_height_mae_m'] = float(herr[active].mean())
    return metrics


def conformal_quantiles(record, alpha=.1):
    """Calibration unit = complete episode; finite-sample corrected rank.

    Bound covers evaluated horizon-5 windows under exchangeable episodes, not
    action optimization, a changed closed-loop policy, or an OOD test domain.
    """
    if not record:
        return {'available': False}
    ids = np.unique(record['episode_index'])
    # Safety refers to internal-step peaks, not end-of-control-step readings.
    # One-sided error suffices for an upper force envelope.
    f = (record['true_peak']-record['pred_peak']).clip(0)
    h = np.abs(record['pred_height']-record['true_height'])
    fscores = np.array([f[record['episode_index']==i].max() for i in ids])
    hscores = np.array([h[record['episode_index']==i].max() for i in ids])
    # Separate alpha/2 allocation makes the joint force+height failure <= alpha.
    rank = int(np.ceil((len(ids)+1)*(1-alpha/2)))
    if rank > len(ids):
        fq = hq = None  # mathematically +infinity; JSON null accompanied by explanation
    else:
        fq = float(np.sort(fscores)[rank-1])
        hq = float(np.sort(hscores)[rank-1])
    return {'available': True, 'alpha_joint': alpha, 'calibration_episodes': len(ids),
            'order_statistic_rank': rank, 'peak_radius_N': fq, 'height_radius_m': hq,
            'force_target': 'per-finger maximum normal force over each control interval physical substeps',
            'infinite_radius': rank > len(ids),
            'scope': 'simulator recording distribution; episode-exchangeable evaluated forecast windows only',
            'not_guaranteed': 'OOD domains, adaptive candidate selection, changed closed-loop policy'}


def evaluate_one(model, episodes, args, run_dir):
    results = {}
    available_splits = sorted(set(e['split'] for e in episodes) - {'train'})
    calibration = {}
    for split in available_splits:
        data = SequenceDataset(episodes, split, args.horizon, args.history,
                               stride=args.eval_stride, max_windows_per_episode=args.eval_windows)
        if not len(data):
            continue
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        record = predictions(model, loader, args.device)
        results[split] = {'world_model': metrics(record)}
        np.savez_compressed(run_dir/f'predictions_{split}.npz', **record)
        if split == 'calibration':
            calibration = conformal_quantiles(record, args.alpha)
        if split.startswith('test'):
            results[split]['persistence_no_dynamics'] = metrics(predictions(model, loader, args.device, True))
            if model.config.variant == 'visuotactile':
                negative = SequenceDataset(episodes, split, args.horizon, args.history,
                    stride=args.eval_stride, max_windows_per_episode=args.eval_windows, shuffle_tactile=True)
                nloader = DataLoader(negative, batch_size=args.batch_size, shuffle=False, num_workers=0)
                results[split]['shuffled_tactile_at_test'] = metrics(predictions(model, nloader, args.device))
    (run_dir/'evaluation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (run_dir/'calibration.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')
    return results, calibration


def train_one(episodes, normalization, variant, seed, args):
    seed_everything(seed)
    run_dir = Path(args.output)/f'{variant}_seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    first = next(e for e in episodes if e['split']=='train')
    config = ModelConfig(proprio_dim=first['proprio'].shape[-1], tactile_dim=first['tactile'].shape[-1],
                         action_dim=first['action'].shape[-1], history=args.history, variant=variant)
    model = SmallWorldModel(config, normalization).to(args.device)
    if model.num_parameters >= 2_000_000:
        raise ValueError('Small model parameter budget exceeded.')
    target = copy.deepcopy(model).eval()
    target.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_data = SequenceDataset(episodes, 'train', args.horizon, args.history, stride=args.train_stride)
    val_data = SequenceDataset(episodes, 'val', args.horizon, args.history,
                               stride=args.eval_stride, max_windows_per_episode=args.eval_windows)
    if not len(train_data) or not len(val_data):
        raise ValueError('Need nonempty train and val episode sequences; test cannot select models.')
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=0,
                              generator=generator, pin_memory=args.device.startswith('cuda'))
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=0)
    best = float('inf')
    beginning = time.perf_counter()
    log_path = run_dir/'training.jsonl'
    log_path.write_text('', encoding='utf-8')
    print(json.dumps({'event':'start','variant':variant,'seed':seed,'parameters':model.num_parameters,
                      'train_windows':len(train_data),'val_windows':len(val_data)}), flush=True)
    for epoch in range(1, args.epochs+1):
        train_stats = do_epoch(model, target, train_loader, args.device, optimizer)
        val_stats = do_epoch(model, target, val_loader, args.device)
        row = {'epoch':epoch,'train':train_stats,'val':val_stats,
               'elapsed_seconds':time.perf_counter()-beginning}
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps({'event':'epoch','variant':variant,'seed':seed, **row}), flush=True)
        # Select on physically supervised forecast loss, not an EMA-target lag effect.
        criterion = val_stats['total']-20*val_stats['latent']
        if criterion < best:
            best = criterion
            torch.save(checkpoint_payload(model, seed=seed, epoch=epoch, validation_loss=best,
                horizon=args.horizon, source='TD-MPC2 e9f5932 adapted layers, model-only',
                data_manifest=split_manifest(episodes), arguments=vars(args),
                num_parameters=model.num_parameters), run_dir/'best.pt')
    payload = torch.load(run_dir/'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(payload['model'])
    results, calibration = evaluate_one(model, episodes, args, run_dir)
    summary = {'variant':variant,'seed':seed,'parameters':model.num_parameters,
               'best_epoch':payload['epoch'],'training_seconds':time.perf_counter()-beginning,
               'evaluation':results,'calibration':calibration}
    (run_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variants', nargs='+', default=['vision','visuotactile'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--history', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train-stride', type=int, default=1)
    parser.add_argument('--eval-stride', type=int, default=5)
    parser.add_argument('--eval-windows', type=int, default=0)
    parser.add_argument('--calibration-fraction', type=float, default=.15)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing an unreported CPU fallback.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    episodes = read_episodes(args.data, args.calibration_fraction)
    norm = normalization_from_training(episodes)
    manifest = split_manifest(episodes)
    (output/'data_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output/'normalization.json').write_text(json.dumps(norm, indent=2), encoding='utf-8')
    summaries = []
    for seed in args.seeds:
        for variant in args.variants:
            summaries.append(train_one(episodes, norm, variant, seed, args))
            (output/'all_runs.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
