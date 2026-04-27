"""
Stage 1: Train an imitation-learning policy for robot grasping.

Uses the LeRobot library (https://github.com/huggingface/lerobot).
Supports ACT and Diffusion Policy.

Dataset format (LeRobot v3.0)
------------------------------
Feature keys expected in the dataset:

  observation.state          float32 (state_dim,)   joint positions +
                                                     gripper aperture +
                                                     tactile/FT readings
  observation.images.wrist   video   (H, W, 3)      wrist RGB camera
  action                     float32 (action_dim,)   target joint positions /
                                                     gripper command

Quick start
-----------
# Train ACT
python scripts/train_il_policy.py \\
    --policy act \\
    --repo_id myorg/grasp_demos \\
    --output_dir outputs/act_grasp

# Train Diffusion Policy
python scripts/train_il_policy.py \\
    --policy diffusion \\
    --repo_id myorg/grasp_demos \\
    --output_dir outputs/diffusion_grasp

# Resume from checkpoint
python scripts/train_il_policy.py \\
    --resume --output_dir outputs/act_grasp
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def _build_policy_cfg(policy_name: str, ds_meta: LeRobotDatasetMetadata):
    features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features  = {k: v for k, v in features.items() if k not in output_features}

    if policy_name == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        return ACTConfig(input_features=input_features, output_features=output_features)
    else:
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        return DiffusionConfig(input_features=input_features, output_features=output_features)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    ds_meta = LeRobotDatasetMetadata(args.repo_id)
    policy_cfg = _build_policy_cfg(args.policy, ds_meta)

    # delta_timestamps: one frame per obs step; full action chunk
    fps = ds_meta.fps
    obs_delta = [i / fps for i in policy_cfg.observation_delta_indices or [0]]
    act_delta  = [i / fps for i in policy_cfg.action_delta_indices   or [0]]
    delta_timestamps: dict[str, list[float]] = {
        "action": act_delta,
        **{k: obs_delta for k in ds_meta.features if k.startswith("observation.")},
    }

    dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)

    drop_last_n = getattr(policy_cfg, "drop_n_last_frames", 0)
    sampler = EpisodeAwareSampler(
        dataset.episode_data_index,
        drop_n_last_frames=drop_last_n,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    dl_iter = cycle(loader)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, dataset_stats=dataset.stats
    )
    policy = make_policy(policy_cfg, dataset_stats=dataset.stats)
    policy = policy.to(device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )

    logger.info("Training for %d steps …", args.steps)
    for step in range(1, args.steps + 1):
        batch = next(dl_iter)
        batch = preprocessor(batch)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        loss, info = policy(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if step % args.log_freq == 0:
            logger.info("step=%06d  loss=%.4f  lr=%.2e",
                        step, loss.item(), scheduler.get_last_lr()[0])

        if step % args.save_freq == 0 or step == args.steps:
            ckpt = get_step_checkpoint_dir(output_dir, args.steps, step)
            save_checkpoint(ckpt, step, None, policy, optimizer, scheduler,
                            preprocessor, postprocessor)
            update_last_checkpoint(ckpt)
            logger.info("Checkpoint → %s", ckpt)

    logger.info("Done.  Best checkpoint → %s/last", output_dir)


def main():
    init_logging()
    parser = argparse.ArgumentParser(
        description="Train Stage-1 grasping IL policy (ACT or Diffusion) via LeRobot"
    )
    parser.add_argument("--policy", choices=["act", "diffusion"], default="act",
                        help="IL policy architecture")
    parser.add_argument("--repo_id", required=True,
                        help="LeRobot dataset repo-id or local root path")
    parser.add_argument("--output_dir", default="outputs/il_policy")
    parser.add_argument("--steps",    type=int,   default=80_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_freq", type=int,  default=10_000)
    parser.add_argument("--log_freq",  type=int,  default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
